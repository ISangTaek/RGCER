"""P1-D optimization primitives; not a training authorization or batch runner.

No data/test access. Defaults of the historical training engines remain intact.
"""
from dataclasses import asdict, dataclass
import math

import torch


class OptimizationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise OptimizationError(message)


ARMS = {
    'B1_high': (0.001, 0),
    'B1_low': (0.0001, 0),
    'HF_high': (0.001, 5),
    'HF_low': (0.0001, 5),
}


@dataclass(frozen=True)
class OptimizationSpec:
    arm: str

    def __post_init__(self):
        require(type(self.arm) is str and self.arm in ARMS, 'unknown optimization arm')

    @property
    def backbone_lr(self):
        return ARMS[self.arm][0]

    @property
    def warmup_epochs(self):
        return ARMS[self.arm][1]

    def identity(self):
        return dict(**asdict(self), backbone_lr=self.backbone_lr, head_lr=0.001,
                    warmup_epochs=self.warmup_epochs, weight_decay=1e-5,
                    scheduler=None, head_optimizer_reset=False,
                    freeze_mode='requires_grad_only', schema='p1d_optimization_v1')


class OptimizationControl:
    """Same model tensors, continuous head Adam, no frozen backbone state.

    Explicitly restricted to a parameterless readout Graphormer encoder plus
    decoders. Do not silently classify a new readout or auxiliary module as head.
    """
    def __init__(self, model, spec):
        require(type(spec) is OptimizationSpec, 'optimization spec type')
        self.model, self.spec = model, spec
        named = list(model.named_parameters())
        require(named and all(p.requires_grad for _, p in named), 'must start from unfrozen B1')
        require(all(n.startswith(('encoder.backbone.', 'decoders.')) for n, _ in named),
                'unexpected parameter scope; parameterized readout is not approved')
        self.backbone = [(n, p) for n, p in named if n.startswith('encoder.backbone.')]
        self.heads = [(n, p) for n, p in named if n.startswith('decoders.')]
        require(self.backbone and self.heads, 'empty backbone/head parameter scope')
        # Preserve the historical single-group ordering for the original arm.
        if spec.arm == 'B1_high':
            parameters = [p for _, p in named]
        else:
            parameters = [dict(params=[p for _, p in self.backbone], lr=spec.backbone_lr),
                          dict(params=[p for _, p in self.heads], lr=.001)]
        self.optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=1e-5)
        self.epoch = None
        self.initial_backbone = {n: p.detach().cpu().clone() for n, p in self.backbone}
        self._expected_groups = self._group_contract(self.optimizer.state_dict())

    def _group_contract(self, state):
        return [{k: v for k, v in g.items()} for g in state['param_groups']]

    def begin_epoch(self, epoch):
        require(type(epoch) is int and 0 <= epoch < 40, 'epoch outside approved range')
        require(self.epoch is None or epoch == self.epoch + 1, 'nonsequential epoch')
        self.epoch = epoch
        frozen = epoch < self.spec.warmup_epochs
        for _, p in self.backbone:
            p.requires_grad_(not frozen)
            if frozen:
                p.grad = None
        require(all(p.requires_grad for _, p in self.heads), 'head unexpectedly frozen')
        # Intentionally no train()/eval(), seeding, optimizer reset or LR update.
        self.verify_frozen()

    def verify_frozen(self):
        require(self.epoch is not None, 'epoch not initialized')
        require(self._group_contract(self.optimizer.state_dict()) == self._expected_groups,
                'optimizer group contract changed')
        require(all(p.requires_grad for _, p in self.heads), 'head unexpectedly frozen')
        if self.epoch < self.spec.warmup_epochs:
            for n, p in self.backbone:
                require(not p.requires_grad and p.grad is None, 'frozen gradient leak')
                require(p not in self.optimizer.state or not self.optimizer.state[p], 'frozen Adam state exists')
                require(torch.equal(p.detach().cpu(), self.initial_backbone[n]), 'frozen tensor changed')
        else:
            require(all(p.requires_grad for _, p in self.backbone), 'backbone was not unfrozen')

    def validate_optimizer_state(self, state, completed_epoch, model_state=None):
        require(type(completed_epoch) is int and 0 <= completed_epoch < 40, 'resume epoch')
        require(type(state) is dict and set(state) == {'state', 'param_groups'}, 'optimizer state schema')
        require(self._group_contract(state) == self._expected_groups, 'optimizer group contract changed')
        groups = state['param_groups']
        ids = [i for g in groups for i in g['params']]
        require(set(state['state']).issubset(ids), 'unknown optimizer parameter')
        ordered = (list(self.model.named_parameters()) if self.spec.arm == 'B1_high'
                   else self.backbone + self.heads)
        parameters = {i: p for i, (_, p) in zip(ids, ordered)}
        if self.spec.arm != 'B1_high' and completed_epoch < self.spec.warmup_epochs:
            require(not (set(groups[0]['params']) & set(state['state'])), 'frozen Adam state exists')
            if model_state is not None:
                require(all(n in model_state and torch.equal(model_state[n].detach().cpu(), initial)
                            for n, initial in self.initial_backbone.items()), 'restored frozen tensor changed')
        for key, item in state['state'].items():
            require(set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam state fields')
            require(all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in item.values()),
                    'nonfinite Adam state')
            step = item['step']
            require(step.numel() == 1 and float(step) > 0 and float(step).is_integer(), 'Adam step')
            require(all(item[k].shape == parameters[key].shape and item[k].dtype == parameters[key].dtype
                        for k in ('exp_avg','exp_avg_sq')), 'Adam moment shape/dtype')

    def record(self):
        self.verify_frozen()
        return dict(epoch=self.epoch, frozen=self.epoch < self.spec.warmup_epochs,
                    backbone_lr=self.spec.backbone_lr, head_lr=.001,
                    backbone_parameters=sum(p.numel() for _, p in self.backbone),
                    head_parameters=sum(p.numel() for _, p in self.heads),
                    backbone_optimizer_parameters=sum(bool(self.optimizer.state.get(p)) for _, p in self.backbone),
                    head_optimizer_steps={n: int(self.optimizer.state[p]['step'])
                                          if self.optimizer.state.get(p) else 0 for n, p in self.heads})


def select_arms(histories, *, setting, seed, task_names):
    """Validate all four complete validation histories before any selection.

    Inputs must already be bound to trusted raw predictions/data by the caller.
    This recomputes macro from endpoint RMSE, not a report's supplied macro/PASS.
    """
    require(setting in ('ToxAcute', 'A', 'B') and type(seed) is int and seed == 42, 'screen identity')
    require(type(task_names) in (list, tuple) and len(task_names) == (3 if setting == 'ToxAcute' else 5)
            and all(type(t) is str and t for t in task_names) and len(set(task_names)) == len(task_names), 'task set')
    require(type(histories) is dict and set(histories) == set(ARMS), 'complete four-arm screen required')
    scores = {}; reference_counts = None
    for arm, history in histories.items():
        require(type(history) is list and len(history) == 40, 'complete 40-epoch history required')
        values = []
        for epoch, row in enumerate(history):
            require(type(row['epoch']) is int and row['epoch'] == epoch and row['split'] == 'validation', 'epoch/split')
            endpoints = row['endpoints']
            require(set(endpoints) == set(task_names), 'missing/extra endpoint')
            errors = []
            for t in task_names:
                r = endpoints[t]
                require(type(r['n']) is int and r['n'] > 0, 'invalid observation count')
                v = r['rmse']
                require(type(v) in (int, float) and math.isfinite(v) and v >= 0, 'invalid endpoint metric')
                errors.append(v)
            counts = tuple(endpoints[t]['n'] for t in task_names)
            if reference_counts is None:
                reference_counts = counts
            require(counts == reference_counts, 'validation population counts changed')
            values.append(math.fsum(errors) / len(errors))
        best = min(range(40), key=values.__getitem__)
        scores[arm] = dict(best_epoch=best, best_validation_macro_rmse=values[best])
    winners = {family: min((family+'_high', family+'_low'),
                          key=lambda a: scores[a]['best_validation_macro_rmse']) for family in ('B1', 'HF')}
    return dict(setting=setting, seed=seed, scores=scores, selected=winners,
                scope='DEVELOPMENT_VALIDATION_ONLY', acceptance_status='PENDING_REVIEW')
