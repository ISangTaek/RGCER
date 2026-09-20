"""Bounded P1D Route A/B factories; no CLI or experiment authorization.

The caller supplies reviewed expectations, never a run's self-declared identity.
``make_trainer(original=True)`` exposes the historical B1 engine for paired
smoke. The default is controlled B1_high, with exactly the same initialization.
``SingleRunAdapter.run`` is an engine API, not permission to launch formal work;
the parent owns smoke acceptance, scheduling, GPU selection and budget gates.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from dataset115_adapter import Dataset115Table
from dataset115_contract import PRIMARY, semantic_digest
from dataset115_route_a_smoke import ARCH
from dataset115_route_a_training import RouteATrainer, SourceTrainer, load_source, read_json
from dataset115_route_b_run import CONFIG, verify_training_run
from dataset115_smoke import state_digest
from dataset115_source import load_route_b_encoder
from dataset115_training import RouteBTrainer, require
from p1d_optimization import OptimizationSpec


CONFIGURATION_FIELDS = (
    'input_identity', 'source_identity', 'train_ids', 'validation_ids',
    'train_observations', 'validation_observations', 'task_names', 'scaler',
    'architecture',
)
EXPECTED_FIELDS = CONFIGURATION_FIELDS + ('initial_encoder', 'initial_heads')


def _sha(value, name):
    require(type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), name + ' SHA256')


def _exact(actual, expected, name):
    """Python equality alone accepts True == 1, including in nested contracts."""
    require(type(actual) is type(expected), name + ' type')
    if type(actual) is dict:
        require(actual.keys() == expected.keys(), name + ' fields')
        for key in actual:
            _exact(actual[key], expected[key], name + '.' + key)
    elif type(actual) in (list, tuple):
        require(len(actual) == len(expected), name + ' length')
        for a, e in zip(actual, expected):
            _exact(a, e, name)
    else:
        require(actual == expected, name + ' differs')


def _validate_expected(expected):
    require(type(expected) is dict and set(expected) == set(EXPECTED_FIELDS),
            'expected_identity fields')
    for key in ('input_identity', 'train_ids', 'validation_ids',
                'train_observations', 'validation_observations',
                'initial_encoder', 'initial_heads'):
        _sha(expected[key], key)
    _exact(expected['task_names'], list(PRIMARY), 'task_names')
    for key in ('source_identity', 'scaler', 'architecture'):
        require(type(expected[key]) is dict and bool(expected[key]), key + ' mapping')
    # Also rejects NaN, infinity and values that cannot be serialized as identity.
    semantic_digest(expected)


def expected_identity_from_record(configuration, *, initial_encoder, initial_heads):
    """Extract only reviewed fields from an optimization_assets configuration.

    Initialization digests must come from the corresponding reviewed payload
    summary. Extra configuration fields are deliberately not copied; the result
    itself must have exactly EXPECTED_FIELDS when supplied to the factory.
    """
    require(type(configuration) is dict
            and set(CONFIGURATION_FIELDS).issubset(configuration), 'configuration fields')
    result = {k: deepcopy(configuration[k]) for k in CONFIGURATION_FIELDS}
    result.update(initial_encoder=initial_encoder, initial_heads=initial_heads)
    _validate_expected(result)
    return result


class RouteFactory:
    """Use an audited Dataset115Table and reverify existing source on every build.

    Route B needs source_repo/source_lock. Route A needs source_output; its
    architecture comes from the existing frozen ARCH and its verifier reads only
    the source TRAIN view. SourceTrainer.run is never called.
    """
    def __init__(self, table, *, route, seed, expected_identity,
                 source_repo=None, source_lock=None, source_output=None):
        require(isinstance(table, Dataset115Table), 'audited Dataset115Table required')
        require(type(route) is str and route in ('A', 'B'), 'route scope')
        require(type(seed) is int and seed in range(42, 47), 'seed scope')
        _validate_expected(expected_identity)
        source = expected_identity['source_identity']
        require(type(source.get('seed')) is int and source['seed'] == seed, 'source seed')
        keys = {'seed', 'teacher_sha256', 'init_sha256'}
        if route == 'A':
            keys |= {'route', 'selection', 'source_identity_sha256'}
            require(source.get('route') == 'A'
                    and source.get('selection') == 'fixed_final_epoch39', 'Route A source binding')
            _sha(source.get('source_identity_sha256'), 'source identity')
        require(set(source) == keys, 'source_identity fields')
        for key in ('teacher_sha256', 'init_sha256'):
            _sha(source[key], key)
        require((route == 'A' and source_output is not None
                 and source_repo is None and source_lock is None)
                or (route == 'B' and source_output is None
                    and source_repo is not None and source_lock is not None), 'source paths for route')
        self._table, self._route, self._seed = table, route, seed
        self._expected = deepcopy(expected_identity)
        self._repo = None if source_repo is None else Path(source_repo)
        self._lock = None if source_lock is None else Path(source_lock)
        self._output = None if source_output is None else Path(source_output)
        _exact(table.identity, self._expected['input_identity'], 'input_identity')

    @classmethod
    def from_files(cls, *, csv_path, split_path, tox_path, expected_tox_sha, **kwargs):
        """Production convenience loader; paths and external expectations are required."""
        _sha(expected_tox_sha, 'tox manifest')
        table = Dataset115Table.load(csv_path, split_path, tox_path,
                                     expected_tox_sha=expected_tox_sha)
        return cls(table, **kwargs)

    @property
    def expected_identity(self):
        return deepcopy(self._expected)

    def make_trainer(self, *, original=False, arm='B1_high', device='cpu'):
        require(type(original) is bool, 'original must be boolean')
        spec = OptimizationSpec(arm)
        require(not original or arm == 'B1_high', 'original requires B1_high')
        require(type(device) is str and device in ('cpu', 'cuda:0'), 'device scope')
        table, route, seed = self._table, self._route, self._seed
        _exact(table.identity, self._expected['input_identity'], 'input_identity')
        train = table.view(route, 'target', 'train')
        validation = table.view(route, 'target', 'validation')
        # Check raw population before any source loading or model construction.
        for split, view in (('train', train), ('validation', validation)):
            _exact(view.input_identity, self._expected['input_identity'], 'input_identity')
            _exact(semantic_digest(list(view.sample_ids)), self._expected[split+'_ids'], split+'_ids')
            _exact(RouteBTrainer._view_digest(view), self._expected[split+'_observations'], split+'_observations')
        if route == 'B':
            encoder, args, receipt = load_route_b_encoder(self._repo, self._lock, seed)
            source = dict(seed=seed, teacher_sha256=receipt['teacher']['sha256'],
                          init_sha256=receipt['init']['sha256'])
            trainer_type = RouteBTrainer
        else:
            args = SimpleNamespace(**deepcopy(ARCH))
            source_trainer = SourceTrainer(args, table.view('A', 'source', 'train'),
                                           seed=seed, config=CONFIG, device=device)
            encoder, source, _ = load_source(self._output, source_trainer)
            trainer_type = RouteATrainer
        _exact(source, self._expected['source_identity'], 'source_identity')
        _exact(vars(args), self._expected['architecture'], 'architecture')
        _exact(state_digest(encoder), self._expected['initial_encoder'], 'initial_encoder')
        trainer = trainer_type(args, encoder, source, train, validation, method='B1',
                               seed=seed, config=CONFIG, device=device,
                               optimization=None if original else spec)
        for key in CONFIGURATION_FIELDS:
            _exact(trainer.identity[key], self._expected[key], key)
        _exact(trainer.initial_encoder, self._expected['initial_encoder'], 'initial_encoder')
        _exact(trainer.initial_heads, self._expected['initial_heads'], 'initial_heads')
        _exact(asdict(trainer.config), asdict(CONFIG), 'fixed CONFIG')
        return trainer

    def original_trainer(self, *, device='cpu'):
        return self.make_trainer(original=True, device=device)

    def controlled_trainer(self, *, arm='B1_high', device='cpu'):
        return self.make_trainer(arm=arm, device=device)


def _verify_optimization(output, trainer):
    """Supplement the legacy verifier with actual controlled Adam/freeze checks."""
    control = trainer.optimization
    counts = {t: math.ceil(len(ds)/CONFIG.batch_size)
              for t, ds in trainer.datasets['train'].items()}
    per_epoch = sum(counts.values())
    named = (list(trainer.model.named_parameters()) if control.spec.arm == 'B1_high'
             else control.backbone + control.heads)
    ids = [i for g in control.optimizer.state_dict()['param_groups'] for i in g['params']]
    mode = trainer.model.encoder.backbone.edge_bias_mode
    inactive_prefix = {'path': 'encoder.backbone.direct_bond_embeddings.',
                       'direct': 'encoder.backbone.path_bond_embeddings.',
                       'direct_plus_path': None}[mode]
    for epoch in range(CONFIG.epochs):
        path = Path(output)/f'epoch_{epoch:03d}.pt'
        raw = path.read_bytes()
        receipt = read_json(Path(output)/f'epoch_{epoch:03d}.receipt.json')
        require(hashlib.sha256(raw).hexdigest() == receipt['sha256'], 'optimization checkpoint SHA')
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        _exact(payload['epoch'], epoch, 'checkpoint epoch')
        _exact(payload['identity'], trainer.identity, 'checkpoint identity')
        opt = payload['optimizer_state']
        control.validate_optimizer_state(opt, epoch, payload['model_state'])
        head_steps = {}
        backbone_states = 0
        for index, (name, _) in zip(ids, named):
            is_head = name.startswith('decoders.')
            steps = (counts[name.split('.')[1]]*(epoch+1) if is_head
                     else per_epoch*max(0, epoch+1-control.spec.warmup_epochs))
            # Both embedding families are registered, but the frozen edge mode
            # deliberately leaves one family disconnected from autograd.
            if inactive_prefix is not None and name.startswith(inactive_prefix):
                steps = 0
                require(torch.equal(payload['model_state'][name], control.initial_backbone[name]),
                        'inactive edge embedding changed')
            item = opt['state'].get(index)
            if is_head:
                head_steps[name] = steps
            if steps == 0:
                require(item is None, 'frozen Adam state exists')
            else:
                require(item is not None and float(item['step']) == steps,
                        'Adam update count: ' + name)
                backbone_states += int(not is_head)
        expected = dict(epoch=epoch, frozen=epoch < control.spec.warmup_epochs,
                        backbone_lr=control.spec.backbone_lr, head_lr=.001,
                        backbone_parameters=sum(p.numel() for _, p in control.backbone),
                        head_parameters=sum(p.numel() for _, p in control.heads),
                        backbone_optimizer_parameters=backbone_states,
                        head_optimizer_steps=head_steps)
        _exact(payload['history'][epoch]['optimization'], expected, 'optimization record')


class SingleRunAdapter:
    """One fresh 40-epoch controlled target run; no resume, retry or budget override.

    Use factory trainers directly for parent-controlled bounded smoke. Calling
    this adapter requires separate parent authorization for a full formal run.
    Verification constructs a fresh trainer and revalidates the source assets.
    """
    def __init__(self, factory, *, arm='B1_high', device='cpu'):
        require(type(factory) is RouteFactory, 'RouteFactory required')
        OptimizationSpec(arm)
        require(type(device) is str and device in ('cpu', 'cuda:0'), 'device scope')
        self._factory, self._arm, self._device = factory, arm, device
        self._used = False

    def verify(self, output):
        fresh = self._factory.controlled_trainer(arm=self._arm, device=self._device)
        # Retain the legacy input/prediction/selection/tensor verifier unchanged.
        receipt = verify_training_run(output, fresh)
        _verify_optimization(output, fresh)
        return dict(receipt, task_id='P1D_ROUTE_'+self._factory._route,
                    route=self._factory._route, arm=self._arm,
                    optimization_checked=True, scope='TRAIN_AND_VALIDATION_ONLY',
                    calibration_predictions_accessed=False)

    def run(self, output):
        require(not self._used, 'single-run adapter already used')
        require(not Path(output).exists(), 'output exists')
        # Consume the attempt even if loading/training/verification fails.
        self._used = True
        trainer = self._factory.controlled_trainer(arm=self._arm, device=self._device)
        trainer.run(output)
        del trainer
        return self.verify(output)
