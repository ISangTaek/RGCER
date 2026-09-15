"""Plain Graphormer Human5 adaptation factory (not a checkpoint authorizer).

The caller must resolve/validate same-seed source assets before passing the
encoder state. This module never reads old Human3 heads or prediction files.
"""
import torch
from torch import nn

from architecture.Graphormer import Encoder, Graphormer
from architecture.prediction_heads import TaskPredictionHead
from dataset115_contract import ContractError, PRIMARY


def build_human5_model(args, *, method, seed, source_encoder_state=None):
    if method not in {'B0', 'B1', 'RPT'} or type(seed) is not int or seed < 0:
        raise ContractError('invalid adaptation method/seed')
    if (method == 'B0') != (source_encoder_state is None):
        raise ContractError('B0 forbids source; B1/RPT require source encoder')
    # Construct on CPU with isolated RNG. Same seed -> identical new heads for
    # all methods regardless of task count or randomness in source pretraining.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        decoders = nn.ModuleDict({t: TaskPredictionHead(args.hidden_dim,
            mode='quantile', head_hidden_dim=args.head_hidden_dim,
            dropout=args.head_dropout) for t in PRIMARY})
        model = Graphormer(list(PRIMARY), Encoder, decoders, torch.device('cpu'), args)
    if getattr(model, 'card', None) is not None:
        raise ContractError('external plain Graphormer forbids CARD')
    if source_encoder_state is not None:
        expected = model.encoder.state_dict()
        if set(source_encoder_state) != set(expected):
            raise ContractError('source encoder keys differ')
        for key, value in source_encoder_state.items():
            target = expected[key]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype or not torch.isfinite(value).all()):
                raise ContractError(f'source encoder tensor differs: {key}')
        model.encoder.load_state_dict(source_encoder_state, strict=True)
    # Match existing S1 semantics: freeze gradients only, retain train-mode
    # dropout. The runner must not silently change this to encoder.eval().
    for p in model.encoder.parameters():
        p.requires_grad_(method != 'RPT')
    return model
