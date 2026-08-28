import warnings

import torch


_RGCER_ROUTER_FLAGS = (
    'rgcer_use_source_response',
    'rgcer_use_target_response',
    'rgcer_use_molecule_query',
    'rgcer_use_sparse_routing',
    'rgcer_use_null_route',
    'router_top_k',
    'router_temperature',
)


def resolve_effective_rgcer_config(params):
    """Separate requested flags from modules active for this mechanism."""

    names = (
        'rgcer_use_source_response',
        'rgcer_use_target_response',
        'rgcer_use_molecule_query',
        'rgcer_use_sparse_routing',
        'rgcer_use_null_route',
        'rgcer_use_film',
        'rgcer_use_adapter',
        'rgcer_use_base_aux_loss',
        'rgcer_fallback_space',
        'rgcer_transfer_mechanism',
        'rgcer_source_policy',
        'router_top_k',
        'router_temperature',
    )
    requested = {name: getattr(params, name, None) for name in names}
    mechanism = requested['rgcer_transfer_mechanism'] or 'endpoint_router'
    effective = dict(requested)
    ignored = []
    if getattr(params, 'arch', None) != 'Graphormer_rgcer':
        ignored = [name for name in names if name != 'rgcer_transfer_mechanism']
        effective = {
            'architecture': getattr(params, 'arch', None),
            'transfer_mechanism': 'hps',
            'routing_enabled': False,
        }
    elif mechanism in {'target_only', 'response_stacking'}:
        for name in _RGCER_ROUTER_FLAGS:
            if name in requested:
                ignored.append(name)
                effective[name] = None
        if mechanism in {'target_only', 'response_stacking'}:
            ignored.append('rgcer_fallback_space')
            effective['rgcer_fallback_space'] = None
    effective['architecture'] = getattr(params, 'arch', None)
    effective['transfer_mechanism'] = 'hps' if getattr(params, 'arch', None) != 'Graphormer_rgcer' else mechanism
    effective['routing_enabled'] = bool(getattr(params, 'routing_enabled', True))
    for name in ignored:
        if requested.get(name) not in (None, True, 'endpoint_router', 0, 1.0, 'animal56_only'):
            warnings.warn(
                f"{name} is ignored by architecture/mechanism {getattr(params, 'arch', None)}/{mechanism}.",
                RuntimeWarning,
                stacklevel=2,
            )
    return {'requested': requested, 'effective': effective, 'ignored': ignored}

def prepare_args(params):
    r"""Return the configuration of hyperparameters, optimizier, and learning rate scheduler.

    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    kwargs = {'weight_args': {}, 'arch_args': {}, 'sample_args': {}}

    # Populate arch_args for model architecture parameters
    # These should align with what your model's __init__ expects from arch_kwargs
    # For Graphormer-like architectures, common args might include:
    # num_layers, num_heads, hidden_dim, etc.
    # We'll take the ones defined in your main.py argparse
    if hasattr(params, 'a_layers'):
        kwargs['arch_args']['a_layers'] = params.a_layers
    if hasattr(params, 'a_heads'):
        kwargs['arch_args']['a_heads'] = params.a_heads
    # if hasattr(params, 'm_layers'): # Example if you have m_layers
    #     kwargs['arch_args']['m_layers'] = params.m_layers
    if hasattr(params, 't_layers'):
        kwargs['arch_args']['t_layers'] = params.t_layers
    if hasattr(params, 't_heads'):
        kwargs['arch_args']['t_heads'] = params.t_heads
    if hasattr(params, 'hidden_dim'):
        kwargs['arch_args']['hidden_dim'] = params.hidden_dim
    if hasattr(params, 'mid_dim'):
        kwargs['arch_args']['mid_dim'] = params.mid_dim
    if hasattr(params, 'spatial_pos_clip'):
        kwargs['arch_args']['spatial_pos_max_clip'] = params.spatial_pos_clip
    for name in [
        'use_factorized_prompt',
        'task_residual_scale',
        'prompt_layers',
        'prompt_heads',
        'prompt_ffn_dim',
        'prompt_dropout',
        'router_mode',
        'router_dim',
        'router_top_k',
        'router_temperature',
        'exclude_target_from_sources',
        'adapter_ratio',
        'prompt_gate_init',
        'response_hidden_dim',
        'edge_bias_mode',
        'prediction_mode',
        'head_hidden_dim',
        'head_dropout',
        'rgcer_use_source_response',
        'rgcer_use_target_response',
        'rgcer_use_molecule_query',
        'rgcer_use_sparse_routing',
        'rgcer_use_null_route',
        'rgcer_use_film',
        'rgcer_use_adapter',
        'rgcer_use_base_aux_loss',
        'rgcer_fallback_space',
        'rgcer_transfer_mechanism',
        'rgcer_source_policy',
        'spatial_pos_max_clip',
        'auxiliary_metadata_overrides',
    ]:
        if hasattr(params, name):
            kwargs['arch_args'][name] = getattr(params, name)
    # Add any other architecture-specific parameters from params here
    # e.g., dropout_rate, attention_dropout_rate if they are in params

    if params.weighting in ['EW', 'UW', 'DWA']:
        pass
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting))

    if params.optim in ['adam', 'adamw']:
        optim_param = {'optim': params.optim,
                       'lr': params.lr,
                       'weight_decay': params.weight_decay}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))

    display_args(params, kwargs, optim_param)

    return kwargs, optim_param

def display_args(params, kwargs, optim_param):

    print('='*40)
    print('General Configuration:')
    print('\tMode:', params.mode)
    print('\tWighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tDataset:', params.dataset)
    print('\tSplitting:', params.splitting)
    print('\tBatch Size:', params.bs)
    print('\tSave Path:', params.save_path)
    print('\tLoad Path:', params.load_path)
    print('\tDevice: {}'.format(f'cuda:{params.gpu_id}' if torch.cuda.is_available() and params.gpu_id != 'cpu' else 'cpu'))

    # Display new data-related parameters
    if hasattr(params, 'preprocessed_data_dir'):
        print('\tPreprocessed Data Dir:', params.preprocessed_data_dir)
    if hasattr(params, 'num_loader_workers'):
        print('\tNum Loader Workers:', params.num_loader_workers)
    if hasattr(params, 'spatial_pos_clip'):
        print('\tSpatial Pos Clip for Collator:', params.spatial_pos_clip)
    if hasattr(params, 'max_nodes_filter') and params.max_nodes_filter is not None:
        print('\tMax Nodes Filter for Collator:', params.max_nodes_filter)


    # Display arch_args if populated
    if kwargs.get('arch_args'): # Check if arch_args has content
        print('Architecture Configuration (arch_args):')
        for k, v in kwargs['arch_args'].items():
            print(f'\t{k}: {v}')
    # Keep other kwargs display if they are used (weight_args, sample_args)
    # for wa_key, p_name in zip(['weight_args', 'sample_args'],
    #                           [params.weighting, 'SampleMethod']): # Adjust if needed
    #     if kwargs[wa_key]:
    #         print('{} Configuration:'.format(p_name))
    #         for k, v in kwargs[wa_key].items():
    #             print('\t'+k+':', v)

    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
    print('='*40)
