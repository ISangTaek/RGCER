import torch

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
    # Add any other architecture-specific parameters from params here
    # e.g., dropout_rate, attention_dropout_rate if they are in params

    if params.weighting in ['EW', 'UW', 'DWA']:
        pass
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting))

    if params.optim in ['adam']:
        if params.optim == 'adam':
            optim_param = {'optim': 'adam',
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