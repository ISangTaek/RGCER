import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from preprocess_data import get_graph_data_from_smiles, convert_to_single_emb_offline
from trainer import Trainer
import weighting as weighting_method
import architecture as architecture_method
from architecture.Graphormer import Encoder as Encoder_Graphormer
from architecture.Graphormer_prompt import Encoder as Encoder_Graphormer_prompt

from metric import ClsMetric, RegMetric
from loss import BCELoss, MSELoss
# Import the modified DataloaderWrapper and DataCollator
from dataset import DataloaderWrapper, DataCollator, PreprocessedDatasetWrapper, \
    SubsetSequentialSampler  # Ensure these are the new versions
from config import prepare_args
from experiment_config import (
    TOXACUTE_PHASE0_CHECKPOINT_NAME,
    TOXACUTE_PHASE0_PREPROCESSED_DIR,
    TOXACUTE_PHASE0_RUN_DIR,
    TOXACUTE_PHASE0_TASKS,
)
from utils import *
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def main(params):
    kwargs, optim_param = prepare_args(params) # prepare_args now also populates arch_args in kwargs

    # Determine device
    if params.gpu_id != 'cpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{params.gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")


    if params.dataset in ['hiv', 'bace', 'bbbp',
                          'qm7', 'esol', 'freesolv', 'lipophilicity']:
        task_name = ['task'] # This implies single-task datasets
    elif params.dataset == 'muv':
        task_name = [
            # 'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652',
            # 'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733',
            # 'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852',
            'MUV-858', 'MUV-859'
            ]
    elif params.dataset == 'tox21':
        task_name = [
            # 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
            # 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ]
    elif params.dataset == 'sider':
        task_name = [
            # ... (full list from original)
            'Nervous system disorders',
            'Injury, poisoning and procedural complications'
        ]
    elif params.dataset == 'clintox':
        task_name = ['FDA_APPROVED', 'CT_TOX']
    elif params.dataset == 'qm8':
        task_name = [
            # ... (full list from original)
            'f1-CAM', 'f2-CAM'
        ]
    elif params.dataset == 'qm9':
        task_name = [
            # 'mu', 'alpha',
            'homo', 'lumo',
            # 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv'
        ]
    elif params.dataset == 'toxacute':
        task_name = [
            # 'mouse_intraperitoneal_LD50',#///
            # 'mouse_intraperitoneal_LDLo',
            # 'mouse_intravenous_LD50',
            # 'mouse_intravenous_LDLo',
            # 'mouse_oral_LD50',
            # 'mouse_oral_LDLo',
            # 'mouse_unreported_LD50',
            # 'mouse_skin_LD50',
            # 'mouse_subcutaneous_LD50',
            # 'mouse_subcutaneous_LDLo',
            # 'mouse_intramuscular_LD50',
            # 'mouse_parenteral_LD50',
            # 'rat_intraperitoneal_LD50',
            # 'rat_intraperitoneal_LDLo',
            # 'rat_intravenous_LD50',
            # 'rat_intravenous_LDLo',
            # 'rat_oral_LD50',
            # 'rat_oral_LDLo',
            #  'rat_unreported_LD50',
            #  'rat_skin_LD50',
            #  'rat_subcutaneous_LD50',
            #  'rat_subcutaneous_LDLo', #/////
            #  'rat_intramuscular_LD50',
            #  'mammal (species unspecified)_intraperitoneal_LD50',
            #  'mammal (species unspecified)_oral_LD50',
            #  'mammal (species unspecified)_unreported_LD50',
            #  'mammal (species unspecified)_subcutaneous_LD50',
            #  'guinea pig_intraperitoneal_LD50', #////
            #  'guinea pig_intravenous_LD50',
            #  'guinea pig_intravenous_LDLo',
            #  'guinea pig_oral_LD50',
            #  'guinea pig_skin_LD50',
            #  'guinea pig_subcutaneous_LD50',
            #  'guinea pig_subcutaneous_LDLo',
            #  'rabbit_intraperitoneal_LD50',
            #  'rabbit_intravenous_LD50',
            #  'rabbit_intravenous_LDLo',
            #  'rabbit_oral_LD50', #///
            #  'rabbit_oral_LDLo',
            #  'rabbit_skin_LD50',
            #  'rabbit_skin_LDLo',
            #  'rabbit_subcutaneous_LD50',
            #  'rabbit_subcutaneous_LDLo',
            #  'dog_intravenous_LD50',
            #  'dog_intravenous_LDLo',
            #  'dog_oral_LD50',
            #  'dog_oral_LDLo',
            #  'cat_intravenous_LD50',
            #  'cat_intravenous_LDLo',
            #  'cat_oral_LD50',
            #  'cat_oral_LDLo',
            #  'bird-wild_oral_LD50',
            #  'quail_oral_LD50',
            #  'duck_oral_LD50',
            # 'chicken_oral_LD50',
            # 'frog_subcutaneous_LDLo',
            'man_oral_TDLo',
            'women_oral_TDLo',
            'human_oral_TDLo'
        ]
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))

    # Phase 0 deliberately fixes toxacute to three tasks.  The longer legacy
    # list above is kept temporarily for historical reference only.
    if params.dataset == 'toxacute':
        task_name = list(TOXACUTE_PHASE0_TASKS)

    # data_path = os.path.join('data', params.dataset + '.csv') # This path is for the raw CSV, used by preprocess_data.py

    # define tasks
    if params.dataset in ['hiv', 'bace', 'bbbp', 'muv',
                          'tox21', 'sider', 'clintox']:
        task_dict = {task: {'metrics': ['AUROC', 'AUPRC'],
                        'metrics_fn': ClsMetric(),
                         'loss_fn': BCELoss(),
                        'weight': [1, 1]} for task in task_name}
    elif params.dataset in ['qm7', 'esol', 'freesolv', 'lipophilicity',
                            'qm8', 'qm9', 'toxacute', ]:
        task_dict = {task: {'metrics': ['RMSE', 'R2'],
                        'metrics_fn': RegMetric(),
                        'loss_fn': MSELoss(),
                        'weight': [-1, 1]} for task in task_name}
    else: # Should have been caught by the dataset check earlier
        raise ValueError(f"Dataset {params.dataset} not configured for task type (Cls/Reg).")


    # Instantiate DataCollator (this will be used as collate_fn)
    collator_instance = DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip,
        device=device, # Pass the determined device
        max_node_filter=params.max_nodes_filter
    )

    # Prepare dataloaders using DataloaderWrapper with preprocessed data
    # Ensure params.preprocessed_data_dir is provided when mode is train or test
    if not params.preprocessed_data_dir and params.mode in ['train', 'test']:
        raise ValueError("Preprocessed data directory (`--preprocessed_data_dir`) is required for train/test mode.")

    data_wrapper = DataloaderWrapper(
        task_list = task_name,
        preprocessed_data_base_dir = params.preprocessed_data_dir,  # Use new path
        batch_size = params.bs,
        splitting = params.splitting,
        valid_size = params.vs,
        test_size = params.ts,
        num_workers = params.num_loader_workers,  # Use new arg
        collate_fn_for_loader = collator_instance  # Pass the collator instance
    )


    # define encoders and decoders
    encoder_class_to_use = None # Renamed from 'encoder' to avoid conflict if 'encoder' is a var elsewhere
    if params.arch == 'Graphormer':
        encoder_class_to_use = Encoder_Graphormer
    elif params.arch == 'Graphormer_prompt':
        encoder_class_to_use = Encoder_Graphormer_prompt
    else:
        # Optionally raise an error or default if architecture is not found
        print(f"Warning: Architecture '{params.arch}' not explicitly mapped to an encoder class. Ensure 'architecture_method' handles it.")
        encoder_class_to_use = None # Or some default

    # Decoders: Ensure hidden_dim from params is used if it influences decoder input size
    decoder_input_dim = params.hidden_dim # Assuming decoders take hidden_dim as input
    decoders = nn.ModuleDict({task: nn.Linear(decoder_input_dim, 1) for task in list(task_dict.keys())})

    # Add species_list to params if it's used by the model/architecture directly from params
    if params.dataset == 'toxacute': # Example condition, adjust if needed
        params.species_list = [
            'mouse', 'rat', 'guinea pig', 'rabbit', 'dog', 'cat',
            'bird-wild', 'quail', 'duck', 'chicken', 'frog',
            'human', 'man', 'women', 'mammal (species unspecified)'
        ]
    else:
        params.species_list = [] # Or some default if not applicable


    # Instantiate Trainer
    # Trainer no longer instantiates its own DataCollator
    model_trainer = Trainer(task_dict=task_dict,
                            weighting=weighting_method.__dict__[params.weighting],
                            architecture=architecture_method.__dict__[params.arch],
                            encoder_class=encoder_class_to_use,
                            decoders=decoders,
                            optim_param=optim_param,
                            args=params, # Pass all command line args
                            save_path=params.save_path,
                            load_path=params.load_path,
                            gpu_id=params.gpu_id, # Already used to set device, but Trainer might use it too
                            **kwargs) # kwargs contains arch_args, weight_args etc. from config.py

    if params.mode == 'train':

        all_task_loaders = data_wrapper.get_data_loaders()  # This returns a dict like {'taskA': {'train':loader, ...}, ...}

        train_dataloaders, val_dataloaders, test_dataloaders = {}, {}, {}
        for task in task_name:
            if task in all_task_loaders:  # Check if task was successfully loaded
                train_dataloaders[task] = all_task_loaders[task].get('train')
                val_dataloaders[task] = all_task_loaders[task].get('val')
                test_dataloaders[task] = all_task_loaders[task].get('test')
            else:
                print(f"Warning: Loaders for task '{task}' not found in all_task_loaders.")
        model_trainer.train(train_dataloaders_dict=train_dataloaders,
                            val_dataloaders_dict=val_dataloaders,
                            test_dataloaders_dict=test_dataloaders,
                            epochs=params.epochs,
                            params_main=params) # Pass params again if train method needs it for ckpt_name etc.
    elif params.mode == 'test':
        all_task_loaders = data_wrapper.get_data_loaders()  # This returns a dict like {'taskA': {'train':loader, ...}, ...}

        train_dataloaders, val_dataloaders, test_dataloaders = {}, {}, {}
        for task in task_name:
            if task in all_task_loaders:  # Check if task was successfully loaded
                train_dataloaders[task] = all_task_loaders[task].get('train')
                val_dataloaders[task] = all_task_loaders[task].get('val')
                test_dataloaders[task] = all_task_loaders[task].get('test')
            else:
                print(f"Warning: Loaders for task '{task}' not found in all_task_loaders.")
        if not test_dataloaders or all(not dl for dl in test_dataloaders.values()):
            print("No test data loaded. Skipping test mode.")
        else:
            model_trainer.test(dataloaders_dict=test_dataloaders)
    elif params.mode == 'single_inference':
        if not params.smiles:
            raise ValueError("SMILES required for single_inference Mode.")
        print(f"--- Running single_inference mode on {params.smiles} SMILES ---")
        # 1. Prepare the graph data from smiles
        # We use a dummy label '0.0' as it won't be used for inference
        graph_data = get_graph_data_from_smiles(params.smiles, 0.0, convert_to_single_emb_offline)
        if not graph_data:
            raise ValueError("SMILES is invalid.")
        # 2. Use the same DataCollator to create a batch of size 1
        # The collator expects a list if data objects
        collated_batch = collator_instance([graph_data])

        # 3. run inference
        model_trainer.model.eval()
        with torch.no_grad():
            predictions = {}
            for task in task_name:
                predictions.update(model_trainer.model(collated_batch, task_name=task, mode='test'))
        # reverse transform


        # 4. Display results
        print("\n--- Predictions ---")
        for task, value in predictions.items():
            output = calculate_mgkg(params.smiles, value.item())
            print(f"{task}: {output:.4f}")
        print("--------------------\n")

    elif params.mode == 'batch_inference':
        if not params.preprocessed_data_dir:
            raise ValueError("Preprocessed data directory (`--preprocessed_data_dir) is required for batch_inference mode.")
        if not params.inference_task:
            raise ValueError("Inference task is required for batch_inference mode.")
        inference_data_path = os.path.join(params.preprocessed_data_dir, params.inference_task)
        print(f"--- Running Batch Inference on data from: {inference_data_path} ---")
        # 1. Prepare DataLoader for the inference dataset
        dataset = PreprocessedDatasetWrapper(inference_data_path)
        if len(dataset) == 0:
            raise ValueError("Empty dataset. Exiting.")
        # Use a sequential sampler to keep the order
        inference_loader = DataLoader(dataset, batch_size = params.bs,
                                      sampler = SubsetSequentialSampler(list(range(len(dataset)))),
                                      num_workers = params.num_loader_workers, collate_fn=collator_instance)
        # 2. Run inference and collect results
        model_trainer.model.eval()
        all_results = []
        with torch.no_grad():
            for batch in tqdm(inference_loader, desc="Running Inference", unit="batch"):
                if batch.get('is_empty', False):
                    continue

                # Process results for each item in the batch
                task_predictions = {
                    task: model_trainer.model(batch, task_name=task, mode='test')[task]
                    for task in task_name
                }
                batch_smiles = batch.smiles
                num_items_in_batch = len(batch_smiles)
                for i in range(num_items_in_batch):
                    result_row = {'smiles' : batch_smiles[i]}
                    for task in task_name:
                        result_row[task] = calculate_mgkg(batch_smiles[i], task_predictions[task][i].item())
                    all_results.append(result_row)

        # 3. Save results to CSV
        output_path = params.inference_output_path or f"./{params.dataset}_{params.inference_task}_predictions.csv"
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index = False)
        print(f"\n--- Batch inference complete. Results saved to: {output_path} ---")



    else:
        raise ValueError("Mode input not recognized.")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Multitask Molecular Graph Learning Framework')
    args.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'single_inference', 'batch_inference'],
                      help='train or test mode')
    args.add_argument('--gpu_id', default='0', type=str, help='GPU ID to use (e.g., "0", "1", or "cpu")')
    args.add_argument('--save_path', type=str, default=TOXACUTE_PHASE0_RUN_DIR, help='Path to save models and logs')
    args.add_argument('--load_path', type=str, default=None, help='Path to load a pretrained model')
    # args.add_argument('--log_path', type=str, default=None, help='Path for extensive logging (if implemented)') # Not used currently
    args.add_argument('--ckpt_name', type=str, default=TOXACUTE_PHASE0_CHECKPOINT_NAME, help='Base name for saved checkpoint files')

    ## New Data Loading Args
    args.add_argument('--preprocessed_data_dir', type=str, default=TOXACUTE_PHASE0_PREPROCESSED_DIR,
                        help='Base directory containing preprocessed graph data files (REQUIRED for train/test)')
    args.add_argument('--num_loader_workers', type=int, default=0, help='Number of workers for DataLoader')
    args.add_argument('--spatial_pos_clip', type=int, default=20, help='Max value for spatial_pos before clipping in DataCollator')
    args.add_argument('--max_nodes_filter', type=int, default=512, # Default to None (no filtering by DataCollator)
                        help='(Optional) Max number of nodes for a graph to be included by DataCollator, filters after loading.')

    ## Inference
    args.add_argument('--smiles', type = str, default = None, help = 'SMILES string for single_inference mode')
    args.add_argument('--inference_task', type = str, default = 'zinc_tox',
                      help = 'Task folder name under preprocessed_data_dir for batch_inference')
    args.add_argument('--inference_output_path', type = str, default = r'./artifacts/results/toxacute_phase0_predictions.csv',
                      help = 'Output CSV file path for batch_inference mode')

    ## MTL Arch
    args.add_argument('--arch', type=str, default='Graphormer_prompt', help='Model architecture name')
    args.add_argument('--a_layers', type=int, default=16, help='Number of layers in atom encoder part (example)')
    args.add_argument('--a_heads', type=int, default=8, help='Number of attention heads in atom encoder part (example)')
    args.add_argument('--t_layers', type=int, default=16, help='Number of layers in Transformer part (example)')
    args.add_argument('--t_heads', type=int, default=8, help='Number of attention heads in Transformer part (example)')
    args.add_argument('--hidden_dim', type=int, default=96, help='Hidden dimension for the model')
    args.add_argument('--mid_dim', type=int, default=128, help='Intermediate dimension (e.g., in FFNs)')

    ## MTL weighting
    args.add_argument('--weighting', type=str, default='EW', choices=['EW', 'UW', 'DWA'], help='MTL Weighting method')

    ## optim
    args.add_argument('--optim', type=str, default='adam', choices=['adam'], help='Optimizer')
    args.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    ## train / dataset
    args.add_argument('--dataset', type=str, default='toxacute', help='Dataset name (used to select tasks and raw data for preprocessing)')
    args.add_argument('--splitting', default='scaffold', type=str, choices=['random', 'scaffold'], help='Data splitting method')
    args.add_argument('--vs', default=0.1, type=float, help='Validation set size ratio')
    args.add_argument('--ts', default=0.1, type=float, help='Test set size ratio')
    args.add_argument('--bs', default=64, type=int, help='Training batch size')
    args.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    args.add_argument('--seed', default=42, type=int, help='Random seed for split-independent training reproducibility')
    args.add_argument('--grad_clip', default=1.0, type=float, help='Maximum gradient norm')

    params = args.parse_args()

    # Create save_path if it doesn't exist
    if params.save_path and not os.path.exists(params.save_path):
        os.makedirs(params.save_path, exist_ok=True)

    main(params)
