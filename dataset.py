import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader
import torch_geometric
from torch.utils.data.sampler import SubsetRandomSampler, Sampler as TorchSampler  # Renamed to avoid conflict
from collections import defaultdict  # For grouping scaffolds


# Definition for SubsetSequentialSampler (if not imported from elsewhere)
class SubsetSequentialSampler(TorchSampler):  # Use TorchSampler as base
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PreprocessedDatasetWrapper(Dataset):
    def __init__(self, task_data_dir):
        super(PreprocessedDatasetWrapper, self).__init__()
        self.task_data_dir = task_data_dir
        self.processed_files = []
        if not os.path.isdir(task_data_dir):
            raise FileNotFoundError(f"Preprocessed data directory not found: {task_data_dir}")

        for f_name in os.listdir(self.task_data_dir):
            if f_name.startswith("data_") and f_name.endswith(".pt"):
                self.processed_files.append(os.path.join(self.task_data_dir, f_name))

        self.processed_files.sort(key = lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

        if not self.processed_files:
            print(f"Warning: No preprocessed .pt files found in {self.task_data_dir}. This dataset will be empty.")

    def __getitem__(self, index):
        file_path = self.processed_files[index]
        try:
            # Load the entire Data object. It should be small enough after preprocessing.
            data = torch.load(file_path, weights_only = False)
            # --- Add this debugging block ---
            # if hasattr(data, 'spatial_pos') and data.spatial_pos is not None:
            #     is_resizable = data.spatial_pos.storage().resizable()
            #     print(
            #         f"Loaded {file_path}: spatial_pos.is_resizable = {is_resizable}, shape = {data.spatial_pos.shape}")
            #     if not is_resizable:
            #         # If it's not resizable even after loading a supposedly cloned version,
            #         # this points to a very deep issue or that the clone didn't happen/save correctly.
            #         print(f"CRITICAL WARNING: spatial_pos in {file_path} loaded with non-resizable storage!")
            # --- End debugging block ---
            return data
        except Exception as e:
            print(f"Error loading preprocessed file {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.processed_files)

    def get_scaffold_smiles_for_splitting(self, index):
        # Helper to get scaffold for a specific index without loading full data if not needed,
        # but for simplicity, we'll load it here as __getitem__ does.
        # A more optimized way would be to save scaffolds in a separate metadata file.
        data = self.__getitem__(index)
        if data and hasattr(data, 'scaffold_smiles'):
            return data.scaffold_smiles
        return ""  # Default scaffold if not found or error


def new_random_split(dataset_len, task_name, valid_size, test_size):
    print(f"{task_name} has {dataset_len} preprocessed samples for random splitting.")
    indices = list(range(dataset_len))
    np.random.shuffle(indices)
    split_valid = int(np.floor(valid_size * dataset_len))
    split_test = int(np.floor(test_size * dataset_len))

    valid_idx = indices[:split_valid]
    test_idx = indices[split_valid: split_valid + split_test]
    train_idx = indices[split_valid + split_test:]

    print(f"  Random Split - Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
    return train_idx, valid_idx, test_idx


def scaffold_split_for_preprocessed(dataset: PreprocessedDatasetWrapper, task_name: str, valid_size: float,
                                    test_size: float):
    """
    Performs scaffold splitting on a PreprocessedDatasetWrapper.
    The dataset items are expected to have a 'scaffold_smiles' attribute.
    Uses a more robust splitting logic.
    """
    print(f"Performing scaffold split for task: {task_name}")
    dataset_len = len(dataset)
    if dataset_len == 0:
        print("  Dataset is empty, returning empty splits.")
        return [], [], []

    scaffolds = defaultdict(list)
    for i in range(dataset_len):
        # Assuming get_scaffold_smiles_for_splitting loads the Data object to get the scaffold.
        # This can be slow if done repeatedly. Consider pre-fetching all scaffolds once if performance is an issue.
        data_item = dataset[i]  # Load the data item once
        if data_item is None or not hasattr(data_item, 'scaffold_smiles'):
            scaffold = ""  # Default for missing scaffold
        else:
            scaffold = data_item.scaffold_smiles if data_item.scaffold_smiles is not None else ""
        scaffolds[scaffold].append(i)

    # Sort scaffolds by size (descending) to ensure deterministic and balanced splits
    # Sorting by x[0] (scaffold string) as a secondary key for full determinism
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key = lambda x: (len(x[1]), x[0]), reverse = True
        )
    ]

    train_inds, valid_inds, test_inds = [], [], []

    # Calculate target sizes
    train_ratio = 1.0 - valid_size - test_size
    target_train_count = train_ratio * dataset_len
    target_valid_count = valid_size * dataset_len
    # target_test_count = test_size * dataset_len # Implicitly the rest

    print(
        f"  Total samples: {dataset_len}, Target train: ~{target_train_count:.2f}, Target valid: ~{target_valid_count:.2f}")

    # Distribute scaffold sets
    for scaffold_set in scaffold_sets:
        # Try to fill train set first
        if len(train_inds) + len(scaffold_set) <= target_train_count + 1e-5:  # Allow for small float inaccuracies
            train_inds.extend(scaffold_set)
        elif len(
                train_inds) < target_train_count * 1.25:  # Allow train to slightly overshoot if it's the only way to add a scaffold
            # Prioritize train if it's not too much over budget and valid/test can still be formed
            if (len(valid_inds) + len(test_inds) == 0 or  # if valid/test are empty, train can take more
                    len(train_inds) / dataset_len < train_ratio + 0.05):  # Or train is not too much over its ratio
                train_inds.extend(scaffold_set)  # Add to train if it doesn't make it excessively large
                continue  # Move to next scaffold

        # Then try to fill valid set
        if len(valid_inds) + len(scaffold_set) <= target_valid_count + 1e-5:
            valid_inds.extend(scaffold_set)
        elif len(valid_inds) < target_valid_count * 1.25:  # Allow valid to slightly overshoot
            if (len(test_inds) == 0 or
                    len(valid_inds) / dataset_len < valid_size + 0.05):
                valid_inds.extend(scaffold_set)
                continue

        # Remaining scaffolds go to the test set
        else:
            test_inds.extend(scaffold_set)

    # Fallback: If any set is empty due to scaffold distribution and others are too full,
    # try to rebalance from the largest set if constraints are too strict.
    # This part can get complex. A simpler approach is to ensure minimums if sets are empty.
    # For now, the above greedy assignment is common. The issue might be very skewed scaffold sizes.
    # If valid or test is empty, and train has all data, we might need to move the smallest
    # training scaffolds to valid/test until they meet a minimum size or ratio.
    # However, for 121 samples, if one scaffold has 100+ items, it's hard to split.

    # Let's use the more standard approach that fills greedily but in order (train, then val, then test):
    train_inds, valid_inds, test_inds = [], [], []  # Reset
    train_cutoff_count = int(train_ratio * dataset_len)
    valid_cutoff_count = int((train_ratio + valid_size) * dataset_len)

    for scaffold_set in scaffold_sets:
        if len(train_inds) + (len(scaffold_set) / 2.0) < train_cutoff_count:  # Prioritize train
            train_inds.extend(scaffold_set)
        elif len(train_inds) + len(valid_inds) + (len(scaffold_set) / 2.0) < valid_cutoff_count:  # Then valid
            valid_inds.extend(scaffold_set)
        else:  # Then test
            test_inds.extend(scaffold_set)

    # If after this distribution, any set is disproportionately small/large compared to others,
    # especially if valid/test are empty and train has everything.
    # This can happen if the largest scaffolds are too big.
    # A common strategy is to iterate: fill train, then fill valid from remaining, then test gets rest.

    # Refined splitting logic (similar to MoleculeNet's approach):
    train_inds, valid_inds, test_inds = [], [], []  # Reset
    # Approximate number of data points for each set
    num_train = int(train_ratio * dataset_len)
    num_valid = int(valid_size * dataset_len)
    # num_test = dataset_len - num_train - num_valid

    # Assign scaffolds to datasets
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) <= num_train + len(scaffold_set) * 0.5:  # allow some flexibility
            # if len(train_inds) < num_train: # Simpler greedy fill for train
            train_inds.extend(scaffold_set)
        elif len(valid_inds) + len(scaffold_set) <= num_valid + len(scaffold_set) * 0.5:
            # elif len(valid_inds) < num_valid: # Simpler greedy fill for valid
            valid_inds.extend(scaffold_set)
        else:
            test_inds.extend(scaffold_set)

    # Shuffle within each set for randomness during training
    if train_inds: np.random.shuffle(train_inds)
    if valid_inds: np.random.shuffle(valid_inds)
    if test_inds: np.random.shuffle(test_inds)

    print(f"  Refined Scaffold Split - Train: {len(train_inds)}, Valid: {len(valid_inds)}, Test: {len(test_inds)}")

    # Check for empty validation/test sets after splitting, especially for small datasets
    if not valid_inds and len(train_inds) > 1 and dataset_len > len(train_inds):  # If valid is empty but there's room
        print("  Warning: Validation set is empty after scaffold split. Trying to move some from train if possible.")
        # This is a simplistic fallback, more advanced balancing might be needed for robust small dataset splitting
        if len(train_inds) > int(0.5 * dataset_len) and test_size > 0:  # ensure train isn't too depleted
            num_to_move_to_valid = max(1, int(valid_size * dataset_len))  # at least 1 or target
            if len(train_inds) > num_to_move_to_valid:
                valid_inds.extend(train_inds[:num_to_move_to_valid])
                train_inds = train_inds[num_to_move_to_valid:]
                np.random.shuffle(train_inds)
                np.random.shuffle(valid_inds)
                print(
                    f"  Fallback applied - Train: {len(train_inds)}, Valid: {len(valid_inds)}, Test: {len(test_inds)}")

    if not test_inds and len(train_inds) + len(valid_inds) < dataset_len:
        print("  Warning: Test set is empty after scaffold split. Trying to move some from train/valid if possible.")
        num_to_move_to_test = max(1, int(test_size * dataset_len))
        # Prioritize moving from train if valid is already small or also empty
        if len(train_inds) > num_to_move_to_test + (1 if valid_inds else 0):  # ensure train has enough left
            test_inds.extend(train_inds[:num_to_move_to_test])
            train_inds = train_inds[num_to_move_to_test:]
        elif len(valid_inds) > num_to_move_to_test:  # try moving from valid
            test_inds.extend(valid_inds[:num_to_move_to_test])
            valid_inds = valid_inds[num_to_move_to_test:]
        if test_inds:
            np.random.shuffle(train_inds)
            np.random.shuffle(valid_inds)
            np.random.shuffle(test_inds)
            print(f"  Fallback applied - Train: {len(train_inds)}, Valid: {len(valid_inds)}, Test: {len(test_inds)}")

    print(f"  Final Scaffold Split - Train: {len(train_inds)}, Valid: {len(valid_inds)}, Test: {len(test_inds)}")
    return train_inds, valid_inds, test_inds


class DataloaderWrapper():
    def __init__(self, task_list, preprocessed_data_base_dir, batch_size,
                 splitting, valid_size, test_size, num_workers = 0,
                 collate_fn_for_loader = None):
        super().__init__()
        self.task_list = task_list
        self.batch_size = batch_size
        self.preprocessed_data_base_dir = preprocessed_data_base_dir
        self.splitting = splitting
        self.valid_size = valid_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.collate_fn_for_loader = collate_fn_for_loader
        if splitting not in ['random', 'scaffold']:
            raise ValueError(f"Splitting type '{splitting}' not supported. Choose 'random' or 'scaffold'.")

    def get_data_loaders(self):
        all_task_loaders = {}
        print("=" * 40)
        print(f"Initializing DataLoaders with num_workers = {self.num_workers}, splitting = {self.splitting}")
        for task_idx, task_name in enumerate(self.task_list):
            all_task_loaders[task_name] = {}
            task_specific_data_dir = os.path.join(self.preprocessed_data_base_dir, task_name)

            dataset = PreprocessedDatasetWrapper(task_specific_data_dir)
            if len(dataset) == 0:
                print(f"Warning: Dataset for task '{task_name}' is empty. Loaders will be empty.")
                all_task_loaders[task_name]['train'] = DataLoader([], batch_size = self.batch_size,
                                                                  collate_fn = self.collate_fn_for_loader)
                all_task_loaders[task_name]['val'] = DataLoader([], batch_size = self.batch_size,
                                                                collate_fn = self.collate_fn_for_loader)
                all_task_loaders[task_name]['test'] = DataLoader([], batch_size = self.batch_size,
                                                                 collate_fn = self.collate_fn_for_loader)
                continue

            train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(dataset, task_name)

            all_task_loaders[task_name]['train'] = train_loader
            all_task_loaders[task_name]['val'] = valid_loader
            all_task_loaders[task_name]['test'] = test_loader
        return all_task_loaders

    def get_train_validation_data_loaders(self, dataset, task_name):
        if self.splitting == 'random':
            train_idx, valid_idx, test_idx = new_random_split(len(dataset), task_name, self.valid_size, self.test_size)
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split_for_preprocessed(dataset, task_name, self.valid_size,
                                                                             self.test_size)
        else:
            raise ValueError(f"Unsupported splitting method: {self.splitting}")


        collate_function = self.collate_fn_for_loader if self.collate_fn_for_loader else None

        # Ensure indices are not empty before creating samplers
        if not train_idx: print(f"Warning: Task {task_name} has an empty training set after splitting.")
        if not valid_idx: print(f"Warning: Task {task_name} has an empty validation set after splitting.")
        if not test_idx: print(f"Warning: Task {task_name} has an empty test set after splitting.")

        train_loader = DataLoader(dataset, batch_size = self.batch_size,
                                  sampler = SubsetRandomSampler(train_idx) if train_idx else None,
                                  num_workers = self.num_workers, pin_memory = False, collate_fn = collate_function,
                                  drop_last = True if train_idx and len(
                                      train_idx) > self.batch_size else False)  # Drop last for train
        valid_loader = DataLoader(dataset, batch_size = self.batch_size,
                                  sampler = SubsetSequentialSampler(valid_idx) if valid_idx else None,
                                  num_workers = self.num_workers, pin_memory = False, collate_fn = collate_function)
        test_loader = DataLoader(dataset, batch_size = self.batch_size,
                                 sampler = SubsetSequentialSampler(test_idx) if test_idx else None,
                                 num_workers = self.num_workers, pin_memory = False, collate_fn = collate_function)
        return train_loader, valid_loader, test_loader


# --- DataCollator (remains the same as provided in the previous step) ---
class DataCollator(torch.nn.Module):
    def __init__(self, spatial_pos_max_clip, device, max_node_filter = None):
        super(DataCollator, self).__init__()
        self.spatial_pos_max_clip = spatial_pos_max_clip
        self.device = device
        self.max_node_filter = max_node_filter

    def pad_1d_unsqueeze(self, x, padlen):
        x = x + 1
        xlen = x.size(0)
        if xlen < padlen: new_x = x.new_zeros([padlen], dtype = x.dtype); new_x[:xlen] = x; x = new_x
        return x.unsqueeze(0)

    def pad_2d_unsqueeze(self, x, padlen):
        x = x + 1
        if x.ndim == 1: x = x.unsqueeze(1)
        xlen, xdim = x.size() if x.numel() > 0 else (0, 0)
        if xlen == 0 and padlen > 0: return x.new_zeros([1, padlen, xdim if xdim > 0 else 1], dtype = x.dtype)
        if xlen < padlen: new_x = x.new_zeros([padlen, xdim], dtype = x.dtype); new_x[:xlen, :] = x; x = new_x
        return x.unsqueeze(0)

    def pad_attn_bias_unsqueeze(self, x, padlen):
        xlen = x.size(0)
        if xlen < padlen:
            new_x = x.new_zeros([padlen, padlen], dtype = x.dtype).fill_(float("-inf"))
            if xlen > 0: new_x[:xlen, :xlen] = x; new_x[xlen:, :xlen] = 0; new_x[:xlen, xlen:] = 0
            x = new_x
        return x.unsqueeze(0)

    def pad_spatial_pos_unsqueeze(self, x, padlen):
        x = x + 1
        xlen = x.size(0)
        if xlen < padlen: new_x = x.new_zeros([padlen, padlen], dtype = x.dtype); new_x[:xlen, :xlen] = x; x = new_x
        return x.unsqueeze(0)

    def move_dict_to_gpu(self, data_dict, device):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)
            elif isinstance(value, dict):
                data_dict[key] = self.move_dict_to_gpu(value, device)
            elif isinstance(value, list):
                data_dict[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        return data_dict

    def __call__(self, data_list):
        # ####################################################################
        # print(f"--- CUSTOM DataCollator __call__ INVOKED! Processing {len(data_list)} items. ---")
        # if data_list and isinstance(data_list[0], torch_geometric.data.Data):
        #     print(
        #         f"--- First item is a PyG Data object. Example node features shape: {data_list[0].x.shape if hasattr(data_list[0], 'x') else 'N/A'} ---")
        # ####################################################################

        # Filter out None items if __getitem__ can return None on error

        data_list = [data for data in data_list if data is not None]
        if not data_list:
            return {'x': torch.empty(0), 'in_degree': torch.empty(0), 'out_degree': torch.empty(0),
                    'spatial_pos': torch.empty(0), 'attn_bias': torch.empty(0), 'y': torch.empty(0), 'is_empty': True}

        if self.max_node_filter is not None:
            data_list = [item for item in data_list if hasattr(item, 'x') and item.x.size(0) <= self.max_node_filter]

        if not data_list:
            return {'x': torch.empty(0), 'in_degree': torch.empty(0), 'out_degree': torch.empty(0),
                    'spatial_pos': torch.empty(0), 'attn_bias': torch.empty(0), 'y': torch.empty(0), 'is_empty': True}

        # keep track of SMILES strings
        smiles_list = [item.smiles for item in data_list]

        xs = [item.x for item in data_list]
        in_degrees = [item.in_degree for item in data_list]
        spatial_poses = [item.spatial_pos for item in data_list]
        ys = [item.y for item in data_list]

        max_node_num_in_batch = 0
        if xs: max_node_num_in_batch = max(i.size(0) for i in xs if hasattr(i, 'numel') and i.numel() > 0)

        padded_x = torch.cat([self.pad_2d_unsqueeze(i, max_node_num_in_batch) for i in xs])
        padded_degree = torch.cat([self.pad_1d_unsqueeze(i, max_node_num_in_batch) for i in in_degrees])
        padded_spatial_pos = torch.cat(
            [self.pad_spatial_pos_unsqueeze(i, max_node_num_in_batch) for i in spatial_poses])

        batch_attn_biases = []
        for item_spatial_pos in spatial_poses:
            N_item = item_spatial_pos.size(0)
            attn_bias_item = torch.zeros([N_item + 1, N_item + 1], dtype = torch.float)
            if N_item > 0:
                condition = item_spatial_pos >= self.spatial_pos_max_clip
                attn_bias_item[1:, 1:][condition] = float("-inf")
            batch_attn_biases.append(attn_bias_item)
        padded_attn_bias = torch.cat(
            [self.pad_attn_bias_unsqueeze(i, max_node_num_in_batch + 1) for i in batch_attn_biases])

        if ys and isinstance(ys[0], torch.Tensor):
            if ys[0].ndim == 0:
                processed_ys = torch.tensor([y.item() for y in ys], dtype = torch.float).view(-1, 1)
            elif ys[0].ndim == 1 and ys[0].size(0) == 1:
                processed_ys = torch.cat(ys, dim = 0).view(-1, 1)
            else:
                processed_ys = torch.stack(ys, dim = 0)
        elif ys:
            processed_ys = torch.tensor(ys, dtype = torch.float).view(-1, 1)
        else:
            processed_ys = torch.empty((0, 1), dtype = torch.float)

        # final_batch_data = {
        #     'x': padded_x.long(), 'in_degree': padded_degree, 'out_degree': padded_degree.clone(),
        #     'spatial_pos': padded_spatial_pos, 'attn_bias': padded_attn_bias, 'y': processed_ys, 'is_empty': False
        # }
        final_batch_data = torch_geometric.data.Data(
            x= padded_x.long(), in_degree = padded_degree, out_degree =  padded_degree.clone(),
        spatial_pos =  padded_spatial_pos, attn_bias = padded_attn_bias, y = processed_ys, is_empty = False,
            smiles= smiles_list
        )
        return self.move_dict_to_gpu(final_batch_data, self.device)