import os
import csv
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # Import for scaffold
import networkx as nx
import argparse
import pandas as pd  # Added for robust CSV reading

from experiment_config import (
    TOXACUTE_PHASE0_PREPROCESSED_DIR,
    TOXACUTE_PHASE0_RAW_CSV,
    TOXACUTE_PHASE0_TASKS,
)

# Assuming algos.py is in the same directory or accessible in PYTHONPATH
try:
    import algos
except ImportError:
    print("Error: The file named 'algos.pyd' for win or 'algos.so' for linux not found. Ensure it's in the same directory or PYTHONPATH.")
    exit()


# --- Feature Extraction Functions (Copied/adapted from your dataset.py) ---
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    fea_Symbol = one_of_k_encoding_unk(atom.GetSymbol(),
                                       ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                        'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                        'Pt', 'Hg', 'Pb', 'Unknown'])
    fea_Degree = one_of_k_encoding_unk(atom.GetDegree(), list(range(17)))
    fea_TotalNumHs = one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(17)))
    fea_ImplicitValence = one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(17)))
    fea_IsAromatic = [atom.GetIsAromatic()]
    return np.array(fea_Symbol + fea_Degree + fea_TotalNumHs + fea_ImplicitValence + fea_IsAromatic, dtype = np.float32)


def _generate_scaffold(smiles, include_chirality = False):  # Copied from original dataset.py
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle cases where MolFromSmiles returns None
        return ''  # Return an empty string or a specific placeholder for invalid SMILES
    try:
        scaffold = MurckoScaffoldSmiles(mol = mol, includeChirality = include_chirality)
    except:  # Catch potential errors in MurckoScaffoldSmiles
        scaffold = ''
    return scaffold


def convert_to_single_emb_offline(x, offset = 1):
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.numel() == 0:
        return x
    feature_num = x.size(1)
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype = torch.float)
    return x + feature_offset


def get_graph_data_from_smiles(smiles_string, label_val, convert_x_fn):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    # Generate scaffold
    scaffold_smiles = _generate_scaffold(smiles_string)

    atom_f_list = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        atom_f_list.append(feature)

    if not atom_f_list:
        return None

    x_np = np.array(atom_f_list)
    x = torch.tensor(x_np, dtype = torch.float)

    if convert_x_fn:
        x = convert_x_fn(x)

    edge_list_tuples = []
    for bond in mol.GetBonds():
        edge_list_tuples.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    N = x.size(0)
    adj = torch.zeros([N, N], dtype = torch.bool)

    final_edges_for_index = []
    if edge_list_tuples:
        for u, v in edge_list_tuples:
            final_edges_for_index.append([u, v])
            final_edges_for_index.append([v, u])
            adj[u, v] = True
            adj[v, u] = True

    if final_edges_for_index:
        edge_index = torch.tensor(final_edges_for_index, dtype = torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype = torch.long)

    in_degree = adj.long().sum(dim = 1)
    out_degree = in_degree.clone()

    spatial_pos = torch.full((N, N), -1, dtype = torch.long)
    if N > 0:
        shortest_paths_matrix, _ = algos.floyd_warshall(adj.numpy())
        # Replace np.inf with a value (e.g., -1 or a large int if your model expects that for disconnected parts)
        # Ensure that your model's embedding layers or attention mechanisms correctly handle this value.
        # Using a large positive number can sometimes be problematic if your model uses these distances directly
        # without appropriate clipping or transformation in embedding layers.
        # Using -1 is a common choice if 0 is a valid distance for self-loops (if any) or very close nodes.
        # Check your model's spatial_pos embedding logic.

        # shortest_paths_matrix[shortest_paths_matrix == np.inf] = -1
        spatial_pos = torch.from_numpy(shortest_paths_matrix).long()

    y_tensor = torch.tensor([float(label_val)], dtype = torch.float)

    data = Data(x = x, edge_index = edge_index, y = y_tensor,
                in_degree = in_degree, out_degree = out_degree,
                spatial_pos = spatial_pos, smiles = smiles_string,
                scaffold_smiles = scaffold_smiles)  # Add scaffold here
    return data


def preprocess_and_save(raw_csv_path, task_list_str, output_dir_base):
    if not os.path.exists(raw_csv_path):
        print(f"Error: Raw CSV data file not found at {raw_csv_path}")
        return
    if task_list_str == 'all':
        tasks = pd.read_csv(raw_csv_path).columns[6:].tolist()
        print('-' *20 + f'Total tasks: {len(tasks)}')
    else:
        tasks = [t.strip() for t in task_list_str.split(',')]
    if not tasks:
        print("Error: No tasks provided.")
        return

    os.makedirs(output_dir_base, exist_ok = True)
    print(f"Preprocessing for tasks: {tasks}")
    print(f"Reading raw data from: {raw_csv_path}")
    print(f"Saving preprocessed data to: {os.path.abspath(output_dir_base)}")

    df = pd.read_csv(raw_csv_path)

    for task_name in tasks:
        print(f"\nProcessing task: {task_name}")
        if task_name not in df.columns:
            print(f"Warning: Task '{task_name}' not found as a column in {raw_csv_path}. Skipping.")
            continue

        task_output_dir = os.path.join(output_dir_base, task_name)
        os.makedirs(task_output_dir, exist_ok = True)

        num_processed = 0
        num_skipped = 0

        for index, row in df.iterrows():
            smiles = row.get('smiles')
            label_str = str(row.get(task_name))

            if not smiles or not label_str or label_str.lower() == 'nan' or label_str == '':
                num_skipped += 1
                if index % 500 == 0 and index > 0: print(
                    f"  Skipped invalid entry at original index {index} for task {task_name}.")
                continue

            try:
                label_val = float(label_str)
            except ValueError:
                num_skipped += 1
                if index % 500 == 0 and index > 0: print(
                    f"  Skipped entry with non-float label '{label_str}' at original index {index} for task {task_name}.")
                continue

            if index % 100 == 0 and index > 0:
                print(f"  Processing molecule {index}/{len(df)} for task {task_name}...")

            try:
                graph_data = get_graph_data_from_smiles(smiles, label_val, convert_to_single_emb_offline)
                if graph_data is not None and graph_data.x.size(0) > 0:
                    output_path = os.path.join(task_output_dir, f"data_{index}.pt")
                    torch.save(graph_data, output_path)
                    num_processed += 1
                else:
                    if graph_data is None or (hasattr(graph_data, 'x') and graph_data.x.size(0) == 0):
                        print(
                            f"  Skipping molecule (original index {index}, SMILES: {smiles}) for task {task_name} due to processing error or empty graph.")
                    num_skipped += 1
            except Exception as e:
                print(
                    f"  Error processing molecule (original index {index}, SMILES: {smiles}) for task {task_name}: {e}")
                num_skipped += 1

        print(f"Finished processing task {task_name}. Successfully processed: {num_processed}, Skipped: {num_skipped}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Offline Data Preprocessing for Graph Neural Networks")
    parser.add_argument('--raw_csv_path', type = str,
                        default = TOXACUTE_PHASE0_RAW_CSV
                        ,help = "Path to the raw CSV data file.")
    parser.add_argument('--task_list', type = str,
                        default = ','.join(TOXACUTE_PHASE0_TASKS),
                        help = "Comma-separated task names. Phase 0 defaults to the fixed toxacute three-task set.")
    parser.add_argument('--output_dir', type = str, default = TOXACUTE_PHASE0_PREPROCESSED_DIR,
                        help = "Directory to save preprocessed graph data.")

    args = parser.parse_args()

    print("Starting offline data preprocessing...")
    preprocess_and_save(args.raw_csv_path, args.task_list, args.output_dir)
    print("Offline data preprocessing complete.")
