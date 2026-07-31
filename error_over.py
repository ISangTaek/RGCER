import torch

# Example file paths from the problematic task/batch
file_path_A = r"processed_graph_data/man_oral_TDLo/data_410.pt" # Use forward slashes or raw strings
file_path_B = r"processed_graph_data/man_oral_TDLo/data_1038.pt" # Another file from same potential batch
file_path_C = r"processed_graph_data/man_oral_TDLo/data_82.pt"   # A reference file

try:
    data_A = torch.load(file_path_A, weights_only=False)
    print(f"Data from {file_path_A}:")
    if hasattr(data_A, 'x'):
        print(f"  x shape: {data_A.x.shape}, dtype: {data_A.x.dtype}")

    data_B = torch.load(file_path_B, weights_only=False)
    print(f"\nData from {file_path_B}:")
    if hasattr(data_B, 'x'):
        print(f"  x shape: {data_B.x.shape}, dtype: {data_B.x.dtype}")

    data_C = torch.load(file_path_C, weights_only=False)
    print(f"\nData from {file_path_C}:")
    if hasattr(data_C, 'x'):
        print(f"  x shape: {data_C.x.shape}, dtype: {data_C.x.dtype}")

except Exception as e:
    print(f"Error loading or inspecting data files: {e}")