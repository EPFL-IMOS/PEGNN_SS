import os
import torch
import torch_geometric
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as graph_dataloader
from torch_geometric.utils import dense_to_sparse, remove_self_loops
import random

def seed_everything(seed = 0):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize(df_train, df_val, df_test):
    df_train_min = df_train.min()
    df_train_max = df_train.max()
    
    df_train_normalized = (df_train - df_train_min) / (df_train_max - df_train_min)
    df_val_normalized = (df_val - df_train_min) / (df_train_max - df_train_min)
    df_test_normalized = (df_test - df_train_min) / (df_train_max - df_train_min)

    
    return df_train_normalized, df_val_normalized, df_test_normalized


def split_dataframe(df, df_physics, train_ratio=0.8, val_ratio=0.1, mode='physics-enhanced'):
        train_ratio = int(0.8 * len(df.index))
        val_ratio = int(0.1 * len(df.index))
        df_train = df.iloc[:train_ratio]
        df_val = df.iloc[train_ratio:train_ratio+val_ratio]
        df_test = df.iloc[train_ratio+val_ratio:]
        df_train, df_val, df_test = normalize(df_train, df_val, df_test)
        df_X_train = df_train.iloc[:,37:]
        df_y_train = df_train.iloc[:,:37]
        df_X_val = df_val.iloc[:,37:]
        df_y_val = df_val.iloc[:,:37]
        df_X_test = df_test.iloc[:,37:]
        df_y_test = df_test.iloc[:,:37]
    
        if mode == 'physics-enhanced':
            df_physics_train = df_physics.iloc[:train_ratio]
            df_physics_val = df_physics.iloc[train_ratio:train_ratio+val_ratio]
            df_physics_test = df_physics.iloc[train_ratio+val_ratio:]
            df_physics_train, df_physics_val, df_physics_test = normalize(df_physics_train, df_physics_val, df_physics_test)
           
            df_X_train = pd.concat([df_X_train, df_physics_train], axis = 1)
            df_X_val = pd.concat([df_X_val, df_physics_val], axis = 1)
            df_X_test = pd.concat([df_X_test, df_physics_test], axis = 1)

    
        return np.array(df_X_train), np.array(df_y_train), np.array(df_X_val), np.array(df_y_val), np.array(df_X_test), np.array(df_y_test)


def gaussian_kernel_distance(feature1, feature2, sigma):
    # Calculate the Euclidean distance between two feature vectors
    distance = np.linalg.norm(feature1 - feature2)
    # Apply the Gaussian kernel function
    weight = np.exp(-distance**2 / (sigma**2))
    return weight


def construct_graph(dataset, threshold_factor=1.0):
    num_features = dataset.shape[1]
    adjacency_matrix = np.zeros((num_features, num_features))

    # Calculate pairwise distances
    pairwise_distances = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(i+1, num_features):
            pairwise_distances[i, j] = np.linalg.norm(dataset[:, i] - dataset[:, j])
            pairwise_distances[j, i] = pairwise_distances[i, j]

    # Calculate sigma as a multiple of the standard deviation of distances
    sigma = np.std(pairwise_distances) * threshold_factor

    # Construct the adjacency matrix with thresholding
    for i in range(num_features):
        for j in range(i+1, num_features):
            if pairwise_distances[i, j] <= threshold_factor:
                weight = gaussian_kernel_distance(dataset[:, i], dataset[:, j], sigma)
                adjacency_matrix[i, j] = weight
                adjacency_matrix[j, i] = weight

    return adjacency_matrix + np.identity(adjacency_matrix.shape[0])


def construct_pyg_data(df_X, df_y, device, window_size = 8):
    PyG_Data = []
    
    for i in range(df_X.T.shape[1] - window_size):
        start_idx = i
        end_idx = start_idx + window_size
        
        # Construct adjacency matrix and edge index
        adj = torch.from_numpy(construct_graph(df_X[start_idx:end_idx, :]).astype(float))
        edge_index = (adj > 0).nonzero().t()
        row, col = edge_index
        edge_weight = adj[row, col]
        
        # Convert NumPy arrays to PyTorch tensors
        x = torch.tensor(df_X.T[:, start_idx:end_idx], dtype=torch.float32)
        y = torch.tensor(df_y.T[:, start_idx:end_idx], dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        # Create PyG Data object and append to list
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y).to(device)
        PyG_Data.append(data)
    
    return PyG_Data