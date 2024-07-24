from pathlib import Path
import argparse
import time

import torch
import random


from utils import *
from model import *
from train_test import *

import warnings
warnings.filterwarnings("ignore")

def get_arguments():
    parser = argparse.ArgumentParser(description="Physics-Enhanced GNN for Soft Sensing", 
                                     add_help=False)

    # Data
    parser.add_argument("--data-dir", type=str, default="data/",
                        help='Path to the data')
    
    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="exp/",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')

    # Optim
    parser.add_argument("--seed", type=int, default=42,
                        help='Seed for experiments')
    parser.add_argument("--epochs", type=int, default=25000,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=64,
                        help='Batch size')
    parser.add_argument("--base-lr", type=float, default=3e-4,
                        help='Base Learning rate')
    parser.add_argument("--window-size", type=int, default=8,
                        help='Window Size')
    parser.add_argument("--patience", type=int, default=200,
                        help='patience for early stopping')

    # Running
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
                        
    return parser


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
    

def mains(args):
    device = args.device if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('Data/df_sensors.csv')
    df_physics = pd.read_csv('Data/df_physics.csv')

    df_X_train, df_y_train, df_X_val, df_y_val, df_X_test, df_y_test = split_dataframe(df, df_physics, train_ratio=0.8, val_ratio=0.1, mode='physics-enhanced')

    
    #Create Dataset
    PyG_Data_Train = construct_pyg_data(df_X_train, df_y_train, device)
    PyG_Data_Val = construct_pyg_data(df_X_val, df_y_val, device)
    PyG_Data_Test = construct_pyg_data(df_X_test, df_y_test, device)
    
    #Create Dataloader
    Train_DATA = graph_dataloader(PyG_Data_Train, batch_size = 64, shuffle = False, drop_last = True)
    Validation_DATA = graph_dataloader(PyG_Data_Val, batch_size = 64, shuffle = False, drop_last = True)
    Test_DATA = graph_dataloader(PyG_Data_Test, shuffle = False)
    
    #define Model
    model = GNNModel().to(device)

    trained_model = train_gnn_model(model, Train_DATA, Validation_DATA, device = device, window_size = args.window_size, patience = args.patience, EPOCHS = args.epochs, lr = args.base_lr)
    preds_list, targets_list, mse = test_gnn_model(trained_model, Test_DATA, window_size = args.window_size, device = device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_arguments()])
    args = parser.parse_args([])
    
    seed_everything(args.seed)
    mains(args)