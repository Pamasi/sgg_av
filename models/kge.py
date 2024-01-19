import os.path as osp
from  os import  getcwd
from random import randint
import torch
from torch.utils.data import ChainDataset, Subset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.dataset import KGDataset
from torch_geometric.nn import TransE
from torch_geometric.nn.kge.loader import KGTripletLoader
from utils.common import  save_ckpt


# constant param 
NUM_UNIQUE_ID_NODES = 7416 + 7 # num_node + yaw, pos, rot 
NUM_UNIQUE_ID_REL = 13
# EMBEED_SIZE = 50
EMBEED_SIZE = 256


def build_kge(device: torch.device, path_ckpt: str) -> TransE:
    """ create a KGE model given its weight 
    
    Args:
        device (torch.device): device use to load the model (cuda|cpu)
        path_ckpt (str): path where the checkpoint file is saved

    Returns:
        TransE: pretrained TransE model
    """
    dataset = KGDataset(osp.join(getcwd(), "prior_kg"))


    model = TransE(
        num_nodes=NUM_UNIQUE_ID_NODES,
        num_relations=NUM_UNIQUE_ID_REL, # 754
        # embeeding size
        hidden_channels=EMBEED_SIZE,   
    ).to(device)
    
    model_dict = torch.load(path_ckpt)['model_state_dict']
    model.load_state_dict(model_dict)
    model.eval()
    
    return model
