# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved

import os.path as osp
from  os import  getcwd
from random import randint
import argparse
import torch
from torch.utils.data import ChainDataset, Subset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd

from utils.dataset import KGDataset
from torch_geometric.nn import TransE
from torch_geometric.nn.kge.loader import KGTripletLoader
from utils.common import  save_ckpt, load_ckpt



def get_args_parser():
    parser = argparse.ArgumentParser('Set TransE parameter', add_help=False)
    parser.add_argument('--test_only', action='store_true', help='test only the weight of TransE')
    parser.add_argument('--ckpt_path', default='ckpt/transE', help='set the directory where ckpt are save and load' )
    parser.add_argument('--res_path', default='log_train/transE_val_rank.csv', help='set the directory where test and validation result are saved as csv file' )
    parser.add_argument('--num_epoch', default=200, type=int, help='set number of training epoch' )
    parser.add_argument('--val_step', default=1, type=int)
    parser.add_argument('--num_unique_id_nodes', default=7423, type=int, help='number of unique nodes in the Knowledge Graph')
    parser.add_argument('--num_unique_id_rel', default=13, type=int, help='number of unique relationships in the Knowledge Graph')
    parser.add_argument('--num_worker', default=16, type=int, help='number of workers')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--emb_size', default=256 type=int, help='dimension of the embedding')
    parser.add_argument('--k_hit', default=100, type=int)
    parser.add_argument('train_split', default=55, type=int)
    parser.add_argument('--test_idx', default=57, type=int, help='test set')
    parser.add_argument('--val_idx', default=56, type=int, help='val set')
    parser.add_argument('--data_len', default=57, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parsr.add_argument('lr', default=1e-4, type=float)

    return parser
    
def train(i, args):
    model.train()
    total_loss = total_examples = 0
    
    if i > args.data_len:
        i = args.data_len % i
        
    train_loader= KGTripletLoader(  dataset[i].edge_index[0]  ,
                                    dataset[i].edge_type ,
                                    dataset[i].edge_index[1] ,
                                    batch_size=args.data_len,
                                    shuffle=True,
                                    num_workers=args.num_workers
                )
    
    for head_index, rel_type, tail_index in train_loader:
        optimizer.zero_grad()
        
        loss = model.loss(head_index.to(device), rel_type.to(device), tail_index.to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
        

    

    return total_loss / total_examples


@torch.no_grad()
def test(data, args):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=args.batch_size,
        k=args.k_hit,
    )



def main(args):
    device = args.device

    dataset = KGDataset(osp.join(getcwd(), "prior_kg"))

    # edge vector of relationship
    num_edge_types  = sum([ i.num_edge_types for  i in dataset ])
    num_node = sum([ i.num_nodes for  i in dataset ])

    print(f'num of total node in knowledge graph  {num_node}')
    print(f'num  of total relationships (edges) in knowledge graph {num_edge_types}')
    model = TransE(
        num_nodes=args.num_unique_id_nodes,
        num_relations=args.num_unique_id_rel,
        # embeeding size
        hidden_channels=args.emb_size,   
    ).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)


    val_data = dataset[args.val_idx]
    step = 10 if args.test_only else 1
    
    mode = 'Test' if args.test_only else 'Train'
    print(f'MODE: {mode}')
    res  = {'is_val' : [], 'hit':[], 'mean_rank': []}
    
    
    for epoch in range(0, args.num_epoch, step):
        if args.test_only==False:
            loss = train(epoch, args)
        else:
            # load corresponding ckpt
            path = f'{args.ckpt_path}/ckpt_{epoch}'
            loss = load_ckpt(path, model, optimizer=optimizer)
            

            
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if (epoch+1) % args.val_step == 0 or epoch==(args.num_epoch-1):
            
            rank, hits = test(val_data, args)
            res['is_val'].append(1)
            res['hit'].append(hits) 
            res['mean_rank'].append(rank)
            
            print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
                f'Val Hits@{args.k_hit}: {hits:.4f}')
            
         
            
            if args.test_only==False:    
                save_ckpt(model, epoch, loss,   optimizer, dir =  args.ckpt_path)
            
        scheduler.step(rank)   
    

    test_data = dataset[args,test_idx]
    rank, hits = test(test_data,args)
    res['is_val'].append(0)
    res['hit'].append(hits) 
    res['mean_rank'].append(rank)
    df_res = pd.DataFrame.from_dict(res)
    print(f'Test Mean Rank: {rank:.2f}, Test Hits@{args.k_hit}: {hits:.4f}')
    
    df_res.to_csv(args.res_path, header=True, index=False)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransE training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

 


