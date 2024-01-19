# Copyright (c) Paolo Dimasi, Politecnico di Torino.

import neptune, argparse,  datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List,Optional, Union
from  os import getcwd, mkdir, remove
from neptune.metadata_containers import  Run
from neptune_pytorch import NeptuneLogger

import os.path as osp
import matplotlib.pyplot as plt

def save_ckpt(
    net: torch.nn.Module,
    epoch:int ,
    loss: int ,
    optimizer: Optional[torch.optim.Optimizer] = None,
    dir: str='', 
    ) -> None:
    """Saves the model checkpoint at a given epoch."""
    
    
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    },
               get_ckpt_dir(epoch, dir)
    )    

def get_ckpt_dir(epoch:int, dir:str='' ) -> str:
    if  osp.exists(dir)==False:
        mkdir(osp.join(getcwd(), dir) )
    
    return osp.join(getcwd(), f'{dir}/ckpt_{epoch}', )


def load_ckpt(
    path:str,
    net: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
    ) -> float:
    
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['loss']



def load_neptune_checkpoint(run: neptune.Run, epoch: int):
 
    checkpoint_name = f'checkpoint_{epoch}'
    print(run)
    ext = run['training']['model']['checkpoints'][checkpoint_name].fetch_extension()
    run['training']['model']['checkpoints'][checkpoint_name].download()  # Download the checkpointDownload the checkpoint
    ckpt  = torch.load(f"{checkpoint_name}.{ext}")  # Load the checkpoint

    
    return ckpt

def save_neptune_checkpoint( run: neptune.Run, model: torch.nn.Module, 
                            optimizer: torch.optim.Optimizer, epoch: int, lr_scheduler: torch.optim.lr_scheduler,
                            output_dir:str, is_best:bool=False):
    """"
    save neptune checkpoint and also a local copy for backup
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        # "loss": loss.item(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    
    if is_best==True:
        # remove best file if ss
        checkpoint_name =  f'best_{epoch}.pt'
        # remove on neptune if present
        try:
            del run[f"training/model/checkpoints/best"]
        except:
            pass
        folder_name = 'best'
    else:  
        checkpoint_name =  f'checkpoint_{epoch}.pt'
        folder_name = f'checkpoint_{epoch}'
        
    
    # TODO: write a proper handle, left just to have immediate results on HPC
    try:
        run[f"training/model/checkpoints/{folder_name}/"].upload(checkpoint)  # Upload to Neptune
    except:
        pass
    

def list_neptune_checkpoint(run:neptune.Run) -> Dict[str, Any]:
    """return the neptune checkpoint excluding the best one"""
    all_checkpoint = list(filter(lambda x: not str(x).startswith('best'), run.get_structure()['training']['model']['checkpoints'] ))
    epochs = sorted([

        int(checkpoint.split('_')[-1][:-3]) for checkpoint in all_checkpoint
    ])# Sort the epochs   
    
    return epochs



def create_neptune_log(best_mAP:float, train_stats:Dict[str,float], val_stats:Dict[str, float],  mean_recall:Dict[str, float],
                        has_weight:bool, neptune_run:Run, npt_logger: NeptuneLogger,
                        rel_categories: List[str], reltr_model:torch.nn.Module,  optimizer: torch.optim.Optimizer,
                        lr_scheduler: torch.optim.lr_scheduler,  epoch:int, args:argparse.ArgumentParser) -> float:
    
     # save best checkpoint
    if best_mAP< val_stats['coco_eval_bbox'][1]:
        if has_weight==True and 'best.pt' in  neptune_run.get_structure()['training']['model']['checkpoints']:
            del neptune_run['training']['model']['checkpoints']['best.pt']
        
        save_neptune_checkpoint(neptune_run, reltr_model, optimizer, epoch, 
                                lr_scheduler, args.output_dir, is_best=True)
                                
        best_mAP = val_stats['coco_eval_bbox'][1]
                    
    for metric, value in train_stats.items():
        neptune_run[npt_logger.base_namespace][f'{args.dataset}/train/epoch/loss/{metric}'].append(value)               
    
    for metric, value in val_stats.items():
            
    
            
        if metric == 'coco_eval_bbox':

            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AP/IoU/0.50:0.95/area/all'   ].append(value[0])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AP/IoU/0.50/area/all'        ].append(value[1])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AP/IoU/0.75/area/all'        ].append(value[2])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AP/IoU/0.50:0.95/area/small' ].append(value[3])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AP/IoU/0.50:0.95/area/medium'].append(value[4])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AP/IoU/0.50:0.95/area/large' ].append(value[5])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AR/IoU/0.50:0.95/area/small' ].append(value[9])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AR/IoU/0.50:0.95/area/medium'].append(value[10])      
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/epoch/AR/IoU/0.50:0.95/area/large' ].append(value[11])  

        else:
            neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/{metric}'].append(value)
            
    neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/mR@20/'].append(mean_recall['R@20'])
    neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/mR@50/'].append(mean_recall['R@50'])
    neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/mR@100/'].append(mean_recall['R@100'])


                    
    if metric in rel_categories:
        for r_k, v in value.items():
            if np.isnan(v)==False:
                neptune_run[npt_logger.base_namespace][f'{args.dataset}/val/{metric}/{r_k}'].append(v)

    return best_mAP


def cm_rel(epoch, test_stats):
    cm = test_stats['rel_confmat'].numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.arange(52)  # Replace with the actual number of classes
    tick_marks = np.arange(52)
    plt.xticks(tick_marks, classes)
    plt.xticks(rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(f'rel_cm_epoch_{epoch}.jpg')

def roc_rel(args, auroc_epoch, recall_epoch):
    epochs =  np.arange(args.start_epoch, args.last_epoch+1, args.save_freq)
    plt.subplots(figsize=(10, 10))
        # split 10 batch epoch
    for cls  in range(0,12):
        plt.plot(epochs, auroc_epoch[cls], label=f'Series {cls}', linestyle='-', marker='o')
        # Add labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC validation-set')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    plt.tight_layout()
    plt.savefig('auroc_epoch_0_12_epoch')

    for cls  in range(20,22):
        plt.plot(epochs, auroc_epoch[cls], label=f'Series {cls}', linestyle='-', marker='o')
        # Add labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC validation-set')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    plt.tight_layout()
    plt.savefig('auroc_epoch_20_22_epoch')

    for cls  in range(22,32):
        plt.plot(epochs, auroc_epoch[cls], label=f'Series {cls}', linestyle='-', marker='o')
        # Add labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC validation-set')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    plt.tight_layout()
        # Show the plot or save it to a file
    plt.savefig('auroc_epoch_22_32_epoch')

    plt.subplots(figsize=(10, 10))
    for cls  in range(0,52):
        plt.plot(epochs, recall_epoch[cls], label=f'Series {cls}', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall  validation-set ')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    plt.tight_layout()
    plt.savefig('auroc_epoch_22_32_epoch')
    
        # Show the plot or save it to a file
    plt.savefig('recall_epoch')

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    parser.add_argument('--jitter_policy', default=2, type=int,help="jitter data augmentation policy:\n0-> no jitter \
                            \n1 -> fixed jitter\n->2->random jitter\n3 jitter also on bounding box")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float, 
                        help='set it > 0 to train the backbone')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=150, type=int)
    parser.add_argument('--cyclic_lr', action='store_true', help='cyclic learning rate enabled')
    parser.add_argument('--c_lr_min', default=1e-5, type=float)
    parser.add_argument('--c_lr_max', default=1e-3, type=float)
    # parser.add_argument('--cut_mix', default=0.0, type=float,
    #                     help='cutmix probability')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=5, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # max number of bbox in a img
    parser.add_argument('--num_entities', default=140, type=int,
                        help="Number of query slots i.e. max n° bbox in a img")
    # max number of relationship in a img
    parser.add_argument('--num_triplets', default=140, type=int,
                        help="Number of query slots i.e. max n° relationship in a img")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Prior knowledge
    # LTN
    parser.add_argument('--enable_ltn', default=False, type=bool,
                        help="enable fuzzy logic for prior knowledge")
    parser.add_argument('--sat_loss_coef', default=1, type=float,
                        help="weight for the satisfation aggregator loss")
    
    # axiom type
    parser.add_argument('--rel_operator', default=0, type=int,
                        help="relationship evaluation either using Lukasiewicz's And(0)  or Reichenbach's Implies(1) ")
        
    # constraint
    parser.add_argument('--neg_constr_path', default='constraint/tgv2_so_neg_constr.csv',
                        help='relative path of negative constraint file')
    parser.add_argument('--pos_constr_path', default='constraint/tgv2_pos_constr_v3.csv',
                        help='relative path of positive constraint file')
    
    parser.add_argument('--use_snc', action='store_true',  help="use SNC instead of FNC")
    # aggregator hyperparam
    parser.add_argument('--aggr_p', default=2, type=int,
                        help="p-param value of AggregatorPMean used to computed satisfiability ")
        
    parser.add_argument('--step_p', default=0, type=int,
                        help="p-param value of AggregatorPMean step used to computed satisfiability")
    
    parser.add_argument('--univ_p', default=2, type=int,
                        help="p-param value of AggregatorPMean used to computed  for all ")
    
    parser.add_argument('--ent_p', default=2, type=int,
                        help="p-param value of AggregatorPMean used to computed  for all ")
    parser.add_argument('--rel_p', default=2, type=int,
                        help="p-param value of AggregatorPMean used to computed  for all ")
    
    parser.add_argument('--neg_p', default=1, type=int,
                        help="p-param value of AggregatorPMean used to computed  for all ")
    
    parser.add_argument('--pos_p', default=2, type=int,
                        help="p-param value of AggregatorPMean used to computed for all ")
        
    parser.add_argument('--univ_step', default=0, type=int,
                        help="p-param value of AggregatorPMean step used for all")
    
    parser.add_argument('--p_epoch', default=30, type=int,
                        help="epoch period to change p value of satisfiability and universal quantification")
    
    parser.add_argument('--different_p', action='store_true', help='compute for all sugin different p according to axiom')

    parser.add_argument('--p_easy_constr', default=2, type=int, help='init p for easy relationships to satisfy')
    parser.add_argument('--p_hard_constr', default=4, type=int, help='init p for hard relationships to satisfy')
    
    parser.add_argument('--incr_neg_p', action='store_true', help='increase only p for negative axiom evaluation')
    
    parser.add_argument('--incr_gr', action='store_true', help='increase dimension of learnable predicated by a default value')
    
    parser.add_argument('--dropout_gr', default=0.2, type=float, help='dropout probability of the logit model using during (learnable) predicate grounding')
    
    parser.add_argument('--reltr_grounding', action='store_true', help='no ltn custom grounding')
    parser.add_argument('--tgt_as_constr', action='store_true', help='use target as ground-truth for constraints instead of RelTr prediction')
    parser.add_argument('--ec_enable', action='store_true', help='use entity constraints')
    parser.add_argument('--rc_enable', action='store_true', help='use relation ground-truth constraints')
    parser.add_argument('--nms_sat_enable', action='store_true', help='apply NMS to prediction before satisfiability verification')
    parser.add_argument('--nms_sat', default=0.5, type=float, help='NMS threshold to prune  prediction before satisfiability verification')
    parser.add_argument('--pure_nesy', action='store_true', help='train using only neurosymbolic approach')
    
    # KGE
    parser.add_argument('--enable_kg', default=False, type=bool, help="enable knowledge graph infusion")
    parser.add_argument('--kge_path', default='ckpt/transE/ckpt_200', type=str)
    parser.add_argument('--margin_loss', default=0.5, type=float)
    parser.add_argument('--prior_strategy', default=0, type=int)
    parser.add_argument('--kg_align_coef', default=1, type=float, help="weight for KGE alignment loss ")

    
    # dataset parameters
    parser.add_argument('--dataset', default='tg', help='Name of dataset. Dataset supported is tg ( traffic genome)')
    parser.add_argument('--ann_path', default='coco_traffic_genome_v2/', type=str)
    parser.add_argument('--img_folder', default='coco_traffic_genome_v2/images/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
                      
    parser.add_argument('--reproduce', action='store_true', help='make results reproducible')
    parser.add_argument('--seed', default=18095109307583507207, type=int)
    # numpy seed is 32-bit int, 18095109307583507207 is not accepted
    parser.add_argument('--numpy_seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from local checkpoint')
    # parser.add_argument('--runs_dir', default=f'runs/exp_{datetime.datetime.now().strftime("%m/%d/%Y, %H_%M_%S")}', help='tensorboard runs directory')
    parser.add_argument('--exp_name', default=f'exp_{datetime.datetime.now().strftime("%m/%d/%Y, %H_%M_%S")}', help='experiment name')
    parser.add_argument('--use_neptune',  default=False, type=bool, help='Use neptune logging')
    parser.add_argument('--old_run', action='store_true', help='ckpt saved using old mode')
    parser.add_argument('--neptune_offline',  default=False, type=bool, help='Use neptune offline')
    parser.add_argument('--save_freq', default=10, type=int, help='Frequency to used checkpoint and log' )
    parser.add_argument('--resume_id', default='', type=str, help='sys/id train to resume' )
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true', help='compute performance only on validation set')
    parser.add_argument('--delete', action='store_true', help='delete run values')
    parser.add_argument('--num_workers', default=2, type=int)
    
    parser.add_argument('--in_xywh', action='store_true', help='input bbox format is xywh, otherwise is xyxy')
    parser.add_argument('--only_r_cls', action='store_true', help='print only recall per class')
    # distributed training parameters
    parser.add_argument('--distributed', action='store_false', help='to enable multigpu support')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes or gpu in case of 1 node sent')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    
    # test 
    parser.add_argument('--path_stats', default='report_rel_recall.csv', type=str, help='File where are saved infos related to mR@50 and p-ignite if LTN are used')
    parser.add_argument('--overfit_epoch', default=170, type=int, help='epoch from which mR@100 are not evaluated to be reported in csv file ')
    parser.add_argument('--last_epoch', default=200, type=int,help='test epoch')
    parser.add_argument('--sys_id', default='',type=str, help='sys id of neptune run')
    parser.add_argument('--offline_ckpt', action='store_true', help='load checkpoint from path')
    parser.add_argument('--mean_recall_fixed', default=True )
    parser.add_argument('--skip_mean_recall', action='store_true', help='do not save mean recall results')
    parser.add_argument('--roc_cm_rel', action='store_true', help='compute ROC, CM for relationships')

    parser.add_argument('--task', default='sgdet', choices=['sgdet', 'sgcls', 'predcls'],
                        help='Select the mR with respect to task: sgdet, sgcls or predcls')
    parser.add_argument('--round', default=2, type=int,
                        help='Select how many digit rounds for mean recall')
    
    
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg', help="Path of the test image")
    parser.add_argument('--inf_dir', type=str, default='demo', help="Path of the test image")
    parser.add_argument('--conf', type=float, default=0.3, help="Confidence during inference")
    
    

    return parser

def constr_version(constr_path:str)-> int:
    """ return a integer symbol according to the constraints file version

    Args:
        constr_path (str): path of the file containing the constraints

    Raises:
        ValueError: Invalid constraints path

    Returns:
        int: constraints version
    """

    if constr_path.endswith('v1.csv'):
        v = 1
    elif constr_path.endswith('v2.csv'):
        v = 2
    elif constr_path.endswith('so_neg_constr.csv') or constr_path.endswith('v3.csv'):
        v = 3
    elif constr_path.startswith('constraint/tail'):
        v = 4
        
    else:
        raise ValueError('Invalid constraints!')
       
        
    return v
