# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved

import argparse, json, hashlib
import os, random, datetime, time, math, csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist


from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
import neptune 

import utils.misc as utils
from coco import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from utils.common import old_load_neptune_checkpoint, get_args_parser, roc_rel, cm_rel

# DEBUG = True
import pandas as pd
# setup var to cuda crash
# os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512'


# to improve performance using cuda 11 sdk
torch.backends.cuda.matmul.allow_tf32 = True
# it should one the same kernel function in nn
torch.backends.cudnn.benchmark = True

    
def main(args):
    

  
    if args.distributed==True:
        utils.init_multigpu_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device =  torch.device(args.device)

    seed = args.seed + utils.get_rank()
    # if args.reproduce:
    torch.manual_seed(args.seed)
   
    np.random.seed(args.numpy_seed)

    random.seed(args.seed)


    reltr_model, criterion, postprocessors = build_model(args)
    

       
    reltr_model.to(device)

    model_without_ddp = reltr_model
    if args.distributed:
        print(f'gpus available are:{args.gpu} and current rank is {args.rank}')
        reltr_model = torch.nn.parallel.DistributedDataParallel(reltr_model, device_ids=[args.gpu])
        model_without_ddp = reltr_model.module
    n_parameters = sum(p.numel() for p in reltr_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    if args.val==False:
        dataset_test = build_dataset(image_set='test', args=args)  
        folder_test = 'test'
    else:  
        dataset_test = build_dataset(image_set='val', args=args)
        if args.ann_path.startswith('coco_mix'):
            folder_test = 'val'

        else:
            folder_test = 'val_tg'
 
        

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)


    
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn,  num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_test)


    if args.use_neptune and  utils.is_main_process():  
        # # handle crash torchrun
        runs_table_df = neptune.init_project().fetch_runs_table().to_pandas()
        
        run = runs_table_df[runs_table_df["sys/id"]==args.sys_id]
        
        if  run.empty:
            raise ValueError('Invalid neptun run id')


        neptune_run =  neptune.init_run(     with_id=args.sys_id )

        
                
        if args.delete:
            path_del = f'training/{args.dataset}/{folder_test}'
            print(f'path_del={path_del}')
            del neptune_run[path_del]
            
            return

        else:
            npt_logger = NeptuneLogger(
                    run=neptune_run,
                    model=reltr_model,
                    log_model_diagram=True,
            )
    if args.roc_cm_rel:
        auroc_epoch = {i: [] for i in range(0,52) }
        recall_epoch = {i: [] for i in range(0,52) }


    # remove data if they were already stored
    if args.use_neptune:
        fixed = 'updated_' if args.mean_recall_fixed else ''
        try:
            del neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{fixed}mR@20/']
        except:
            pass

        try:
            del neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{fixed}mR@50/']
        except:
            pass
        
        try:
            del neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{fixed}mR@100/']
        except:
            pass

        for metric in data_loader_test.dataset.rel_categories:
            if metric!='__background__':
                for r_k in ['R@100', 'R@50', 'R@20']:
                    #if neptune_run.exists(f'{args.dataset}/{folder_test}/{metric}/{r_k}'):
                    try:
                        del neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{metric}/{r_k}']

                    except:
                        print('Neptune does not exist: it cannot be canceled!')



          
    relation_analysis = { 'rel': [], 'max_mR100': [], 'max_mR@100@epoch':[], 'mR@100@overfit_epoch':[] }
 
    if args.enable_ltn:
        relation_analysis['p_ignite']= []
        relation_analysis['max_mR@100@epoch(p)']= []

        for r in  data_loader_test.dataset.rel_categories:
            relation_analysis['rel'].append(r)
            relation_analysis['max_mR100'].append(0)
            relation_analysis['p_ignite'].append(0)
            relation_analysis['max_mR@100@epoch(p)'].append('')
            relation_analysis['max_mR@100@epoch'].append('')
            relation_analysis['mR@100@overfit_epoch'].append(0)
   
    else:
        for r in  data_loader_test.dataset.rel_categories:
            relation_analysis['rel'].append(r)
            relation_analysis['max_mR100'].append(0)
            relation_analysis['max_mR@100@epoch'].append('')
            relation_analysis['mR@100@overfit_epoch'].append(0)

    aggr_p = args.aggr_p
    univ_p = args.univ_p 
    
    if args.different_p==True:
        list_p = [args.neg_p,args.pos_p, args.ent_p] 

    for epoch in range(args.start_epoch, args.last_epoch+1, args.save_freq):
        
        if args.offline_ckpt:
            # path 
            ckpt_path = f'{args.output_dir}/checkpoint{epoch:04}.pth'
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            
        else:
            ckpt = old_load_neptune_checkpoint(neptune_run, epoch)
            model_without_ddp.load_state_dict(ckpt, strict=True)




                
        test_stats, _ , mean_recall = evaluate(reltr_model, criterion, postprocessors, 
                                                        data_loader_test, base_ds, device, 
                                                        args, silent=False, eval=True)

        # save data  confunsion matrix and ROC 
        if args.roc_cm_rel:
            cm_rel(epoch, test_stats)

            for cls  in range(0, 52):
                auroc_epoch[cls].append(test_stats[f'rel_auroc_{cls}'] )
                recall_epoch[cls].append(test_stats[f'rel_recall_{cls}'] )

        if args.use_neptune:
            for metric, value in test_stats.items():
                    
                if metric in data_loader_test.dataset.rel_categories:
                    for r_k, v in value.items():
                        if np.isnan(v)==False:
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{metric}/{r_k}'].append(v)

                      
                        report_recall(args, metric, relation_analysis, univ_p, epoch, value)


                elif args.only_r_cls==False:
                
                    if metric == 'coco_eval_bbox' :
           
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AP/IoU/0.50:0.95/area/all'   ].append(value[0])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AP/IoU/0.50/area/all'        ].append(value[1])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AP/IoU/0.75/area/all'        ].append(value[2])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AP/IoU/0.50:0.95/area/small' ].append(value[3])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AP/IoU/0.50:0.95/area/medium'].append(value[4])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AP/IoU/0.50:0.95/area/large' ].append(value[5])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AR/IoU/0.50:0.95/area/small' ].append(value[9])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AR/IoU/0.50:0.95/area/medium'].append(value[10])      
                            neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/AR/IoU/0.50:0.95/area/large' ].append(value[11])  
        

                    elif metric !='rel_confmat' and metric!='rel_roc':
                        neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{metric}'].append(value)
                    
            # save mean recall     
            if args.skip_mean_recall==False:
                    fixed = 'updated_' if args.mean_recall_fixed else ''
                    neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{fixed}mR@20/'].append(mean_recall['R@20'])
                    neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{fixed}mR@50/'].append(mean_recall['R@50'])
                    neptune_run[npt_logger.base_namespace][f'{args.dataset}/{folder_test}/{fixed}mR@100/'].append(mean_recall['R@100'])

       
        else:
            # save only mR to the specific epochs
            for metric, value in test_stats.items():
                if metric in data_loader_test.dataset.rel_categories:     
                    report_recall(args, metric, relation_analysis, univ_p, epoch, value)
                               

        # change aggregator norm 
        if  args.epochs>0 and epoch%args.p_epoch==0  and args.step_p>0:
            aggr_p +=args.step_p
            
            
            if args.univ_step>0:
                univ_p +=args.univ_step
                if args.different_p==True:
                    list_p[0]+=args.univ_step
                    
                    if args.incr_neg_p==False:
                        list_p[1]+=args.univ_step
                        list_p[2]+=args.univ_step
                    
                    criterion.change_aggr_p(aggr_p, list_p=list_p ) 
                    
                else:
                    criterion.change_aggr_p(aggr_p, univ_p)
                
            else:
                criterion.change_aggr_p(aggr_p)

        
    if args.use_neptune:
        neptune_run.stop()

    # save stats on mR on  a csv file
    print(relation_analysis)
    pd_relationship = pd.DataFrame.from_dict(relation_analysis)

    if args.enable_ltn:
        pd_relationship.sort_values(by=['rel']).to_csv(args.path_stats, header=False, columns=['rel', 'max_mR@100@epoch(p)', 'mR@100@overfit_epoch'], index=False)
    else:
        pd_relationship.sort_values(by=['rel']).to_csv(args.path_stats, header=False, columns=['rel', 'max_mR@100@epoch', 'mR@100@overfit_epoch'], index=False)  
       
    if args.roc_cm_rel:
        roc_rel(args, auroc_epoch, recall_epoch)



def report_recall(args, metric, relation_analysis, univ_p, epoch, value):
    v =  value['R@100']
    if np.isnan(v)==False:
        mR_100 = v
        idx = relation_analysis['rel'].index(metric)
        val_rounded= np.round(mR_100*100, args.round)

        if mR_100 > relation_analysis['max_mR100'][idx] and epoch<=args.overfit_epoch:
            relation_analysis['max_mR100'][idx]=mR_100
            if args.enable_ltn: 
                if relation_analysis['p_ignite'][idx]==0:
                    relation_analysis['p_ignite'][idx] = univ_p
                p_ignite = relation_analysis['p_ignite'][idx]

                relation_analysis['max_mR@100@epoch(p)'][idx]=f'{val_rounded}@{epoch}({p_ignite})'
                                
                            
            relation_analysis['max_mR@100@epoch'][idx]=f'{val_rounded}@{epoch}'

        if epoch==args.overfit_epoch:
            relation_analysis['mR@100@overfit_epoch'][idx]=val_rounded



if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    print(f'args before main {args}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
