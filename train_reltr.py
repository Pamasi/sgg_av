# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved

import argparse
import os, random, datetime, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
import neptune 

from utils.common import (
    load_neptune_checkpoint, 
    list_neptune_checkpoint, 
    get_args_parser,
    create_neptune_log,
    constr_version
)
import utils.misc as utils
from coco import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

import pdb

# setup var to cuda crash
#os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:1024'


# to improve performance using cuda 11 sdk
torch.backends.cuda.matmul.allow_tf32 = True
# it should one the same kernel function in nn
torch.backends.cudnn.benchmark = True



def main(args):
    
  
    if args.distributed==True:
        utils.init_multigpu_mode(args)
        args.seed += utils.get_rank()

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device =  torch.device(args.device)


    if args.reproduce:
        torch.manual_seed(args.seed)
        np.random.seed(args.numpy_seed)
        random.seed(args.seed)

    print(f'numpy_seed:{args.numpy_seed}|seed:{args.seed}')

    reltr_model, criterion, postprocessors = build_model(args, use_log=True)
    

       
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
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.c_lr_min, max_lr=args.c_lr_max, cycle_momentum=False) if args.cyclic_lr else torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)


    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn,  num_workers=args.num_workers)
    
    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
            torch.set_rng_state(checkpoint['torch_state'])
            np.random.set_state(checkpoint['numpy_state'])
            random.setstate(checkpoint['random_state'])
            



    if args.use_neptune and  utils.is_main_process():
        # handle crash torchrun
        runs_table_df = neptune.init_project().fetch_runs_table().to_pandas()
        
        failed_run = runs_table_df[runs_table_df["sys/custom_run_id"]==os.environ["NEPTUNE_CUSTOM_RUN_ID"] ]
        
        if not failed_run.empty:
            args.resume_id = failed_run['sys/id'].values[0]

    

        has_weight = False
        
        exp_type = 'test' if args.eval==True else 'train'
        
        if args.enable_kg:
            strategy = 'kge_s' + str(args.prior_strategy)  + '_a' + str(args.rel_operator)
            
            margin = f'margin_{args.margin_loss}'
            tags =  [exp_type,'pytorch', strategy, margin, f'kg_align_coeff_{args.kg_align_coef}']
        elif args.enable_ltn:
            strategy ='ltn_s' + str(args.prior_strategy)  + '_a' + str(args.rel_operator) 
            pos_constr = 'also_pos_constr'
            
            neg_p = f'neg_p_{args.neg_p}'
            pos_p = f'pos_p_{args.pos_p}'
            ent_p = f'ent_p_{args.ent_p}'
            
          
            pos_v = constr_version(args.pos_constr_path)
            
            pos_constr_v = f'pos_constr_v{pos_v}' 
            
            neg_v = constr_version(args.neg_constr_path)
            
            neg_constr_v = f'neg_constr_v{neg_v}'
            
            
            ground = 'ltn_ground'  if args.reltr_grounding==True else  'reltr_ground'
            constr_as = 'tgt_as_constr' if  args.tgt_as_constr==True else  'pred_as_constr'
                
            
            
            if args.incr_neg_p==True:

                tags = [exp_type,'pytorch', strategy, pos_constr, 'incr_neg_p', neg_p, pos_p, ent_p, pos_constr_v, neg_constr_v, ground,  constr_as]
            elif args.different_p == True:
                tags = [exp_type,'pytorch', strategy, pos_constr, 'different_p',  neg_p, pos_p, ent_p, pos_constr_v, neg_constr_v, ground,  constr_as]
            else:
                tags = [exp_type,'pytorch', strategy, pos_constr, pos_constr_v, neg_constr_v, ground,  constr_as]
            
            
        else:
            strategy = 'baseline'
            tags =  [exp_type,'pytorch', strategy,
                     f'cost_bbox_{args.set_cost_bbox}',f'cost_class{args.set_cost_class}',f'cost_giou_{args.set_cost_giou}',
                     f'enc_layers_{args.enc_layers}',f'dec_layers_{args.dec_layers}',f'nheads_{args.nheads}',
                     f'dropout_{args.dropout}',f'ne_{args.num_entities}',f'nt_{args.num_triplets}',
                     f'loss_rel_{args.rel_loss_coef}', f'loss_bbox_{args.bbox_loss_coef}', f'loss_giou_{args.giou_loss_coef}',
                     f'eos_{args.eos_coef}', f'iou_thres_matcher_{args.set_iou_threshold}',
                     f'bk_dim_feedforward_{args.dim_feedforward}', f'tr_hidden_dim{args.hidden_dim}'
                     ]
            
        if args.neptune_offline:
            
            neptune_run =  neptune.init_run(   name =strategy,
                                        tags=tags,
                                        mode='offline'
                                    ) 
        else:
            if args.resume_id != '':
                neptune_run =  neptune.init_run(   name =strategy,
                                                  with_id=args.resume_id,
                                            tags=tags
                                        )
                
                # handle torchrun run restart in case of crash:
                if  neptune_run.exists('training/model/checkpoints') and args.resume=='':
                   
                    epochs = list_neptune_checkpoint(neptune_run)
                    
                    last_epoch = epochs[-1]  # Fetch the last epoch
                    checkpoint = load_neptune_checkpoint(neptune_run, last_epoch)  
                
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    args.start_epoch = checkpoint['epoch'] + 1
                  
                    
                    has_weight = True
          
            else: 
                neptune_run =  neptune.init_run(   name =strategy,
                                                custom_run_id= os.environ["NEPTUNE_CUSTOM_RUN_ID"],
                                            tags=[exp_type,'pytorch', strategy]
                                        ) 
        
    
   
        general_info = {
            "lr_backbone": args.lr_backbone,
            "backbone_name": args.backbone,
            "batch": args.batch_size,
            "model_filename": f'{args.output_dir}/checkpoint.pth',
            "device": args.device,
            "epochs": args.epochs,
            "reltr_parameter": n_parameters,
            "set_criterion_parameter":  sum(p.numel() for p in reltr_model.parameters() if p.requires_grad),
            'exp_name': args.exp_name,
            'in_xywh': args.in_xywh,
            'query_entities': args.num_entities,
            'query_triplets': args.num_triplets,
            'eos_coef': args.eos_coef,
            'bbox_loss_coef': args.bbox_loss_coef,
            'giou_loss_coef': args.giou_loss_coef,
            'rel_loss_coef': args.rel_loss_coef,
            'rel_axiom': args.rel_operator if args.enable_ltn else None,
            'sat_loss_coef': args.sat_loss_coef if args.enable_ltn else None,
            'aggregator_p_sat':args.aggr_p if args.enable_ltn else None, 
            'step_aggr_p_sat':args.step_p if args.enable_ltn else None,
            'univ_p':args.univ_p if args.enable_ltn else None, 
            'step_univ_p':args.univ_step if args.enable_ltn else None, 
            'ent_p': args.ent_p if args.enable_ltn else None,
            'neg_p': args.neg_p if args.enable_ltn else None,
            'pos_p': args.pos_p if args.enable_ltn else None,
            'different_p': args.different_p if args.enable_ltn else None,
            'incr_neg_p': args.incr_neg_p if args.enable_ltn else None,
            'neg_constraints_file': args.neg_constr_path if args.enable_ltn else None,
            'pos_constraints_file': args.pos_constr_path if args.enable_ltn else None,
            'kg_align_coef': args.kg_align_coef if args.enable_kg  else None,
            'kge_margin_loss': args.margin_loss if args.enable_kg  else None,
            'matcher_set_cost_class': args.set_cost_class,
            'matcher_set_cost_bbox': args.set_cost_bbox,
            'matcher_set_cost_giou': args.set_cost_giou,
            'matcher_set_iou_threshold': args.set_iou_threshold,
            'dataset': args.dataset
 
        }
        npt_logger = NeptuneLogger(
            run=neptune_run,
            model=reltr_model,
            log_model_diagram=True
        )

        neptune_run[npt_logger.base_namespace]["general_info"] = stringify_unsupported(  
            general_info
        )

    print("*********************************************************************************")
    print("Start training")
    
    
    aggr_p = args.aggr_p
    univ_p = args.univ_p 
    
    if args.different_p==True:
        list_p = [args.neg_p,args.pos_p, args.ent_p] 
    
    start_time = time.time()
    
    best_mAP = 0.0
    if args.test==False:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)
            # to handle crash
            os.environ['NESY_START_EPOCH']=str(epoch)
            train_stats = train_one_epoch( reltr_model, criterion,
                                        data_loader_train, optimizer,
                                        device, epoch, args.enable_kg, args.enable_ltn,
                                        args.clip_max_norm, ec_enable=args.ec_enable,
                                        pure_nesy=args.pure_nesy, different_p=args.different_p)

            
            lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth'] # anti-crash
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or epoch  % args.save_freq == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # save for reproducibility and later analysis
                        'torch_state': torch.get_rng_state(),
                        'numpy_state': np.random.get_state(),
                        'random_state':random.getstate()
                        
                    }, checkpoint_path)




            # validation result
            val_stats, coco_evaluator, mean_recall= evaluate(reltr_model, criterion, postprocessors, data_loader_val, base_ds, device, args, silent=True, eval=True)

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
                    
            # update neptune writer
            if utils.is_main_process():      
                if args.use_neptune:
                    best_mAP = create_neptune_log(best_mAP, train_stats, val_stats, mean_recall,
                                       has_weight, neptune_run, npt_logger,
                                       data_loader_val.dataset.rel_categories,
                                       reltr_model.module, optimizer, lr_scheduler,
                                       epoch, args)


            if args.output_dir and utils.is_main_process() and  coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        print("*********************************************************************************")
    
    if args.use_neptune:
        neptune_run.stop()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    print(f'args before main {args}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
