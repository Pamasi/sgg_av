# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys, time
from typing import Iterable, Dict
import numpy as np

import torch


from coco.coco_eval import CocoEvaluator
import utils.misc as utils
from utils.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassROC, MulticlassAUROC, MulticlassRecall

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,  enable_kg:bool=False, 
                    enable_ltn:bool=False, max_norm: float = 0, ec_enable=False, rc_enable=True,
                    pure_nesy:bool=False, different_p:bool=False
                    ) -> Dict[str, float]:
    """
    train one given: model param, hyperparam and prior kg strategy is present
    """

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # use only to see how error propagate during training
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))   
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    

    
    if enable_kg:
        metric_logger.add_meter('kge_align_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
    if enable_ltn:
        metric_logger.add_meter('sat_agg_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
        
        if ec_enable==True:
            metric_logger.add_meter('sat_entity_axiom', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('entity_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            
        if rc_enable==True:
            metric_logger.add_meter('sat_relationship_axiom', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('relationship_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            
        metric_logger.add_meter('sat_neg_axiom', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('sat_pos_axiom', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
        if different_p:
            metric_logger.add_meter('neg_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('neg_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            
            metric_logger.add_meter('pos_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))       
            metric_logger.add_meter('pos_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}')) 
                  
        else:
            metric_logger.add_meter('neg_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('pos_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
        criterion.save_sat_debug = True



    start_time = time.time()

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

        outputs = model(samples)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        
        if pure_nesy==False:
            metric_logger.update(rel_error=loss_dict_reduced['rel_error'])
        
        if enable_kg:
            metric_logger.update(kge_align_error=loss_dict_reduced['loss_kg_align'])
        
        if enable_ltn:
            metric_logger.update(sat_agg_error=loss_dict_reduced['sat_agg_error'])
            
            if ec_enable:
                metric_logger.update(sat_entity_axiom = criterion.sat_debug['entity_axiom_sat'])
                metric_logger.update(entity_axiom_fuzzy = criterion.axiom_debug['entity_axiom'])
                
            if rc_enable:
                metric_logger.update(sat_relationship_axiom=criterion.sat_debug['rel_axiom_sat'])
                metric_logger.update(relationship_axiom_fuzzy=criterion.axiom_debug['rel_axiom'] )
                
       
            metric_logger.update(sat_neg_axiom = criterion.sat_debug['neg_axiom_sat'])
            metric_logger.update(sat_pos_axiom = criterion.sat_debug['pos_axiom_sat'])
           
            if different_p:
                metric_logger.update(neg_easy_axiom_fuzzy =    criterion.axiom_debug['neg_easy_axiom'])
                metric_logger.update(neg_hard_axiom_fuzzy =    criterion.axiom_debug['neg_hard_axiom'])
                
                metric_logger.update(pos_easy_axiom_fuzzy =    criterion.axiom_debug['pos_easy_axiom'])
                metric_logger.update(pos_hard_axiom_fuzzy =    criterion.axiom_debug['pos_hard_axiom'])
                
            else:
                metric_logger.update(neg_axiom_fuzzy =    criterion.axiom_debug['neg_axiom'])
                metric_logger.update(pos_axiom_fuzzy =    criterion.axiom_debug['pos_axiom'])                


            
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    epoch_time =  time.time() - start_time
    print(f'Epoch time(mm:ss):{int(epoch_time/60)}:{int(epoch_time%60)}')
    
    stats_logger = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    


    return stats_logger

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args, silent=False, eval=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    if args.pure_nesy==False:
        metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

  
    if  criterion.enable_ltn:
        metric_logger.add_meter('sat_agg_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
        if args.ec_enable:
            metric_logger.add_meter('entity_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
        if args.rc_enable:
            metric_logger.add_meter('relationship_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            
        if args.different_p:
            metric_logger.add_meter('neg_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('neg_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            
            metric_logger.add_meter('pos_easy_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))       
            metric_logger.add_meter('pos_hard_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}')) 
                  
        else:
            metric_logger.add_meter('neg_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            metric_logger.add_meter('pos_axiom_fuzzy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
        criterion.save_sat_debug = False
        
    if criterion.enable_kg:
        metric_logger.add_meter('kge_align_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        

    # initilize evaluator
    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    if eval:
        evaluator_list = []
        for index, name in enumerate(data_loader.dataset.rel_categories):
             # FIXME:skip background cls
            if name!='__background__':
                evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    else:
        evaluator_list = None
    # else:
    #     all_results = []

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    
    if args.roc_cm_rel:
        rel_confmat =  MulticlassConfusionMatrix(num_classes=criterion.num_rel_classes+1)

        # use [.25,.5,.75,1] threshold for ROC
        rel_roc = MulticlassROC(num_classes=criterion.num_rel_classes+1, thresholds=4)
        rel_auroc = MulticlassAUROC(num_classes=criterion.num_rel_classes+1, average=None,thresholds=4)
        rel_recall = MulticlassRecall(num_classes=criterion.num_rel_classes+1, average=None)

    #generator_iter =  data_loader  if silent else metric_logger.log_every(data_loader, 100,  'Test:')
    for samples, targets in data_loader:

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)


          
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        if args.pure_nesy==False:
            metric_logger.update(rel_error=loss_dict_reduced['rel_error'])
        
        # left to debug LTN method more precisely   
        if  criterion.enable_ltn:
            metric_logger.update(sat_agg_error=loss_dict_reduced['sat_agg_error'])
            
            if args.ec_enable:
                metric_logger.update(entity_axiom_fuzzy = criterion.axiom_debug['entity_axiom'])
                
            if args.rc_enable:
                metric_logger.update(relationship_axiom_fuzzy=criterion.axiom_debug['rel_axiom'] )

            if args.different_p:
                metric_logger.update(neg_easy_axiom_fuzzy =    criterion.axiom_debug['neg_easy_axiom'])
                metric_logger.update(neg_hard_axiom_fuzzy =    criterion.axiom_debug['neg_hard_axiom'])
                
                metric_logger.update(pos_easy_axiom_fuzzy =    criterion.axiom_debug['pos_easy_axiom'])
                metric_logger.update(pos_hard_axiom_fuzzy =    criterion.axiom_debug['pos_hard_axiom'])
                
            else:                
                metric_logger.update(neg_axiom_fuzzy =    criterion.axiom_debug['neg_axiom'])
                metric_logger.update(pos_axiom_fuzzy =    criterion.axiom_debug['pos_axiom'])
            
        if criterion.enable_kg:
            metric_logger.update(kge_align_error=loss_dict_reduced['loss_kg_align'])
            
        if args.roc_cm_rel:
            all_rel_gt = criterion.rel_assign.cpu()
            all_rel_pred = outputs['rel_logits'].cpu()

            for b in range(all_rel_gt.size(0)):
                rel_confmat.update(all_rel_pred[b], all_rel_gt[b])
                rel_roc.update(all_rel_pred[b], all_rel_gt[b])
                rel_auroc.update(all_rel_pred[b], all_rel_gt[b])
                rel_recall.update(all_rel_pred[b], all_rel_gt[b])

        evaluate_rel_batch(outputs, targets, evaluator, evaluator_list, 
                           False if args.ann_path.endswith('_v2/') else True, args.task )




        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    
    


    if silent==False:
        print(f'TASK is {args.task}')
        evaluator[args.task].print_stats()

    # call only during test set
    mean_recall = None
    recall_per_cls = None
    if eval: #and args.dataset == 'vg':
        mean_recall, recall_per_cls = calculate_mR_from_evaluator_list(evaluator_list, args.task)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # add mean recall per class
    if recall_per_cls:
        for cls, r_k in recall_per_cls.items():
            stats[cls] = r_k
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

 

    # just to avoid to compute it during evaluation if not necessary by default
    if args.roc_cm_rel:
        stats['rel_confmat'] = rel_confmat.compute()
        stats['rel_roc'] = rel_roc
        
        auroc_all = rel_auroc.compute()
        recall_all = rel_recall.compute()

        for cls in range(criterion.num_rel_classes+1):
            stats[f'rel_auroc_{cls}'] = auroc_all[cls]
            stats[f'rel_recall_{cls}'] = recall_all[cls]

    return stats, coco_evaluator, mean_recall

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list, background_first,  task):


    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        if background_first:
            rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)
        else:
            rel_scores = outputs['rel_logits'][batch][:,:-1].softmax(-1)

        pred_entry = {'sub_boxes': sub_bboxes_scaled,
                      'sub_classes': pred_sub_classes.cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled,
                      'obj_classes': pred_obj_classes.cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy()}

        evaluator[task].evaluate_scene_graph_entry(gt_entry, pred_entry, background_first)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel[task].evaluate_scene_graph_entry(gt_entry_rel, pred_entry, background_first)


