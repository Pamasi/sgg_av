# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn, linalg
import pandas as pd
import numpy as np
import random
from torchvision.ops import batched_nms

from typing import Dict, List, Optional, Tuple

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .kge import build_kge
from .fuzzy_logic import *
import ltn_log

from utils.dataset import MAP_PD2TG

from torch_geometric.nn import TransE
import ltn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RelTR(nn.Module):
    """ RelTR: Relation Transformer for Scene Graph Generation """
    def __init__(self, backbone:torch.nn.Module, transformer:torch.nn.Module,
                 num_classes:int, num_rel_classes:int, num_entities:int, num_triplets:int,
                 aux_loss=False, enable_kg:bool= False, enable_ltn:bool=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_entities: number of entity queries
            num_triplets: number of coupled subject/object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            enable_kg: True if prior KGE has to be used (Default False).
            enable_ltn: True if ltn has to be used (Default False).
         
        """
        super().__init__()
        self.num_entities = num_entities
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        #
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entities, hidden_dim*2)
        # 
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim*3)
        self.so_embed = nn.Embedding(2, hidden_dim) # subject and object encoding

        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


        # mask head : sub and obj attenation heatmaps
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                           nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))

        # predicate classification
        self.rel_class_embed = MLP(hidden_dim*2+128, hidden_dim, num_rel_classes + 1, 2)

        # subject/object label classfication and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # prior kg
        self.enable_kg  = enable_kg
        self.enable_ltn = enable_ltn
   
    



    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the entity classification logits (including no-object) for all entity queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": the normalized entity boxes coordinates for all entity queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "sub_logits": the subject classification logits
               - "obj_logits": the object classification logits
               - "sub_boxes": the normalized subject boxes coordinates
               - "obj_boxes": the normalized object boxes coordinates
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)



        features, pos = self.backbone(samples)


        src, mask = features[-1].decompose()
        assert mask is not None

        # hs is hidden space [num entities, 256] from the transformer
        # ht is hidden space [num triple, 256] concat for sub and obj
        hs, hs_t, so_masks, _  = self.transformer(self.input_proj(src), mask, self.entity_embed.weight,
                                                 self.triplet_embed.weight, pos[-1], self.so_embed.weight)
        so_masks = so_masks.detach()
        so_masks = self.so_mask_conv(so_masks.view(-1, 2, src.shape[-2],src.shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
        so_masks = self.so_mask_fc(so_masks)
        # output from Entity Decoder
        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)

        # using [num_entities .e. bbox , num_class] -
        outputs_class = self.entity_class_embed(hs)


        # output: [num triplets, 4 i.e. bbox's coordinates]
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        # output: [num triplets, 36 i.e. n° class]
        outputs_class_sub = self.sub_class_embed(hs_sub)


        pred_logits = outputs_class[-1]


        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

        out = {'pred_logits': pred_logits, 'pred_boxes': outputs_coord[-1],
                'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
                'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
                'rel_logits': outputs_class_rel[-1]
        }

        if self.enable_kg:
            prob_out_class = pred_logits.sigmoid()
            num_ent = prob_out_class.size()[-2]
            batch =  prob_out_class.size()[0]
            num_cls =  prob_out_class.size()[-1]


            max_prob_class =torch.stack([ torch.stack([ torch.argmax(prob_out_class[b, e, :]) for e in range(num_ent)]) for b in range(batch) ])
            assert max_prob_class.size()[1] == num_ent, 'Number of entities must be fixed'

            # map cls from class emb to know idx inside hs module
            pos_dict = [{ c:[] for c in range(num_cls) } for _ in range(batch) ]
            [ pos_dict[b][max_prob_class[b, e].item()].append(e)   for e in range(num_ent) for b in range(batch) ]





            # computation of logits triple and relative boxes
            out['cls_index_hs'] = pos_dict 
            out['entity_emb'] = hs[-1]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                  outputs_class_obj, outputs_coord_obj, outputs_class_rel)



        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g}
                for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for RelTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes:int, num_rel_classes:int, matcher:nn.Module,
                 weight_dict:Dict[str, float], eos_coef:float, losses:Dict[str, float],
                 prior_kge: TransE=None, device:torch.device='cuda',
                 sim_strategy:int=0, sim_margin:int=0.5, 
                 kge_key_path:str = 'map2emb.txt',
                 enable_ltn:bool=False,
                 entity_fuzzy_th:int = 0.7,
                 ltn_strategy:int = 0,
                 rel_operator:int = 0,
                 aggr_p:int =2,
                 univ_p:int=2,
                 different_p:Optional[List[int]]=None,
                 neg_constr_path:str='constraint/adria_neg_constr.csv',
                 pos_constr_path:str='constraint/adria_pos_constr.csv',
                 increase_dim_net:bool=False,
                 p_dropout:float=0.2,
                 reltr_grounding:float=False,
                 ec_enable:bool=False,
                 rc_enable:bool=True,
                 tgt_as_constr:bool=False,
                 use_pred_nms:bool=False,
                 sat_pred_nms:float=0.5,
                 max_nms_bbox:int=30,
                 pure_nesy:bool=False,
                 rel_hard_axiom_path:str='constraint/rel_hard_axiom.csv',
                 rel_easy_axiom_path:str='constraint/rel_easy_axiom.csv',
                 use_snc:bool=True
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            prior_kge(TransE): KGE model representing the prior knowledge
            sim_strategy(int):  0 -> cosine embedding similarity
                                1 -> hinge similarity
            kge_key_path(str): path where key to index value of KGE are saved
            enable_ltn(bool): enable fuzzy logic during training
            constr_path(int): path where negative constraints are saved
            entity_fuzzy_th(int) =0.7  threshold for intepreth as guess of ground truth entity in  fuzzy fashion
            ltn_strategy(int) = 0  -> ground entity = logit entity
                                1  -> ground entity follows https://arxiv.org/pdf/1910.00462.pdf
            rel_operator(int) = 0 -> and  for  relationship axiom w.r.t negative constraints
                              1 -> imply for relationship axiom w.r.t negative constraints
            aggr_p(int) =2  p value of aggregator parameter for satisfiability aggregation
            univ_p(int) =2  p value of aggregator parameter for universal quantification
            constr_path(str) path of constraint file
            increase_dim_net(bool) increase the number of layer in the network by adding a 3 hidden layer of size 128
            p_dropout(float) = 0.2 dropout probability inside the GroundVanilla network,
            reltr_grounding(float) = False addition ground by ltn 
            ec_enable(bool) = False enable entity constraints
            er_enable(bool) = False enable relationship satisfiability wrt to ground-truth
            tgt_as_constr(bool) = use target as  constraints
            use_pred_nms(float) = False, enable NMS for pruning prediction to verified by LTN instead of evaluating all predictions
            sat_pred_nms(float) = NMS threshold for pruning prediciton to verified by LTN
            max_nms_bbox(int) = 30
            pure_nesy(bool)=False  use a pure neurosymbolic approach for entity and relationship detection 
            rel_hard_axiom_path(str): path where relationship class related to high p for all are saved
            rel_easy_axiom_path(str): path where relationship class related to moderate  p for all are saved
            use_snc(bool) = True : use the second formulation of range constraints 
            """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        self.pure_nesy= pure_nesy
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)


        self.num_rel_classes = num_rel_classes # Using entity class numbers to adapt rel class numbers
        empty_weight_rel = torch.ones(num_rel_classes+1)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

        self.device = device
        # Prior Kge
     
        if prior_kge:
            self.enable_kg = True
            self.prior_kge = prior_kge


            # similarity
            self.sim_strategy= sim_strategy
            match sim_strategy:
                case 0:
                    self.similarity = nn.CosineEmbeddingLoss(margin=sim_margin)
                    print(f'KGE: Cosine Embedding Loss method')
                    
                case 1:
                    self.pair_distance = nn.PairwiseDistance(p=2, keepdim=True)
                    self.similarity = nn.HingeEmbeddingLoss(margin=sim_margin)
                    print(f'KGE: Hinge Embedding Loss method')
                case _:
                    raise ValueError(f'kge strategy {sim_strategy} is invalid')



            with open(kge_key_path, 'r') as fp:
                self.key2idx = { l.replace('\n', ''): i for i, l in enumerate(fp.readlines()) }
        else:
            self.enable_kg=False
        self.enable_ltn = enable_ltn
        self.rel_assign = None

        if enable_ltn:
            self.ec_enable = ec_enable
            self.rc_enable = rc_enable
            self.use_tgt_constr = tgt_as_constr
            use_log = True

            if not use_log:
                if different_p == None:
                    self.equal_p = True
                    self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
                    self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")

                    # for simplicity same forall for entity and relationship
                    self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")

                else:
                    self.equal_p = False
                    self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[0]), quantifier="f")
                    self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[0]), quantifier="f")
                    self.forall_hard = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[1]), quantifier="f")

                    self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=different_p[-1]),
                                                           quantifier="f")

                    self.rel_cls_easy = self._parse_type_constraint(rel_easy_axiom_path, device=device)
                    self.rel_cls_hard = self._parse_type_constraint(rel_hard_axiom_path, device=device)

                self.c_not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
                self.and_luk = ltn.Connective(ltn.fuzzy_ops.AndLuk())

                self.ltn_strategy = ltn_strategy

                match rel_operator:
                    case 0:
                        self.rel_operator = ltn.Connective(ltn.fuzzy_ops.AndLuk())

                    case 1:
                        self.rel_operator = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

                    case 2:
                        self.rel_operator = ltn.Connective(ltn.fuzzy_ops.ImpliesGodel())

                    case _:
                        raise ValueError(f'rel_operator {rel_operator} is invalid')

                match self.ltn_strategy:
                    case 0:
                        # just use the embeeding class output of the network
                        self.p_sub_is_a = ltn.Predicate(model=PredFromEmbedding())

                        self.p_obj_is_a = ltn.Predicate(model=PredFromEmbedding())
                        self.p_sub_in_set = ltn.Predicate(model=EntInSet())
                        self.p_obj_in_set = ltn.Predicate(model=EntInSet())

                        gr_last_dim = self.num_classes * 2 + 7 + (self.num_rel_classes + 1)

                    case 1:
                        self.ground_entity = GroundVanilla(layer_sizes=(self.num_classes + 4 + 1, 64, self.num_classes))

                        self.p_sub_is_a = ltn.Predicate(model=GroundPred(self.ground_entity))
                        self.p_obj_is_a = ltn.Predicate(model=GroundPred(self.ground_entity))
                        gr_last_dim = (self.num_classes + 4 + 1) * 2 + 7 + (self.num_rel_classes + 1)
                    case _:
                        raise ValueError(f'ltn strategy {ltn_strategy} is invalid')

                if reltr_grounding == True:
                    if increase_dim_net == False:
                        self.ground_relationship = GroundVanilla(layer_sizes=(gr_last_dim, 64, self.num_rel_classes),
                                                                 p_dropout=p_dropout)


                    else:
                        self.ground_relationship = GroundVanilla(
                            layer_sizes=(gr_last_dim, 64, 128, self.num_rel_classes), p_dropout=p_dropout)

                    self.use_regressor = True
                    self.p_rel_in_set = ltn.Predicate(model=RelInSetL(self.ground_relationship))
                    self.p_rel_is_a = ltn.Predicate(model=RelIsAL(self.ground_relationship))

                else:
                    self.ground_relationship = None
                    self.use_regressor = False

                    self.p_rel_in_set = ltn.Predicate(model=RelInSetL(use_logit=False))
                    self.p_rel_is_a = ltn.Predicate(model=RelIsAL(use_logit=False))

            else:

                if different_p == None:
                    self.equal_p = True
                    self.forall_neg = ltn_log.log.Wrapper_Quantifier(ltn_log.log.fuzzy_ops.Aggreg_Mean(),
                                                                     semantics="forall")
                    self.forall_pos = ltn_log.log.Wrapper_Quantifier(ltn_log.log.fuzzy_ops.Aggreg_Mean(),
                                                                     semantics="forall")

                    # for simplicity same forall for entity and relationship
                    self.forall_ent_o_rel = ltn_log.log.Wrapper_Quantifier(ltn_log.log.fuzzy_ops.Aggreg_Mean(),
                                                                           semantics="forall")

                else:
                    self.equal_p = False
                    self.forall_neg = ltn_log.log.Wrapper_Quantifier(ltn_log.log.fuzzy_ops.Aggreg_Mean(),
                                                                     semantics="forall")
                    self.forall_pos = ltn_log.log.Wrapper_Quantifier(ltn_log.log.fuzzy_ops.Aggreg_Mean(),
                                                                     semantics="forall")
                    self.forall_hard = ltn_log.log.Wrapper_Quantifier(ltn_log.log.fuzzy_ops.Aggreg_Mean(),
                                                                      semantics="forall")

                    self.forall_ent_o_rel = ltn_log.log.Wrapper_Quantifier(ltn_log.log.fuzzy_ops.Aggreg_Mean(),
                                                                           semantics="forall")

                    self.rel_cls_easy = self._parse_type_constraint(rel_easy_axiom_path, device=device)
                    self.rel_cls_hard = self._parse_type_constraint(rel_hard_axiom_path, device=device)

                self.c_not = ltn_log.log.Wrapper_Connective(ltn_log.fuzzy_ops.Not_Std())
                self.and_luk = ltn_log.log.Wrapper_Connective(ltn_log.log.fuzzy_ops.And_Sum())

                self.ltn_strategy = ltn_strategy

                match rel_operator:
                    case 0:
                        self.rel_operator = ltn_log.Wrapper_Connective(ltn_log.log.fuzzy_ops.And_Sum())

                    case 1:
                        self.rel_operator = ltn.Connective(ltn.fuzzy_ops.ImpliesKleeneDienes())

                    case 2:
                        self.rel_operator = ltn.Connective(ltn.fuzzy_ops.ImpliesGodel())

                    case _:
                        raise ValueError(f'rel_operator {rel_operator} is invalid')

                match self.ltn_strategy:
                    case 0:
                        # just use the embeeding class output of the network
                        self.p_sub_is_a = ltn_log.log.core.Predicate(model=PredFromEmbedding())

                        self.p_obj_is_a = ltn_log.log.core.Predicate(model=PredFromEmbedding())
                        self.p_sub_in_set = ltn_log.log.core.Predicate(model=EntInSet())
                        self.p_obj_in_set = ltn_log.log.core.Predicate(model=EntInSet())

                        gr_last_dim = self.num_classes * 2 + 7 + (self.num_rel_classes + 1)

                    case 1:
                        self.ground_entity = GroundVanilla(layer_sizes=(self.num_classes + 4 + 1, 64, self.num_classes))

                        self.p_sub_is_a = ltn_log.log.core.Predicate(model=GroundPred(self.ground_entity))
                        self.p_obj_is_a = ltn_log.log.core.Predicate(model=GroundPred(self.ground_entity))
                        gr_last_dim = (self.num_classes + 4 + 1) * 2 + 7 + (self.num_rel_classes + 1)
                    case _:
                        raise ValueError(f'ltn strategy {ltn_strategy} is invalid')

                if reltr_grounding == True:
                    if increase_dim_net == False:
                        self.ground_relationship = GroundVanilla(layer_sizes=(gr_last_dim, 64, self.num_rel_classes),
                                                                 p_dropout=p_dropout)


                    else:
                        self.ground_relationship = GroundVanilla(
                            layer_sizes=(gr_last_dim, 64, 128, self.num_rel_classes), p_dropout=p_dropout)

                    self.use_regressor = True
                    self.p_rel_in_set = ltn_log.log.core.Predicate(model=RelInSetL(self.ground_relationship))
                    self.p_rel_is_a = ltn_log.log.core.Predicate(model=RelIsAL(self.ground_relationship))

                else:
                    self.ground_relationship = None
                    self.use_regressor = False

                    self.p_rel_in_set = ltn_log.log.core.Predicate(model=RelInSetL(use_logit=False))
                    self.p_rel_is_a = ltn_log.log.core.Predicate(model=RelIsAL(use_logit=False))

            if not use_log:
                # and_ = ltn_log.Wrapper_Connective(ltn_log.fuzzy_ops.And_Sum())
                # or_ = ltn_log.Wrapper_Connective(ltn_log.fuzzy_ops.Or_LogSumExp(alpha=1))
                # implies = None
                # forall = ltn_log.Wrapper_Quantifier(ltn_log.fuzzy_ops.Aggreg_Mean(), semantics="forall")
                # and_aggreg = ltn_log.Wrapper_Formula_Aggregator(ltn_log.fuzzy_ops.Aggreg_Mean())
                # exists = ltn_log.Wrapper_Quantifier(ltn_log.fuzzy_ops.Aggreg_LogSumExp(alpha=1), semantics="exists")
                # or_aggreg = ltn_log.Wrapper_Formula_Aggregator(ltn_log.fuzzy_ops.Aggreg_LogSumExp(alpha=1))

                self.sat_aggr = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

                # satisfiability for debugging purpose
                self.sat_ent_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_rel_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_neg_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                self.sat_pos_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
                # float conversion is enough for fuzzification: account also for traffic label taxonomy
                self.ot_e_label = [ltn.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_classes), self.num_classes), 1)]
                # define constraint relationship here
                # impl one-hot enconding for constraints
                self.ot_r_label = [ltn.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_rel_classes), self.num_rel_classes),
                                               1)]
                self.entity_fuzzy_th = entity_fuzzy_th

                self.all_label_type = {}

                self.not_op = ltn.Connective(ltn.fuzzy_ops.NotStandard())

                if use_snc == True:
                    print('USING Second Negative Constraints formulation ')
                    self.constr_neg_type = 0
                    self.neg_indices = None
                    self.pos_indices = (1, 2)
                    self.rel_neg_const_map = self._parse_so4r_constraint(neg_constr_path)

                else:
                    print('USING First Negative Constraints formulation ')
                    self.constr_neg_type = 2
                    self.neg_indices = (0, 1)
                    self.pos_indices = (2, 3)

                    self.rel_neg_const_map = self._parse_r4so_constraint(neg_constr_path, self.neg_indices)
                    self.or_luk = ltn.Connective(ltn.fuzzy_ops.OrLuk())

            else:

                self.sat_aggr = ltn_log.log.core.Wrapper_Formula_Aggregator(ltn_log.log.fuzzy_ops.Aggreg_Sum())

                # satisfiability for debugging purpose
                self.sat_ent_axiom = ltn_log.log.core.Wrapper_Formula_Aggregator(ltn_log.log.fuzzy_ops.Aggreg_Sum())
                self.sat_rel_axiom = ltn_log.log.core.Wrapper_Formula_Aggregator(ltn_log.log.fuzzy_ops.Aggreg_Sum())
                self.sat_neg_axiom = ltn_log.log.core.Wrapper_Formula_Aggregator(ltn_log.log.fuzzy_ops.Aggreg_Sum())
                self.sat_pos_axiom = ltn_log.log.core.Wrapper_Formula_Aggregator(ltn_log.log.fuzzy_ops.Aggreg_Sum())
                # float conversion is enough for fuzzification: account also for traffic label taxonomy
                self.ot_e_label = [ltn.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_classes), self.num_classes), 1)]
                # define constraint relationship here
                # impl one-hot enconding for constraints
                self.ot_r_label = [ltn.Constant(t.t().float()) for t in
                                   torch.split(F.one_hot(torch.arange(0, self.num_rel_classes), self.num_rel_classes),
                                               1)]
                self.entity_fuzzy_th = entity_fuzzy_th

                self.all_label_type = {}

                self.not_op = ltn_log.Wrapper_Connective(ltn_log.fuzzy_ops.Not_Std())

                if use_snc == True:
                    print('USING Second Negative Constraints formulation ')
                    self.constr_neg_type = 0
                    self.neg_indices = None
                    self.pos_indices = (1, 2)
                    self.rel_neg_const_map = self._parse_so4r_constraint(neg_constr_path)

                else:
                    print('USING First Negative Constraints formulation ')
                    self.constr_neg_type = 2
                    self.neg_indices = (0, 1)
                    self.pos_indices = (2, 3)

                    self.rel_neg_const_map = self._parse_r4so_constraint(neg_constr_path, self.neg_indices)
                    self.or_luk = ltn_log.Wrapper_Formula_Aggregator(ltn_log.log.fuzzy_ops.Aggreg_LogSumExp(alpha=1))

            self.rel_pos_const_map =  self._parse_r4so_constraint(pos_constr_path, self.pos_indices)
                

            self.sat_debug ={'rel_axiom_sat' : -1.0, 'neg_axiom_sat' : -1.0, 'pos_axiom_sat' : -1.0, 'entity_axiom_sat' : -1.0}
            self.axiom_debug ={'rel_axiom' : -1.0, 'neg_axiom' : -1.0, 'pos_axiom' : -1.0, 'neg_easy_axiom' : -1.0, 'pos_aeasy_xiom' : -1.0, 'neg_hard_axiom' : -1.0, 'pos_hard_axiom' : -1.0, 'entity_axiom' : -1.0}
            
            self._save_sat_debug = False
            
            # NMS for predicition pruning
            self.sat_nms_enable = use_pred_nms
            self.sat_nms_threshold = sat_pred_nms
            
            self.max_bbox=max_nms_bbox

            
    @property
    def save_sat_debug(self):
        return self._save_sat_debug
        
    @save_sat_debug.setter
    def save_sat_debug(self, val:bool):
        self._save_sat_debug = val
        
    def _parse_type_constraint(self, rel_path:str, device:torch.device) -> torch.Tensor:
        """
        generate tensor containg all possible classes for a specific axiom, where the aggregator norm of universal quantification
        is designed a priori based on difficulty to learn those classes
        Parameters:
            rel_path(str): path containg the relationship classes to be saved
        
        Return:
            tensor containg all the relationship classes desired
        """

        df = pd.read_csv(rel_path)

        rel_tensor = torch.tensor(df.values, device=device)

        return rel_tensor
        
    def _parse_so4r_constraint(self, constr_path:str) -> Dict[Tuple[int, int], List[ltn.Constant] ]:
        """
        generate constraint  <if sub and obj, then not rel > given file where they are stored
        """
        constr = {}

        with open(constr_path, 'r') as fp:
            for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), fp.readlines() )):
                s, r, o = int(tr[0]), int(tr[1]), int(tr[2])
                if (s,o) not in constr.keys():
                    constr[(s,o)] = []

                constr[(s,o)].append( self.ot_r_label[r].value.t())

            # only for compact version
            for k, v in constr.items():
                constr[k]= torch.vstack(v)

        self.all_label_type[0] = torch.vstack( [constr[k] for k in  constr.keys()] )
        #sanity check
        correct_shape = sum( [ v.size(0) for v in  constr.values()] )
        assert self.all_label_type[0].size(0) == correct_shape, f'wrong concatenation  for neg {self.all_label_type[0].size()} vs {correct_shape}'
    
        return constr

    def _parse_r4so_constraint(self, constr_path:str, indices:Tuple[int]=(1,2)) -> Dict[Tuple[int, int], List[ltn.Constant] ]:
        """
        generate  <if rel, then not sub or not obj > given file where they are stored
        idx index used inside the tuple all possible constraints (2 type by construction)
        """
        constr = {}        
        used_label = {}

        with open(constr_path, 'r') as fp:
            for tr in list(map(lambda l: list(l.replace('\n', '').rsplit(',')), fp.readlines() )):
                # save the axiom:  s -> o
                s, r, o = int(tr[0]), int(tr[1]), int(tr[2])
                # save all implication related to s into a map
                
               
                if r not in constr.keys():
                    constr[r] = { 0:[], 1:[]}
                    used_label[r] = {0: [], 1: []}

                
                # choose label dependens on agent constraints or action ones
                
                # check if label not already present in list
                if  int(s) not in used_label[r][0]:
                    used_label[r][0].append(int(s))
                    
                    # print(f's({s})')
                    # print(f'{self.ot_e_label[int(s)]}')
                    
                    constr[r][0].extend([ self.ot_e_label[int(s)].value.t()  ])
                
                if  int(o)  not in  used_label[r][1]:
                    used_label[r][1].append(int(o))
                    constr[r][1].extend([ self.ot_e_label[int(o)].value.t()  ])
 
            # sanity check
            for r in constr[r].keys():
                assert r <len(self.ot_r_label), f'relationship {r} cannot be cannot as a constraint'            
            # only for compact version 
            for k, v in constr.items():
                constr[k][0]= torch.vstack(v[0])
                constr[k][1]= torch.vstack(v[1])

        self.all_label_type[indices[0]] = torch.vstack( [constr[k][0] for k in  constr.keys()] )
        #sanity check 
        correct_shape = sum( [ v[0].size(0) for v in  constr.values()] )
        assert self.all_label_type[indices[0]].size(0) == correct_shape, f'wrong concatenation  for neg {self.all_label_type[indices[0]].size()} vs {correct_shape}'
        
        self.all_label_type[indices[1]] = torch.vstack( [constr[k][1] for k in  constr.keys()] )
        #sanity check 
        correct_shape = sum( [ v[1].size(0) for v in  constr.values()] )
        assert self.all_label_type[indices[1]].size(0) == correct_shape, f'wrong concatenation  for neg {self.all_label_type[indices[1]].size()} vs {correct_shape}'
    
    
        return constr

    def change_aggr_p(self, aggr_p:int, univ_p:int=0, list_p:Optional[List[int]]=None):
        """change aggregator norm 

        Args:
            aggr_p (int): new p-norm for aggregator FolAll and SatAggr
            univ_p (int): new p-norm for aggregator FolAll and SatAggr
            aggr_p (int): new p-norm for aggregator FolAll and SatAggr
        """
        # self.forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p), quantifier="f")
        self.sat_aggr = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
        
        if self.save_sat_debug==True:
            self.sat_ent_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
            self.sat_neg_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))
            self.sat_pos_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

        
        if self.equal_p==True and univ_p!=0:
            self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
            self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
            self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=univ_p), quantifier="f")
            
        elif self.equal_p==False and list_p is not None:
            self.forall_neg = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[0]), quantifier="f")
            self.forall_pos = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[0]), quantifier="f")
            self.forall_hard = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[1]), quantifier="f")
            self.forall_ent_o_rel = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=list_p[-1]), quantifier="f")            
            
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Entity/subject/object Classification loss
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        # fill target_classes with value that it should be no object class
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
      

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']

        rel_idx = self._get_src_permutation_idx(indices[1])

        # return the class label of the target bbox involved in re
        target_rels_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        target_relo_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])

        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64, device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64, device=obj_logits.device)
        # file target with "real" object label
        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o

        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)

        if self.pure_nesy==False:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

            loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
            losses = {'loss_ce': (loss_ce * loss_weight).sum()/self.empty_weight[target_classes].sum()}
            
        else:            
            losses = {}



        if log:
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]

        if self.enable_ltn==True:
            rel_logits = outputs['rel_logits']
                            
            rel_target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
            rel_target_classes = torch.full(rel_logits .shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
            rel_target_classes[rel_idx] = rel_target_classes_o
                         
             
            idx_2 = rel_target_classes[rel_idx] < self.num_rel_classes
            
            if self.use_tgt_constr==True:

                              

                losses['sat_agg_error'] =  1 - self._compute_sat_level(outputs, sub_logits, obj_logits,  target_sub_classes, 
                                                                    target_obj_classes, rel_target_classes,  rel_target_classes[rel_idx][idx_2],  rel_logits[rel_idx][idx_2])

            else:
                sub_class = sub_logits.sigmoid().argmax(-1).squeeze(0)
                obj_class = obj_logits.sigmoid().argmax(-1).squeeze(0)
                rel_class = rel_logits.sigmoid().argmax(-1).squeeze(0)     
  

                losses['sat_agg_error'] =  1 - self._compute_sat_level(outputs, sub_logits, obj_logits, sub_class,
                                                                       obj_class, rel_class, rel_target_classes[rel_idx][idx_2],  rel_logits[rel_idx][idx_2])

                
        return losses
    

    def _compute_sat_level(self, outputs:Dict[str,torch.Tensor], sub_logits:torch.Tensor, obj_logits:torch.Tensor, target_sub_classes:torch.Tensor, 
                           target_obj_classes:torch.Tensor, rel_target_classes:torch.Tensor, ground_truth_rel:Optional[torch.Tensor]=None, rel_to_verify:Optional[torch.Tensor]=None):
        """
            compute sat level given output, target and logits,
        """
        sat_aggr = 0.0
        rel_logits = outputs['rel_logits']


        n_batch = rel_logits.size()[0]
          
        device = rel_logits.device
        if self.sat_nms_enable:
            
            sub_boxes = outputs['sub_boxes']
            obj_boxes = outputs['obj_boxes']
            # find sub and obj logits index that above a certaint threshold, then use the inclusion set to verifity
            # constraints satisfiability
            sub_scores, sub_labels = sub_logits.max(dim=-1)
            obj_scores, obj_labels = obj_logits.max(dim=-1)
            
            
            mask_nms_idx = torch.zeros(sub_scores.size(), device=device)
            for b in range(n_batch):
                sub_idx_pos = batched_nms(sub_boxes[b], sub_scores[b], sub_labels[b], self.sat_nms_threshold)
                obj_idx_pos = batched_nms(obj_boxes[b], obj_scores[b], obj_labels[b], self.sat_nms_threshold)
                

                # consider the corner case
                min_idx = min(sub_idx_pos.size(0), obj_idx_pos.size(0))
                
                # handlen nms retrive two different set
                if min_idx>self.max_bbox:
                    sub_idx = sub_idx_pos[:self.max_bbox] 
                    obj_idx = obj_idx_pos[:self.max_bbox] 
                    
                else:
                    sub_idx = sub_idx_pos[:min_idx]
                    obj_idx = obj_idx_pos[:min_idx]
                 
                # use intersection
   
                tmp_s  = torch.zeros(sub_scores[b].size(),device=device).scatter_(0, sub_idx,1).bool()
                tmp_o = torch.zeros(obj_scores[b].size(), device=device).scatter_(0, obj_idx, 1).bool()
                
                nms_idx = tmp_s & tmp_o
                mask_nms_idx[b][nms_idx] = 1
         
            mask_nms_idx = mask_nms_idx.bool()
            if torch.sum(mask_nms_idx)==0:
                                # no valid idx, then network not ready to evaluate the constraints
                if self._save_sat_debug:
                    self.sat_debug['neg_axiom_sat'] = -1.0
                    self.sat_debug['pos_axiom_sat'] = -1.0
                    
                    if self.rc_enable:
                        self.sat_debug['rel_axiom_sat'] = -1.0
                        
                    if self.ec_enable:
                        self.sat_debug['entity_axiom_sat'] = -1.0
            

                if self.equal_p:
                    self.axiom_debug['neg_axiom'] = -1.0  
                    self.axiom_debug['pos_axiom'] = -1.0   
                
                else:
                    self.axiom_debug['neg_easy_axiom'] = -1.0  
                    self.axiom_debug['neg_hard_axiom'] = -1.0       
                    self.axiom_debug['pos_easy_axiom'] = -1.0  
                    self.axiom_debug['pos_hard_axiom'] = -1.0    
                    
                if self.rc_enable:
                    self.axiom_debug['rel_axiom'] = -1.0
                
                if self.ec_enable:
                    self.axiom_debug['entity_axiom'] = -1.0
                    
                return 0

                    
            rel_valid_idx = (rel_target_classes < self.num_rel_classes) &  (target_sub_classes < self.num_classes) & (target_obj_classes < self.num_classes) & mask_nms_idx

        
          
        else: 
            # relationship generation
            rel_valid_idx = (rel_target_classes < self.num_rel_classes) &  (target_sub_classes < self.num_classes) & (target_obj_classes < self.num_classes)
            
            
        rel_valid_cls = rel_target_classes[ rel_valid_idx ]
        sub_valid_cls = target_sub_classes[ rel_valid_idx ]
        obj_valid_cls = target_obj_classes[ rel_valid_idx ]


        
      
        # sanity check
        assert rel_valid_cls.size()[0] == sub_valid_cls.size()[0], 'number of rel and sub must be equal'
        assert rel_valid_cls.size()[0] == obj_valid_cls.size()[0], 'number of rel and obj must be equal'


        # case eval sub, obj
        if sub_valid_cls.size()[0]>0 and obj_valid_cls.size()[0]>0 and  rel_valid_cls.size()[0]>0: 
            
            
            sub_boxes = outputs['sub_boxes'][rel_valid_idx]
            obj_boxes = outputs['obj_boxes'][rel_valid_idx]
            sub_logits = sub_logits[rel_valid_idx]
            obj_logits = obj_logits[rel_valid_idx]
            rel_logits = rel_logits[rel_valid_idx]


            if self.rc_enable:
                # check None as sanity check
                if rel_to_verify is not None and  ground_truth_rel is not None and rel_to_verify.size(0)>0 and ground_truth_rel.size(0)>0:
                    # check if the relationship must be verified 

                    has_rc = True
                    rel_axiom = self.eval_rel_constr(rel_to_verify, ground_truth_rel)
                    
                    if self._save_sat_debug:
                        self.sat_debug['rel_axiom_sat'] = self.sat_rel_axiom(rel_axiom)
                    else:
                        self.sat_debug['rel_axiom_sat'] = -1.0
                
                    self.axiom_debug['rel_axiom'] = rel_axiom.value.item()
                
                else:
                    has_rc  = False
                    if self._save_sat_debug:
                        self.sat_debug['rel_axiom_sat'] = -1.0
                    else:
                        self.sat_debug['rel_axiom_sat'] = -1.0
                
                    self.axiom_debug['rel_axiom'] = -1.0                   
            
            
            if self.ec_enable:  
                entity_axiom = self.eval_entity_constr(sub_boxes,  obj_boxes,
                                                       sub_logits, obj_logits, 
                                                       sub_valid_cls, obj_valid_cls)
                
                if self._save_sat_debug:
                    self.sat_debug['entity_axiom_sat'] = self.sat_ent_axiom(entity_axiom)
                else:
                    self.sat_debug['entity_axiom_sat'] = -1.0
                
                self.axiom_debug['entity_axiom'] = entity_axiom.value.item()
            
            

            if self.constr_neg_type==0:
                neg_constr_hit = len( list(filter(lambda so:  (so[0].item(), so[1].item()) in self.rel_neg_const_map.keys(), zip( sub_valid_cls,obj_valid_cls)   ) ) )
            
            else:
                neg_constr_hit = len( list(filter(lambda s:  s.item() in self.rel_neg_const_map.keys(),  rel_valid_cls ) ) ) 
                
            pos_constr_hit = len( list(filter(lambda s:  s.item() in self.rel_pos_const_map.keys(),  rel_valid_cls ) ) ) 
            
        
            if neg_constr_hit>0:
                
                if self.constr_neg_type==0:
                    if self.equal_p:
                        neg_axiom = self.eval_neg_constr_v1(sub_boxes,  obj_boxes,
                                                        sub_logits, obj_logits , rel_logits,
                                                        sub_valid_cls, obj_valid_cls, rel_valid_cls) 
                     
                        if self._save_sat_debug:
                            self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_axiom)
                        else:
                            self.sat_debug['neg_axiom_sat'] = -1.0
                        
                        self.axiom_debug['neg_axiom'] = neg_axiom.value.item()
                    else:
                        raise ValueError('SNC cannot use constraints differentiation!')
                else:
        
                    if self.equal_p:
                        neg_axiom = self.eval_neg_constr_v2(sub_boxes,  obj_boxes,
                                                    sub_logits, obj_logits, rel_logits,
                                                    sub_valid_cls, obj_valid_cls, rel_valid_cls, forall_type=0)
                        if self._save_sat_debug:
                            self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_axiom)
                        else:
                            self.sat_debug['neg_axiom_sat'] = -1.0

                        
                        self.axiom_debug['neg_axiom'] = neg_axiom.value.item()
                
                    else:
                        rel_easy_idx = self.get_rel4axiom_type(rel_valid_cls, 0)
                        if torch.sum(rel_easy_idx)>0:
                            rel_easy_n_valid_cls = rel_valid_cls[rel_easy_idx]
                            neg_easy_axiom = self.eval_neg_constr_v2(sub_boxes,  obj_boxes,
                                                        sub_logits, obj_logits, rel_logits,
                                                        sub_valid_cls, obj_valid_cls, rel_easy_n_valid_cls, forall_type=0)
                            
                            n_have_easy = True
                        else:     
                            n_have_easy = False         
                        
                
                        rel_hard_idx= self.get_rel4axiom_type(rel_valid_cls, 1)
                        
                        if torch.sum(rel_hard_idx)>0:
                            rel_hard_n_valid_cls = rel_valid_cls[rel_hard_idx]                

                            neg_hard_axiom = self.eval_neg_constr_v2(sub_boxes,  obj_boxes,
                                                        sub_logits, obj_logits, rel_logits,
                                                        sub_valid_cls, obj_valid_cls, rel_hard_n_valid_cls, forall_type=1)
                            
                            n_have_hard = True
                        
                        else:
                            n_have_hard = False
                        
                        if self._save_sat_debug:
                            if n_have_hard and n_have_easy:
                                self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_easy_axiom, neg_hard_axiom)
                            
                            elif n_have_hard:
                                self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_hard_axiom)
                            
                            elif n_have_easy:
                                self.sat_debug['neg_axiom_sat'] = self.sat_neg_axiom(neg_easy_axiom)
                            else:
                                self.sat_debug['neg_axiom_sat'] = -1.0
                            
                                
                            
                        
                        if n_have_easy:
                            self.axiom_debug['neg_easy_axiom'] = neg_easy_axiom.value.item()
                        else:
                            self.axiom_debug['neg_easy_axiom'] = -1.0  
                            
                        if n_have_hard:
                            self.axiom_debug['neg_hard_axiom'] = neg_hard_axiom.value.item()
                    
                        else:
                            self.axiom_debug['neg_hard_axiom'] = -1.0                        
        
            if pos_constr_hit>0:
                if self.equal_p:
                    pos_axiom = self.eval_pos_constr(sub_boxes,  obj_boxes,
                                                    sub_logits, obj_logits, rel_logits,
                                                    sub_valid_cls, obj_valid_cls, rel_valid_cls, forall_type=0)
                    
                
                    if self._save_sat_debug:
                        self.sat_debug['pos_axiom_sat'] = self.sat_pos_axiom(pos_axiom)
                    else:
                        self.sat_debug['pos_axiom_sat'] = -1.0
                        
                    self.axiom_debug['pos_axiom'] = pos_axiom.value.item()
                else:
                    rel_easy_idx = self.get_rel4axiom_type(rel_valid_cls, 0)
                    if torch.sum(rel_easy_idx)>0:
                        rel_easy_p_valid_cls = rel_valid_cls[rel_easy_idx]
                        pos_easy_axiom = self.eval_pos_constr(sub_boxes,  obj_boxes,
                                                        sub_logits, obj_logits, rel_logits,
                                                        sub_valid_cls, obj_valid_cls, rel_easy_p_valid_cls, forall_type=0) 
                        
                        p_have_easy = True
                        
                    else:     
                        p_have_easy = False         
                      
                                       
                    rel_hard_idx= self.get_rel4axiom_type(rel_valid_cls, 1)
                    
                    if torch.sum(rel_hard_idx)>0:
                        rel_hard_p_valid_cls = rel_valid_cls[rel_hard_idx]    
                    
                        pos_hard_axiom = self.eval_pos_constr(sub_boxes,  obj_boxes,
                                                        sub_logits, obj_logits, rel_logits,
                                                        sub_valid_cls, obj_valid_cls, rel_hard_p_valid_cls, forall_type=1)   
                                        
                        p_have_hard = True
                    else:
                        p_have_hard = False
                        
                    if self._save_sat_debug:
                        if p_have_hard and p_have_easy:
                            self.sat_debug['pos_axiom_sat'] = self.sat_pos_axiom(pos_easy_axiom, pos_hard_axiom)
                            
                        elif p_have_hard:
                            self.sat_debug['pos_axiom_sat'] = self.sat_neg_axiom(pos_hard_axiom)
                        
                        elif p_have_easy:
                            self.sat_debug['pos_axiom_sat'] = self.sat_neg_axiom(pos_easy_axiom)
                        else:
                            self.sat_debug['pos_axiom_sat'] = -1.0
                                            
                        
                    if p_have_easy:
                        self.axiom_debug['pos_easy_axiom'] = pos_easy_axiom.value.item()
                    else:
                        self.axiom_debug['pos_easy_axiom'] = -1.0  
                        
                    if p_have_hard:
                        self.axiom_debug['pos_hard_axiom'] = pos_hard_axiom.value.item()
                
                    else:
                        self.axiom_debug['pos_hard_axiom'] = -1.0        
                    
                    
            
            # sat aggregation on constraints' predicates
            if neg_constr_hit>0 and pos_constr_hit>0:
                # we only consider that case where pos and neg constrain have same relationship set fro all different p
                if self.ec_enable:
                    
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(entity_axiom, neg_axiom, pos_axiom)
                    else:
                      
                        if p_have_hard and p_have_easy:
                            sat_aggr += self.sat_aggr(entity_axiom, neg_easy_axiom, neg_hard_axiom, pos_easy_axiom, pos_hard_axiom)
                            
                        elif p_have_hard:
                            sat_aggr += self.sat_aggr(entity_axiom,neg_hard_axiom,pos_hard_axiom)
                        
                        elif p_have_easy:
                            sat_aggr += self.sat_aggr(entity_axiom, neg_easy_axiom, pos_easy_axiom)    
                        
                        
                    
                elif self.rc_enable and has_rc:
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(rel_axiom, neg_axiom, pos_axiom)
                        
                    else:
                        if p_have_hard and p_have_easy:
                            sat_aggr += self.sat_aggr(rel_axiom, neg_easy_axiom, neg_hard_axiom, pos_easy_axiom, pos_hard_axiom)
                            
                        elif p_have_hard:
                            sat_aggr += self.sat_aggr(rel_axiom,neg_hard_axiom,pos_hard_axiom)
                        
                        elif p_have_easy:
                            sat_aggr += self.sat_aggr(rel_axiom, neg_easy_axiom, pos_easy_axiom)    
                    
                else:
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(neg_axiom, pos_axiom)
                    
                    else:
                        if p_have_hard and p_have_easy:
                            sat_aggr += self.sat_aggr( neg_easy_axiom, neg_hard_axiom, pos_easy_axiom, pos_hard_axiom)
                            
                        elif p_have_hard:
                            sat_aggr += self.sat_aggr(neg_hard_axiom,pos_hard_axiom)
                        
                        elif p_have_easy:
                            sat_aggr += self.sat_aggr(neg_easy_axiom, pos_easy_axiom)    
                
            elif neg_constr_hit>0:
                if self.ec_enable:
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(entity_axiom, neg_axiom)
                    else:
                        if n_have_hard and n_have_easy:
                            sat_aggr += self.sat_aggr(entity_axiom, neg_easy_axiom, neg_hard_axiom)
                            
                        elif n_have_hard:
                            sat_aggr += self.sat_aggr(entity_axiom,neg_hard_axiom)
                        
                        elif n_have_easy:
                            sat_aggr += self.sat_aggr(entity_axiom, neg_easy_axiom)    
                        
                    
                elif self.rc_enable and has_rc:
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(rel_axiom, neg_axiom)
                        
                    else:
                        if n_have_hard and n_have_easy:
                            sat_aggr += self.sat_aggr(rel_axiom, neg_easy_axiom, neg_hard_axiom)
                            
                        elif n_have_hard:
                            sat_aggr += self.sat_aggr(rel_axiom,neg_hard_axiom,)
                        
                        elif n_have_easy:
                            sat_aggr += self.sat_aggr(rel_axiom, neg_easy_axiom)  
                    
                else:
                        if n_have_hard and n_have_easy:
                            sat_aggr += self.sat_aggr( neg_easy_axiom, neg_hard_axiom)
                            
                        elif n_have_hard:
                            sat_aggr += self.sat_aggr(neg_hard_axiom)
                        
                        elif n_have_easy:
                            sat_aggr += self.sat_aggr(neg_easy_axiom)    
                    
                
                if self.save_sat_debug:                
                    self.sat_debug['pos_axiom_sat'] = -1.0
                
                if self.equal_p:
                    self.axiom_debug['pos_axiom'] = -1.0
                
                else:
                    self.axiom_debug['pos_easy_axiom'] = -1.0  
                    self.axiom_debug['pos_hard_axiom'] = -1.0
                    
                    
            elif pos_constr_hit>0:
                if self.ec_enable:
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(entity_axiom, pos_axiom)
                    
                    else:
                                                # we only consider that case where pos and neg constrain have same relationship set
                        if p_have_hard and p_have_easy:
                            sat_aggr += self.sat_aggr(entity_axiom, pos_easy_axiom, pos_hard_axiom)
                            
                        elif p_have_hard:
                            sat_aggr += self.sat_aggr(entity_axiom,pos_hard_axiom)
                        
                        elif p_have_easy:
                            sat_aggr += self.sat_aggr(entity_axiom, pos_easy_axiom)    
                    
                elif self.rc_enable and has_rc:
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(rel_axiom, pos_axiom)
                        
                    else:
                        if p_have_hard and p_have_easy:
                            sat_aggr += self.sat_aggr(rel_axiom, pos_easy_axiom, pos_hard_axiom)
                            
                        elif p_have_hard:
                            sat_aggr += self.sat_aggr(rel_axiom,pos_hard_axiom)
                        
                        elif p_have_easy:
                            sat_aggr += self.sat_aggr(rel_axiom, pos_easy_axiom)    
                                    
                else:
                    if self.equal_p:
                        sat_aggr += self.sat_aggr(pos_axiom)
                        
                    else:
                        if p_have_hard and p_have_easy:
                            sat_aggr += self.sat_aggr( pos_easy_axiom, pos_hard_axiom)
                            
                        elif p_have_hard:
                            sat_aggr += self.sat_aggr(pos_hard_axiom)
                        
                        elif p_have_easy:
                            sat_aggr += self.sat_aggr(pos_easy_axiom)   
                
                if self._save_sat_debug:  
                    self.sat_debug['neg_axiom_sat'] = -1.0
                    
                if self.equal_p:
                    self.axiom_debug['neg_axiom'] = -1.0
                
                else:
                    self.axiom_debug['neg_easy_axiom'] = -1.0  
                    self.axiom_debug['neg_hard_axiom'] = -1.0                
                        
                    
                
            elif self.ec_enable:
                sat_aggr += self.sat_aggr(entity_axiom)
                
                if self._save_sat_debug:
                    self.sat_debug['entity_axiom_sat'] = self.sat_ent_axiom(entity_axiom)
                    self.sat_debug['neg_axiom_sat'] = -1.0
                    self.sat_debug['pos_axiom_sat'] = -1.0
                    
                if self.equal_p:
                    self.axiom_debug['neg_axiom'] = -1.0  
                    self.axiom_debug['pos_axiom'] = -1.0   
                
                else:
                    self.axiom_debug['neg_easy_axiom'] = -1.0  
                    self.axiom_debug['neg_hard_axiom'] = -1.0       
                    self.axiom_debug['pos_easy_axiom'] = -1.0  
                    self.axiom_debug['pos_hard_axiom'] = -1.0                      
                         
            elif self.rc_enable:
                sat_aggr += self.sat_aggr(rel_axiom)
                
                if self._save_sat_debug:
                    self.sat_debug['neg_axiom_sat'] = -1.0
                    self.sat_debug['pos_axiom_sat'] = -1.0
                    
                if self.equal_p:
                    self.axiom_debug['neg_axiom'] = -1.0  
                    self.axiom_debug['pos_axiom'] = -1.0   
                
                else:
                    self.axiom_debug['neg_easy_axiom'] = -1.0  
                    self.axiom_debug['neg_hard_axiom'] = -1.0       
                    self.axiom_debug['pos_easy_axiom'] = -1.0  
                    self.axiom_debug['pos_hard_axiom'] = -1.0        
                
        else:
            if self._save_sat_debug:
                    self.sat_debug['rel_axiom_sat'] = -1.0
                    self.sat_debug['entity_axiom_sat'] = -1.0
                    self.sat_debug['neg_axiom_sat'] = -1.0
                    self.sat_debug['pos_axiom_sat'] = -1.0
                    
            
            self.axiom_debug['rel_axiom'] = -1.0
            self.axiom_debug['entity_axiom'] = -1.0
            
            if self.equal_p:
                self.axiom_debug['neg_axiom'] = -1.0  
                self.axiom_debug['pos_axiom'] = -1.0   
            
            else:
                self.axiom_debug['neg_easy_axiom'] = -1.0  
                self.axiom_debug['neg_hard_axiom'] = -1.0       
                self.axiom_debug['pos_easy_axiom'] = -1.0  
                self.axiom_debug['pos_hard_axiom'] = -1.0        
                            


        return  sat_aggr/n_batch
    
    def get_rel4axiom_type(self, rel_cls:torch.Tensor, difficulty:int):
        """
        return the proper index according to difficulty to learn that particular classes present in the sensor
        
        rel_cls(torch.Tensor): tensor of classes to be pruned 
        difficulty(int): the difficulty of class to retrieve
                    0 -> easy class to respect axioms
                    1 -> hard class to respect axioms
        """
        
        # dummy initalization, just for code compactness
        valid_idx = torch.zeros( rel_cls.size(), dtype=bool, device=rel_cls.device)
        
        
        if difficulty==0:
            for i,r in enumerate(rel_cls):
                if r in self.rel_cls_easy:
                    valid_idx[i]= True
        elif difficulty==1:
            for i,r in enumerate(rel_cls):
                if r in self.rel_cls_hard:
                    valid_idx[i]= True
        
        else:
            raise ValueError(f'relationship\'s axiom difficulty=={difficulty}: it must be either 0 or 1')
        

       
        return valid_idx
      
  

    def eval_entity_constr(self, sub_boxes, obj_boxes, sub_logits, obj_logits, sub_valid_cls, obj_valid_cls):
        gr_sub, gr_obj, _, _ , _ = grounding(sub_boxes, obj_boxes,
                                            sub_logits, obj_logits,
                                            self.ltn_strategy)
                
        cls_s = ltn.Variable(f'cls_s', torch.hstack([self.ot_e_label[cls].value for cls in sub_valid_cls]).t() )
        cls_o = ltn.Variable(f'cls_o', torch.hstack([self.ot_e_label[cls].value for cls in obj_valid_cls]).t() )

        axiom_entity = self.forall_ent_o_rel( 
                            ltn.diag( gr_sub, cls_s, gr_obj,  cls_o),
                            self.and_luk( self.p_sub_is_a(gr_sub, cls_s), self.p_obj_is_a(gr_obj, cls_o) )
                            )

        return axiom_entity
    

    def eval_rel_constr(self, rel_logits, rel_gt_cls ):
        """
        evaluated relationships satisfiability wrt to ground-truth
        """
        gr_rel = ltn.Variable(f'gr_rel_box', rel_logits[:, :-1])


        try:
            cls_r = ltn.Variable(f'cls_r', torch.hstack([self.ot_r_label[cls].value for cls in  rel_gt_cls ]).t() )
            
            
            axiom_rel = self.forall_ent_o_rel( 
                                ltn.diag( gr_rel, cls_r),
                                self.p_rel_is_a(gr_rel, cls_r)
                                )
        
        except Exception as e:
            print(f'EXCEPTION:{e}')
            
            
            print(f'rel_gt_cls:{rel_gt_cls}')

            print(f'pytorch seed:{torch.get_rng_state()}')
            print(f'numpy seed:{np.random.get_state()}')
            print(f'random seed:{random.getstate()}')
            raise e
        


        return axiom_rel
        
    def eval_neg_constr_v1(self,  sub_boxes, obj_boxes, sub_logits, obj_logits,  rel_logits, sub_valid_cls, obj_valid_cls, rel_valid_cls):
        """
        evaluated the negative constraints axiom in a fuzzy-fashion
        """

     
        idx_with_constr = torch.stack( tuple(map(lambda o: torch.tensor(o[0]) , filter(lambda so:  (so[1][0].item(), so[1][1].item()) in self.rel_neg_const_map.keys(),  enumerate(zip( sub_valid_cls,obj_valid_cls))))  ) ) 
  

        cls_s_tensor =   torch.hstack([self.ot_e_label[cls].value for cls in sub_valid_cls[ idx_with_constr  ]]).t()
        cls_o_tensor =   torch.hstack([self.ot_e_label[cls].value for cls in obj_valid_cls[ idx_with_constr  ]]).t()
        cls_r_tensor =   torch.hstack([self.ot_r_label[cls].value for cls in rel_valid_cls[ idx_with_constr  ]]).t()
            

        sub_logits_valid = sub_logits[idx_with_constr]
        obj_logits_valid = obj_logits[idx_with_constr]
        rel_logits_valid = rel_logits[idx_with_constr]
            
        out_sub_box = sub_boxes[idx_with_constr]
        out_obj_box = obj_boxes[idx_with_constr]
                
        
        assert cls_s_tensor.size(0) == cls_o_tensor.size(0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_o_tensor size ({cls_o_tensor.size(0)})'
        assert cls_s_tensor.size(0) == cls_r_tensor.size(0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'
       
        gr_sub, gr_obj, gr_rel , mask_idx_tensor, constr_tensor = grounding(out_sub_box, out_obj_box ,
                                                                                sub_logits_valid, obj_logits_valid,
                                                                                self.ltn_strategy,rel_logits_valid,
                                                                                cls_s_tensor, cls_o_tensor, cls_r_tensor,
                                                                                self.rel_neg_const_map, self.device,
                                                                                self.all_label_type, self.neg_indices,
                                                                                constr_type=self.constr_neg_type,
                                                                                use_regressor=self.use_regressor)
            
        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        cls_s = ltn.Variable(f'cls_s', cls_s_tensor)
        cls_o = ltn.Variable(f'cls_o', cls_o_tensor)
        cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

        mask_idx   = ltn.Variable(f'mask_id', mask_idx_tensor)
        neg_constr = ltn.Variable(f'neg_constr', constr_tensor)
        

        axiom_fuzzy = self.forall_neg( ltn.diag( gr_sub, cls_s, gr_obj, cls_o, gr_rel, cls_r,  mask_idx, neg_constr), 
                            self.rel_operator(
                                self.and_luk( self.p_sub_is_a(gr_sub, cls_s), self.p_obj_is_a(gr_obj, cls_o) ),
                                self.not_op(self.p_rel_in_set(gr_rel, mask_idx, neg_constr, training=True))
                                
                            )
                    )



        return axiom_fuzzy 
    
    
    def eval_pos_constr(self, sub_boxes, obj_boxes,  sub_logits, obj_logits,  rel_logits, sub_valid_cls, obj_valid_cls, rel_valid_cls, forall_type):
        """
        evaluated the negative constraints axiom in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty, 
                           1 hard constraints
        """

        idx_with_constr = torch.stack( tuple(map(lambda o: torch.tensor(o[0]) , filter(lambda r:  r[1].item() in self.rel_pos_const_map.keys() ,  enumerate(rel_valid_cls )))  ) ) 

        
        cls_s_tensor =   torch.hstack([self.ot_e_label[cls].value for cls in sub_valid_cls[ idx_with_constr  ]]).t()
        cls_o_tensor =   torch.hstack([self.ot_e_label[cls].value for cls in obj_valid_cls[ idx_with_constr  ]]).t()
        cls_r_tensor =   torch.hstack([self.ot_r_label[cls].value for cls in rel_valid_cls[ idx_with_constr  ]]).t()
            

        sub_logits_valid = sub_logits[idx_with_constr]
        obj_logits_valid = obj_logits[idx_with_constr]
        rel_logits_valid = rel_logits[idx_with_constr]
            
        out_sub_box = sub_boxes[idx_with_constr]
        out_obj_box = obj_boxes[idx_with_constr]
                
        assert cls_s_tensor.size(0) == cls_o_tensor.size(0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_o_tensor size ({cls_o_tensor.size(0)})'
        assert cls_s_tensor.size(0) == cls_r_tensor.size(0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'
       
        
        gr_sub, gr_obj, gr_rel , mask_idx_tensor, pos_contr_tensor = grounding(out_sub_box, out_obj_box ,
                                                                                sub_logits_valid, obj_logits_valid,
                                                                                self.ltn_strategy,rel_logits_valid,
                                                                                cls_s_tensor, cls_o_tensor, cls_r_tensor,
                                                                                self.rel_pos_const_map, self.device,
                                                                                self.all_label_type, self.pos_indices,
                                                                                constr_type=1,
                                                                                use_regressor=self.use_regressor)
            
        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'


        cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

        mask_idx_s   = ltn.Variable(f'mask_id', mask_idx_tensor[0])
        pos_constr_s = ltn.Variable(f'pos_constr', pos_contr_tensor[0])
        mask_idx_o   = ltn.Variable(f'mask_id', mask_idx_tensor[1])
        pos_constr_o = ltn.Variable(f'pos_constr', pos_contr_tensor[1])
        
        # 
        if forall_type==0:
            axiom_fuzzy = self.forall_pos( ltn.diag( gr_sub, gr_obj, gr_rel, cls_r, 
                                            mask_idx_s, pos_constr_s, mask_idx_o, pos_constr_o), 
                            self.rel_operator(
                                self.p_rel_is_a(gr_rel, cls_r ),
                                self.and_luk( self.p_sub_in_set(gr_sub, mask_idx_s, pos_constr_s),
                                             self.p_obj_in_set(gr_obj, mask_idx_o, pos_constr_o ) )
                                
                                
                            )
                    )
        elif forall_type==1:
            axiom_fuzzy = self.forall_hard( ltn.diag( gr_sub, gr_obj, gr_rel, cls_r, 
                                            mask_idx_s, pos_constr_s, mask_idx_o, pos_constr_o), 
                            self.rel_operator(
                                self.p_rel_is_a(gr_rel, cls_r ),
                                self.and_luk( self.p_sub_in_set(gr_sub, mask_idx_s, pos_constr_s),
                                             self.p_obj_in_set(gr_obj, mask_idx_o, pos_constr_o ) )
                                
                                
                            )
                    )        
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,1]')           
            

        return axiom_fuzzy     
    
    def eval_neg_constr_v2(self, sub_boxes, obj_boxes,  sub_logits, obj_logits,  rel_logits, sub_valid_cls, obj_valid_cls, rel_valid_cls, forall_type):
        """
        evaluated the negative constraints axiom in a fuzzy-fashion
        forall_type(bool): 0 easy or differentiation among axiom difficulty, 
                           1 hard constraints
        """

        idx_with_constr = torch.stack( tuple(map(lambda o: torch.tensor(o[0]) , filter(lambda r:  r[1].item() in self.rel_neg_const_map.keys() ,  enumerate(rel_valid_cls )))  ) ) 

        
        cls_s_tensor =   torch.hstack([self.ot_e_label[cls].value for cls in sub_valid_cls[ idx_with_constr  ]]).t()
        cls_o_tensor =   torch.hstack([self.ot_e_label[cls].value for cls in obj_valid_cls[ idx_with_constr  ]]).t()
        cls_r_tensor =   torch.hstack([self.ot_r_label[cls].value for cls in rel_valid_cls[ idx_with_constr  ]]).t()
            

        sub_logits_valid = sub_logits[idx_with_constr]
        obj_logits_valid = obj_logits[idx_with_constr]
        rel_logits_valid = rel_logits[idx_with_constr]
            
        out_sub_box = sub_boxes[idx_with_constr]
        out_obj_box = obj_boxes[idx_with_constr]
                
        assert cls_s_tensor.size(0) == cls_o_tensor.size(0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_o_tensor size ({cls_o_tensor.size(0)})'
        assert cls_s_tensor.size(0) == cls_r_tensor.size(0), f'cls_s_tensor size({cls_s_tensor.size(0)}) different from cls_r_tensor size ({cls_r_tensor.size(0)})'
       
        
        gr_sub, gr_obj, gr_rel , mask_idx_tensor, neg_contr_tensor = grounding(out_sub_box, out_obj_box ,
                                                                                sub_logits_valid, obj_logits_valid,
                                                                                self.ltn_strategy,rel_logits_valid,
                                                                                cls_s_tensor, cls_o_tensor, cls_r_tensor,
                                                                                self.rel_neg_const_map, self.device,
                                                                                self.all_label_type, self.neg_indices,
                                                                                constr_type=self.constr_neg_type,
                                                                                use_regressor=self.use_regressor)
            
        assert mask_idx_tensor is not None, f'mask_idx_tensor cannot be None'

        cls_r = ltn.Variable(f'cls_r', cls_r_tensor)

        mask_idx_s   = ltn.Variable(f'mask_id', mask_idx_tensor[0])
        neg_constr_s = ltn.Variable(f'neg_constr', neg_contr_tensor[0])
        mask_idx_o   = ltn.Variable(f'mask_id', mask_idx_tensor[1])
        neg_constr_o = ltn.Variable(f'neg_constr', neg_contr_tensor[1])
        
        if forall_type==0:
            axiom_fuzzy = self.forall_neg( ltn.diag( gr_sub, gr_obj, gr_rel, cls_r, 
                                                mask_idx_s, neg_constr_s, mask_idx_o, neg_constr_o), 
                                self.rel_operator(
                                    self.p_rel_is_a(gr_rel, cls_r ),
                                    self.and_luk( self.not_op( self.p_sub_in_set(gr_sub, mask_idx_s, neg_constr_s) ),
                                                self.not_op( self.p_obj_in_set(gr_obj, mask_idx_o, neg_constr_o ) )
                                    )
                                    
                                )
                        )       

        elif forall_type==1:
            axiom_fuzzy = self.forall_hard( ltn.diag( gr_sub, gr_obj, gr_rel, cls_r, 
                                        mask_idx_s, neg_constr_s, mask_idx_o, neg_constr_o), 
                        self.rel_operator(
                            self.p_rel_is_a(gr_rel, cls_r ),
                            self.and_luk( self.not_op( self.p_sub_in_set(gr_sub, mask_idx_s, neg_constr_s) ),
                                        self.not_op( self.p_obj_in_set(gr_obj, mask_idx_o, neg_constr_o ) )
                            )
                            
                        )
                )        
        else:
            raise ValueError(f'Invalid for_all type={forall_type}: it must be in [0,2]')           

        return axiom_fuzzy 
        
          

    @torch.no_grad()
    def loss_cardinality(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], indices:torch.Tensor,  num_boxes:int):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], indices:torch.Tensor,  num_boxes:int):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], indices:torch.Tensor,  num_boxes:int, log:bool=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        self.rel_assign = target_classes
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses



    def loss_kge_alignment(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], indices:torch.Tensor,  num_boxes:int ) -> float:
        """Computer the Knowledge Graph Embeeding Contrastive loss given different strategy

        Args:
            outputs (Dict[torch.Tensor]): outputs from the forward method of the RelTr network
            targets (targets: Dict[torch.Tensor]): targets to be cmp with outputs  from the forward method of the RelTr network
            indices (indices:torch.Tenso): bounding box indices

        Returns:
            contr_loss (float): constrastive loss according to the strategy defined by constant STRATEGY in this module
        """

        losses = {'loss_kg_align': 0.0}

        # useful check to whether there was a relationship or not in the img
        if 'cls_index_hs' in outputs.keys() and 'entity_emb' in outputs.keys():

            # extract boduning in latent space
            cls_emb = outputs['entity_emb']
            # n_ent = cls_emb .size()[-2]

            cls_map = outputs['cls_index_hs']


            match self.sim_strategy:
                case 0:
                    losses['loss_kg_align'] += sum([self._cos_emb_sim(b, cls_emb, d_pos)  for b, d_pos in enumerate(cls_map) ])
                case 1:
                    losses['loss_kg_align'] += sum([self._hinge_sim(b, cls_emb, d_pos)  for b, d_pos in enumerate(cls_map) ])

                case _:
                    raise ValueError(f'kge strategy {self.sim_strategy} is invalid')





        return losses



    def _hinge_sim(self, batch:int, cls_emb: torch.Tensor, d_pos:List[torch.Tensor]) -> torch.Tensor:
        cum_loss = 0.0
        n_ent = 0
        
        for k, v  in d_pos.items():
            # if there is no prior knowledge loss remains has it is
            if len(v)>0 and k in MAP_PD2TG.keys():
                        # naive approach: use only index related to cls
                anchor_node =  self.prior_kge.node_emb( torch.tensor(self.key2idx[MAP_PD2TG[k]]).to(self.device) )


                loss_sim = sum([ self.similarity(self.pair_distance(anchor_node,cls_emb[batch][cv]), torch.tensor( 1 if ki == k else -1) )  for cv in  v for ki, v  in d_pos.items() ]  )

                n_ent+=len(v)

                cum_loss += loss_sim
        if cum_loss>0:
            cum_loss=cum_loss/n_ent
            
        return cum_loss

    def _cos_emb_sim(self, batch:int, cls_emb: torch.Tensor, d_pos:List[torch.Tensor]) -> torch.Tensor:
        cum_loss = 0.0
        
        n_ent = 0
        for k, v  in d_pos.items():
                    # if there is no prior knowledge loss remains has it is
            if len(v)>0 and k in MAP_PD2TG.keys():
                        # naive approach: use only index related to cls
                anchor_node =  self.prior_kge.node_emb( torch.tensor(self.key2idx[MAP_PD2TG[k]]).to(self.device) )

                loss_sim = sum([ self.similarity(anchor_node, cls_emb[batch][cv], torch.tensor( 1 if ki == k else -1) )  for cv in  v for ki, v  in d_pos.items() ]  )

                # TODO understand whether useless or not
                n_ent+=len(v)

                cum_loss += loss_sim
        if cum_loss>0:
            cum_loss=cum_loss/n_ent
            
        return cum_loss


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'kge_alignment': self.loss_kge_alignment
            # 'constraints_satisfaction': self.loss_constr_sat
        }
        
        if self.pure_nesy==False:
            loss_map['relations'] = self.loss_relations

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"])+len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
               
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results




def build(args):

    # add background class for vg compatibility, whihc is added in reltr architecture
    # since all imgs have either relationship or class
    
    num_classes = 35 #if args.dataset == 'tg' else 26
    num_rel_classes = 51 
    
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)


    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss,
        enable_kg=args.enable_kg,
        enable_ltn=args.enable_ltn,
    )

    weight_dict = {}
    weight_dict['loss_bbox']= args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    # use aso non symbolic loss for classification
    
    losses = []
    if args.pure_nesy==False:    
        weight_dict['loss_ce'] = 1 
        weight_dict['loss_rel'] = args.rel_loss_coef
        losses.append('relations')
    else:
        # enable all nesy param just for safety
        args.ec_enable = True
        args.rc_enable = True

    losses.extend(['labels', 'boxes', 'cardinality'])
        

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)


    if args.enable_kg:

        prior_kge=build_kge(device, args.kge_path)


        losses.append('kge_alignment')
        weight_dict['loss_kg_align'] = args.kg_align_coef




        criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses,
                                 prior_kge=prior_kge,
                                 sim_margin=args.margin_loss,
                                 sim_strategy=args.prior_strategy,
                                 reltr_grounding=args.reltr_grounding  
        )

    elif args.enable_ltn:
        # losses.append('constraints_satisfaction')
        
        weight_dict['sat_agg_error'] = args.sat_loss_coef
        criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses,
                                 enable_ltn=True, ltn_strategy=args.prior_strategy,
                                 rel_operator=args.rel_operator,
                                 neg_constr_path=args.neg_constr_path,
                                 pos_constr_path=args.pos_constr_path,
                                 increase_dim_net=args.incr_gr,
                                 p_dropout=args.dropout_gr,
                                 aggr_p= args.aggr_p,
                                 univ_p= args.univ_p,
                                 different_p= None if args.different_p == False else [args.p_easy_constr, args.p_hard_constr, args.rel_p ],
                                 reltr_grounding=args.reltr_grounding,
                                 tgt_as_constr=args.tgt_as_constr,
                                 ec_enable=args.ec_enable ,
                                 rc_enable=args.rc_enable,
                                 use_pred_nms= args.nms_sat_enable,
                                 sat_pred_nms=args.nms_sat,
                                 pure_nesy=args.pure_nesy,
                                 use_snc=args.use_snc                                              
        )       

        print(f'rc_enable:{args.rc_enable}')
    else:
        criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, device=device, reltr_grounding=args.reltr_grounding)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors



