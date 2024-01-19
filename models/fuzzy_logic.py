# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved


from typing import Tuple, Dict, Optional
import torch, ltn
from utils import box_ops

class GroundPred(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    
    def __init__(self,model:torch.nn.Module):
        super(GroundPred, self).__init__()

        self.logit_model = model
        self.softmax = torch.nn.Softmax(dim=1)
        

    def forward(self, x:torch.Tensor, label: torch.Tensor):
       
        # batch dim is the number of grounding to need to squeeze to done it properly
        logits = self.logit_model(x)
        probs = self.softmax(logits)

        out = torch.sum(probs*label, dim = 1)
      
        assert out.size()[0] == x.size()[0], 'input tensor dimension is wrong'
        return out
    
class RelInSetL(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    
    def __init__(self, model:Optional[torch.nn.Module]=None, use_logit:bool=True):
        super(RelInSetL, self).__init__()
        
        if use_logit:
            self.logit_model = model
            
        self.use_logit = use_logit  
            
        
        self.softmax = torch.nn.Softmax(dim=1)
        

    def forward(self, x:torch.Tensor, mask:torch.Tensor, constr:torch.Tensor, training:bool=False):
       
        # batch dim is the number of grounding to need to squeeze to done it properly
        assert mask.size()[1]==constr.size()[1], 'invalid dimension between constraint and relative mask'
        
        # take logit from the model
        if self.use_logit==True:
            logits = self.logit_model(x, training)
        else:
            logits = x
            
        # remove background 
        probs = self.softmax(logits)
        
        # prune only logit related to mask
        out =torch.stack( [torch.sum(probs[i]*constr[i][mask[i]]) for i in range(mask.size()[0])] )
      
        assert out.size(0) == x.size(0), 'input tensor dimension is wrong'

        
        # if torch.sum( (out >1).all() )>0:
        #     victim_idx = out[(out >1).all()]
        #     print(f'victim is {victim_idx}')
        #     print(f'victim dtype: {victim_idx.dtype}')

        # hack to handle float precision that could lead to 1.00000000000000000000000001
        out = torch.clamp(out, min=0.0, max=1.0)
        assert (out >=0).all() and (out <=1).all(), f'output evaluation from not fuzzy for num of constr {out.size(0)} of \
                                                     out:\n{out}\nfrom INPUTS:\nmask:{mask}\nconstr={constr}\nlogits={logits}\nprobs={probs}\n'
                                                
        return out
    
    
class RelIsAL(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    
    def __init__(self, model:Optional[torch.nn.Module]=None, use_logit:bool=True):
        super(RelIsAL, self).__init__()

        if use_logit:
            self.logit_model = model
            
        self.use_logit = use_logit  
            
        
        self.softmax = torch.nn.Softmax(dim=1)
        

    def forward(self, x:torch.Tensor, label:torch.Tensor,training:bool=False):
       
        
        # take logit from the model
        if self.use_logit==True:
            logits = self.logit_model(x, training)
        else:
            logits = x
      
        probs = self.softmax(logits)
        
        # prune only logit related to mask
        out = torch.sum(probs*label, dim = 1)
      
        assert out.size(0) == x.size(0), 'input tensor dimension is wrong'
        #assert (out >=0).all() and (out <=1).all(), f'output evaluation from not fuzzy for num of constr {out.size(0)} of out:\n{out}\n'
        return out

class EntInSet(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    
    def __init__(self):
        super(EntInSet, self).__init__()

  
        self.softmax = torch.nn.Softmax(dim=1)
        

    def forward(self, x:torch.Tensor, mask:torch.Tensor, constr:torch.Tensor):
       
        # batch dim is the number of grounding to need to squeeze to done it properly
        assert mask.size()[1]==constr.size()[1], 'invalid dimension between constraint and relative mask'
        # take logit from the model
        
        probs = self.softmax(x)
        
        # prune only logit related to mask
        out = torch.stack( [torch.sum(probs[i]*constr[i][mask[i]]) for i in range(mask.size()[0])] )
      
        assert out.size(0) == x.size(0), 'input tensor dimension is wrong'
        out = torch.clamp(out, min=0.0, max=1)
        assert (out >=0).all() and (out <=1).all(), f'output evaluation from not fuzzy for num of constr {out.size(0)} of \
                                                     out:\n{out}\nfrom INPUTS:\nmask:{mask}\nconstr={constr}\nlogits={x}\nprobs={probs}\n'
        return out
    


class PredFromEmbedding(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """
    
    def __init__(self):
        super(PredFromEmbedding, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        

    def forward(self, logits:torch.Tensor, label: torch.Tensor):
       

        probs = self.softmax(logits)

        out = torch.sum(probs*label, dim = 1)
      
        assert out.size()[0] == logits.size()[0], 'input tensor dimension is wrong'
        return out


class GroundVanilla(torch.nn.Module):
    """
    This model returns a simple grounding (fuzzyfication)
    """
    def __init__(self, layer_sizes:Tuple[int]=(40, 64, 35), p_dropout:float=0.2):
        super(GroundVanilla, self).__init__()
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p_dropout)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])
      
    def forward(self, x:torch.Tensor, training:bool=False):
        """
        Method which defines the forward phase of the neural network for our multi class classification task.
        In particular, it returns the logits for the classes given an input example.

        :param x: the features of the example
        :param training: whether the network is in training mode (dropout applied) or validation mode (dropout not applied)
        :return: logits for example x
        """
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
            if training:
                x = self.dropout(x)
                
        logits = self.linear_layers[-1](x)
        
        return logits
       
  

def tensor_mask_rel(cls_sub:torch.Tensor, cls_obj:torch.Tensor, cls_rel:torch.Tensor, rel_dict: Dict[Tuple[int,int], torch.Tensor ], device:torch.device,
                    list_label:Optional[torch.device],
                    ) -> torch.Tensor: 
    """
    generate the mask of the relationship
    """  
    # remove to debug
    # assert type_constr<3, f'{type_constr} not supported: it must be between 0 and  2'
    mask_list =  []
    
    # 0 negative, 1 positive
    cls_constr = cls_rel # if type_constr == 1 else cls_rel
 
    # create mask for constraints in cls_constr tensor
    
    for i in range(cls_constr.size(0)):
        tmp = []
        for k,v  in rel_dict.items(): 
            tmp.extend( [False]* v.shape[0] if (cls_sub[i].argmax().item(),cls_obj[i].argmax().item()) !=k else [True]*v.shape[0])
                                
        mask_list.append(tmp)
        
    assert len(mask_list[0]) == len(mask_list[-1]), f'all element inside the mask tensor 0 must have the same size'
        
    mask = torch.tensor(mask_list, device=device  )
    assert mask.size(1)==list_label.size(0), f'mask_idx shape {mask.size(0)} not equal to label shape{list_label.size(0)}'

      
    return  mask

def dict_tensor_mask_rel( cls_rel:torch.Tensor, rel_dict: Dict[Tuple[int,int], torch.Tensor ], device:torch.device,
                    list_label:Optional[Dict[int,torch.device]], indices:Tuple[int]=(1,2)
                    ) -> Dict[int,torch.Tensor]:   
  
    """
    generate the tensor mask of relationship in  a compact way using dictionary
    """
    # 0 negative, 1 positive
    cls_constr = cls_rel # if type_constr == 1 else cls_rel
 
    # create mask for constraints in cls_constr tensor
    mask_list = {0: [], 1:[]}
    for i in range(cls_constr.size(0)):
        tmp0 = []
        tmp1 = []
        for k,v  in rel_dict.items(): 
            tmp0.extend( [False]* v[0].shape[0] if cls_constr[i].argmax().item() !=k else [True]*v[0].shape[0])
            tmp1.extend( [False]* v[1].shape[0] if cls_constr[i].argmax().item() !=k else [True]*v[1].shape[0])
                                
        mask_list[0].append(tmp0)        
        mask_list[1].append(tmp1)        
            
    # sanity check
    assert len(mask_list[0][0]) == len(mask_list[0][-1]), f'all element inside the mask tensor 0 must have the same size'
    assert len(mask_list[1][0]) == len(mask_list[1][-1]), f'all element inside the mask tensor 1 must have the same size'

    mask = [torch.tensor(mask_list[0], device=device  ), torch.tensor(mask_list[1], device=device  )]
    assert mask[0].size(1)==list_label[indices[0]].size(0), f'mask_idx shape {mask[0].size(1)} not equal to label shape{list_label[indices[0]].size(0)}'
    assert mask[1].size(1)==list_label[indices[1]].size(0), f'mask_idx shape {mask[1].size(1)} not equal to label shape{list_label[indices[1]].size(0)}'

    return  mask




def grounding(sub_bbox:torch.Tensor,  obj_bbox:torch.Tensor, sub_logits:torch.Tensor, obj_logits:torch.Tensor, 
                  ltn_strategy:int,
                  rel_logits:Optional[torch.Tensor]=None,
                  sub_cls:Optional[torch.Tensor]=None, obj_cls:Optional[torch.Tensor]=None, rel_cls:Optional[torch.Tensor]=None,
                  rel_const_map:Optional[Dict[Tuple[int,int], torch.Tensor]]=None,
                  device:Optional[torch.device]=None,
                  list_label:Optional[Dict[int, torch.device]]=None,
                  indices:Optional[Tuple[int]]=None,
                  use_regressor:bool=False,
                  constr_type:int=0,
                  ) ->Tuple[ltn.Variable,ltn.Variable,ltn.Variable,
                            # Optional[Tuple[ltn.Variable,ltn.Variable,ltn.Variable]],
                            Optional[torch.Tensor],
                            Optional[torch.Tensor]
                            ]:
        """
        given the strategy it create a regular grounding for subject, object and relationship
        """

        if ltn_strategy==0:
            # remove background cls since already processed before invalid bbox (eos)
            tensor_sub_bbox = sub_logits[:,:-1]
            tensor_obj_bbox = obj_logits[:,:-1]
        elif ltn_strategy==1 or ltn_strategy==2:
            tensor_sub_bbox = torch.cat( (sub_logits, sub_bbox) , dim=-1 )
            tensor_obj_bbox = torch.cat( (obj_logits, obj_bbox) , dim=-1 )
        else:
            raise ValueError('Invalid strategy')
                
        gr_sub_bbox = ltn.Variable(f'gr_sub_box', tensor_sub_bbox    )
        gr_obj_bbox = ltn.Variable(f'gr_obj_box', tensor_obj_bbox    )
                    
        gr_rel_bbox = None
        mask_idx = None
        constr_item = None
        # mask_usable_rel = None
        if rel_logits is not None:
            
            #  check whether constraints can be applied or not
            # pruning (cls_s, cls_o)
            assert rel_const_map is not None, 'constraints map cannot be None'
            assert device is not None, 'device cannot be None'
            
            # mask_usable_rel = tensor_mask_rel(sub_cls, obj_cls, rel_const_map, device)
            if constr_type == 0:
                mask_idx = tensor_mask_rel(sub_cls, obj_cls, rel_cls, rel_const_map, device, list_label[0])
            
            else:
                mask_idx = dict_tensor_mask_rel(rel_cls, rel_const_map, device, list_label, indices)   
    

            if use_regressor == True:
                ir_subobj = box_ops.box_ir(sub_bbox, obj_bbox)
                ir_objsub = box_ops.box_ir(obj_bbox, sub_bbox)
                area_sub =  box_ops.box_area( sub_bbox).unsqueeze(dim=1)
                area_obj =  box_ops.box_area( obj_bbox).unsqueeze(dim=1)

                sub_centr = box_ops.box_centroid(sub_bbox)
                obj_centr = box_ops.box_centroid(obj_bbox)
                cos_bbox =  torch.norm(sub_centr* obj_centr, dim=1 )/( torch.norm(sub_centr, dim = 1)*torch.norm(obj_centr, dim = 1) )
                
                # assert cos_bbox.size()[0] == sub_logits[mask_usable_rel].size()[0], 'batch dimension should be respected during grounding!'
                sin_bbox = torch.sqrt(1- cos_bbox.pow(2) )
                eucl_dist = torch.norm(sub_centr - obj_centr, dim=1 ).unsqueeze(dim=1)
            
                tensor_gr_rel = torch.hstack( (tensor_sub_bbox, tensor_obj_bbox,
                    ir_subobj, ir_objsub,
                    area_sub / area_obj,   area_obj /  area_sub,
                    eucl_dist,sin_bbox.unsqueeze(1), cos_bbox.unsqueeze(1),
                    rel_logits,
                    
                    )
                )
            
            else:
                tensor_gr_rel = rel_logits[:,:-1]

            if constr_type==0:
                constr_item =list_label[0].unsqueeze(0).repeat(mask_idx.size(0), 1,1)           # sanity check to avoid dumb error
                assert torch.sum(constr_item[0] - constr_item[-1]).item() == 0, 'tensor is not  correctly repeated'
                

            else:
                

                constr_item = [ list_label[indices[0]].unsqueeze(0).repeat(mask_idx[0].size(0), 1,1),
                            list_label[indices[1]].unsqueeze(0).repeat(mask_idx[1].size(0), 1,1) ]
        
                    
                assert torch.sum(constr_item[0][0] - constr_item[0][-1]).item() == 0, 'tensor-0 is not  correctly repeated'
                assert torch.sum(constr_item[1][0] - constr_item[1][-1]).item() == 0, 'tensor-1 is not  correctly repeated'
                
            

            gr_rel_bbox = ltn.Variable(f'gr_rel_box', tensor_gr_rel )
            
   
        
        return gr_sub_bbox, gr_obj_bbox,  gr_rel_bbox ,mask_idx, constr_item
        
    