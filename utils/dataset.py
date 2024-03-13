# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved

from collections import Counter
import os.path as osp
import os, h5py, json, shutil
from random import randint
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import Any, Optional, Tuple, Dict, Set
import cv2
import numpy as np
from tqdm import tqdm



AV_CLASS ={
          11 :'bike',
          23: 'bus',
          26: 'car',
          80: 'motorcycle',
          114: 'sidewalk',
          124: 'street',
          135: 'train',
          137: 'truck',
          141: 'vehicle'
}

MAP_CLASS ={
          11:   ('bike', 34),
          22:    ('building', 12),
          23:    ('bus', 29),
          26:    ('car', 27),
          29:    ('child', 25),
          45:    ('fence', 14),
          53:    ('girl', 25),
          56:    ('guy', 25),
          68:    ('kid', 25),
          71:    ('lady', 25),
          78:    ('man',  25),
          79:    ('men',  25),
          80:    ('motorcycle', 33),
          90:    ('person', 25),
          91:    ('people', 25),
          99:    ('pole', 18),
          114:   ('sidewalk',  9),
          115:   ('sign', 21),
          124:   ('street', 8),
          135:   ('train', 32),
          137:   ('truck', 28),
          142:   ('vehicle', 6),
          149:   ('woman', 25),
          
          # static object according to cityspace
          28:    ('cat', 5),
          33:    ('cow', 5),
          37:    ('dog', 5),
          104:   ('rock', 5),
          109:   ('sheep', 5),
          136:   ('tree', 5),
          141:   ('vegetable', 5),
          148:   ('wire', 5)
        }
    
MAP_REL={
        1:  ('above', 17),
        3:  ('against', 10),
        4:  ('along', 7),
        6:  ('at', 2),
        7:  ('attached to', 24),
        8:  ('behind', 13),
        9:  ('belonging to', 1),
        10: ('between', 16),
        11: ('carrying', 34),
        12: ('covered in', 6),
        18: ('growing on', 22),
        19: ('hanging from', 25),
        20: ('has', 5), 
        21: ('holding', 28),
        22: ('in', 3),
        23: ('in front of', 12),
        25: ('looking at', 27),
        26: ('lying on', 26),
        28: ('mounted on', 36),
        29: ('near', 18),
        31: ('on', 3),
        32: ('on back of', 14),
        33: ('over', 11),
        35: ('parked on', 21),
        36: ('part of', 9),
        38: ('riding', 29),
        40: ('sitting on', 33),
        41: ('standing on', 23),
        43: ('under', 15),
        44: ('using', 35),
        45: ('walking on', 30),
        47: ('watching', 31),
        50: ('with', 8)
        
}

MAP_PD2TG= {
    6: '<http://pandaset.org/Dynamic>',
    9: '<http://pandaset.org/Road>',
    5: '<http://pandaset.org/Static>',
    14: '<http://pandaset.org/Fence>',
    25: '<http://pandaset.org/Person>',
    27: '<http://pandaset.org/Car>',
    29: '<http://pandaset.org/Bus>',
    30: '<http://pandaset.org/Caravan>',
    33: '<http://pandaset.org/Motorcycle>',
    34: '<http://pandaset.org/Bicycle>',
}


class KGDataset(Dataset):
    """Dataset  class for nuscene knowledge graph .
 
    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        reduce_to (int): A flag to reduce the number of subdivision to evaluate
            (default: :obj:`0`)
        subdivision (int): A flag to indicate the number of subdivision of dataset
             (default: :obj:`292`)
    """          
    def __init__(self, root: str, reduce_to:int=0, subdivision:int=59, block_size:int=10000, dataset_name:str='pandaset_kg'):
       self.subdivision = subdivision  if reduce_to==0 else reduce_to
       self.block_size = block_size
       self.dataset_name = dataset_name
    
       
       super().__init__(root)
        
        
    @property
    def raw_file_names(self):
        # fine save in raw_path ( ) which is dataset_root/raw
        return [ f'{self.dataset_name}_{i}.nt' for i in range(self.subdivision)]


    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in range(self.subdivision-1)]

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     return ['prior_kg/prior_kg.nt']

    def iterate_rdf(self, raw_path) ->pd.DataFrame:


        return pd.read_csv(raw_path,
                           sep=' ',
                           header=None,
                           names=['subj', 'pred', 'obj'],
                           dtype=str,
                           # 0 -> subject, 1-> predicate , 2-> object
                           usecols=[0, 1, 2],
                           lineterminator='\n',
                           # chunksize=1000,
                           )
    def __mapping_triple(self) -> None:
        # retrieve the one-shot enconding on all possible relation inside the dataset
        objects = set()
        subjects = set()
        relations = set()
        
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            df = self.iterate_rdf(raw_path)

            
            freq = df['pred'].value_counts()

            rel_i = sorted(set(df['pred'].values), key=lambda p: -freq.get(p, 0))
            subj_i = set(df['subj'].values)
            
            # one-shot enconding only for not numeric value
            obj_i = set(filter(lambda x: not str(x).endswith('<http://www.w3.org/2001/XMLSchema#double>'),   df['obj'].values) )

            # index mapping
            objects.update(obj_i)
            subjects.update(subj_i)
            relations.update(rel_i)
            
     
        
        nodes = set(subjects.union(objects))    
        self.rel_dict = {rel: i for i, rel in enumerate(relations)}
        # subj_dict = {rel: i for i, rel in enumerate(subjects)}
        # obj_dict = {rel: i for i, rel in enumerate(objects)}
        self.node_dict = {node: i for i, node in enumerate(nodes)}
        
        # uncomment only for debug
        # print(self.node_dict)
        
        with open('map2emb.txt', 'w') as fp:
            fp.writelines([ f'{n}\n' for n in nodes] )
            
        tot_node = len(self.node_dict)
        self.pos ={ 'X': tot_node+1, 
                    'Y': tot_node+2, 
                     'Z': tot_node+3 } 
        
        self.rot ={ 'X': tot_node+4, 
                    'Y': tot_node+5, 
                     'Z': tot_node+6 }
        self.yaw = tot_node+7
        
    
    def process(self):
        
        data_list = []

     
        self.__mapping_triple()
        # save to pt format
        print(f'Creation of dataset in pt format')
        for  raw_path in tqdm(self.raw_paths):
            
            # Read data from `raw_path`.
            df = self.iterate_rdf(raw_path)

           
            # loop over item for edge generation
            # edge = number of edges of df
            # num_edge = df.shape[0]
            #num_edge = self.block_size
            num_edge = len(df.index)
            # edge_index = torch.empty((2, num_edge), dtype=torch.long)

            # edge_type = torch.empty(num_edge, dtype=torch.long)
            # num_rel  = len(rel_dict)
            
            # index of the adjacency matrix
            edge_index = torch.empty((2, num_edge), dtype=torch.long)
            
            # tensor of relationship value, which depends on block size
            edge_type = torch.empty(num_edge, dtype=torch.long)
            i = 0
            # j = 0
    
        
            for row in df.itertuples():

                edge_index[0, i] = self.node_dict[row.subj]
                
                
                
                # handle numeric object in a relationship
                if row.obj.endswith('<http://www.w3.org/2001/XMLSchema#double>'):
                    # tmp = str(row.obj).removesuffix('^^<http://www.w3.org/2001/XMLSchema#double>')
                    # val = float(tmp)
                    
                    # edge_index[1, i] = val
                    
                    if row.pred.endswith('RotationZ'):
                        edge_index[1, i] =  self.yaw
                        
                    elif row.pred.endswith('PosX'):
                        edge_index[1, i] =  self.pos['X']
                    elif row.pred.endswith('PosY'):
                        edge_index[1, i] =  self.pos['Y']
                    elif row.pred.endswith('PosZ'):
                        edge_index[1, i] =  self.pos['Z']
                        
                    elif row.pred.endswith('TraslationX'):
                        edge_index[1, i] =  self.rot['X']
                    elif row.pred.endswith('TraslationY'):
                        edge_index[1, i] =  self.rot['Y']
                    else: # row.pred.endswith('TraslationZ')
                        edge_index[1, i] =  self.rot['Z']
                        
                    
                    
                else:
                    edge_index[1, i] =  self.node_dict[row.obj]
                    
                edge_type[i] = self.rel_dict[row.pred]

       
                i+=1
  
            
            data = Data(edge_index=edge_index, edge_type=edge_type, 
                        num_edge_types=2*i,num_nodes=i
                        )
            
            data_list.append(data)
            

        for data, path in zip(data_list, self.processed_paths):
          
            torch.save(data, path)
            #idx += 1

        print("raw dataset loaded")


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data



class H5ToCoco():
    """ TrafficGenome dataset ht5 format into json """
    
    def __init__(self, h5_file:str, root_dir:str, img_dir:str, json_file:str=None, bbox_dim:int=1024,
                 limit_db:bool=False, show_img:bool=False, is_tg:bool = True, v2:bool=False):
        """_summary_

        Args:
            h5_file (str): file containg traffic dataset in VG150 format
            root_dir (str): directory used to store the dataset
            img_dir (str): directory where the images are saved
            json_file (str): file containg Traffic Genome metadata in VG150 format
            bbox_dim (int, optional): use either the bbox for  2048x1024 (default) resolution or 1024x512.
            limit_db (bool, optional): reduce the dimension of the dataset (Default false) for debug purposes
            show_imb (bool, optional): to show images  with bounding box (default false)
            v2(bool): do not account for background class during labeling process
        """
        
        self.h5_file = h5py.File(h5_file, 'r')
        
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.show_img = show_img

        # recall that h5 get return numpy array 
        self.pred = self.h5_file.get('predicates')
        self.rel_bb = self.h5_file.get('relationships')
        self.bbox_name = self.h5_file.get('labels')

        self.fst_bbox = self.h5_file.get('img_to_first_box')
        self.last_bbox =  self.h5_file.get('img_to_last_box')
        self.fst_rel =self.h5_file.get('img_to_first_rel')
        self.last_rel =  self.h5_file.get('img_to_last_rel')
    
        self.is_tg = is_tg
 
    
        split = np.array(self.h5_file.get('split'))
        
   
        if not limit_db:
            self.num_img = int(split.shape[0])
        
            self.img_test = np.where(split ==2 )[0]
            img_trainval =  np.where(split == 0 )[0]
            split_val = int(img_trainval.shape[0]*0.10)
        
        else:
            self.num_img = 50
            self.img_test = np.asarray(range(0,10))
            img_trainval =  np.asarray(range(10,50))
            split_val = int(img_trainval.shape[0]*0.10)
    
        self.img_train, self.img_val = img_trainval[:-split_val], img_trainval[-split_val:]
   
       
        self.bbox_xyxy = np.array(self.h5_file.get('boxes_1024') if bbox_dim==1024 else self.h5_file.get('boxes_512') )

        self.box_scale = bbox_dim
        self.v2 = v2
        
        if self.is_tg:
            with open(json_file) as f: 
                tmp= json.load(f)
                self.bbox_legend = tmp['idx_to_label']
                if self.v2:
                    self.pred_name = [str(v) for v  in tmp['idx_to_predicate'].values()][1:]
                    self.bbox_legend.pop('0')
                       
                else:
                    self.pred_name = [str(v) for v  in tmp['idx_to_predicate'].values()]
                    

              
        else:
            # Visual genome
     
            self.pred_name = ["__background__", "above", "across", "against", "along", "and",
                "at", "attached to", "behind", "belonging to", "between", "carrying", 
                "covered in", "covering", "eating", "flying in", "for", "from", "growing on", 
                "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at",
                "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", 
                "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on",
                "standing on", "to", "under", "using", "walking in", "walking on", "watching", 
                "wearing", "wears", "with"]
        
            bbox_label = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
        

            self.bbox_legend = { str(i): v for i, v in enumerate(bbox_label) }

        # in case runned multiple it starts again without data folder
        try:
            shutil.rmtree(root_dir)
            os.mkdir(root_dir)
        except:
            os.mkdir(root_dir)
            
            
        self.corrupted_img = set()
        self.false_bbox = set()
        self.corrupted_bbox = {}
        self.bbox_id_scaled = {}
      
            
        
    def create_rel(self):
        """generates the relationships
        """
        
        data_rel = { 'train':{}, 'val':{}, 'test':{}}
     
        print(f'Generating relationship labels')
        for img_id in tqdm(range(self.num_img)):
            
            if img_id  < self.num_img and img_id not in self.corrupted_img:
                
                # check there are relationship in the image, since vg has img without rel
                if self.fst_rel[img_id] >=0 and self.last_rel[img_id]>=0:
                    for i_rel in range( self.fst_rel[img_id],self.last_rel[img_id]+1  ):
            
                        offset = -1 if self.v2 else 0
                        pred_i = int(self.pred[i_rel] + offset)
                        assert pred_i>=0, 'pred cannot be negative'
                        key_id = str(img_id) 
                        # rescale idx wrt to img
                        bbox1_id = self.rel_bb[i_rel, 0]
                        bbox2_id = self.rel_bb[i_rel, 1]
                        
                        # avoid saving false bbox that correspond to points
                        if bbox1_id not in  self.corrupted_bbox[img_id] and bbox2_id not in self.corrupted_bbox[img_id]:
                            
               
                            bbox1_id_scaled = int( self.bbox_id_scaled[img_id][self.rel_bb[i_rel, 0]])
                            
                            bbox2_id_scaled = int(self.bbox_id_scaled[img_id][self.rel_bb[i_rel, 1]])
                        
                    
                        
                        
                            if  img_id in self.img_val:
                                selector = 'val'
                            elif img_id in self.img_test:
                                selector = 'test'
                            else:
                                selector = 'train'

                            
                            if key_id not in data_rel[selector].keys():
                                data_rel[selector][key_id] = []
                            
                            # add relationship in format [ first bouding box,  second bounding box,  predicate ]
                            data_rel[selector][str(img_id)].append( [bbox1_id_scaled, bbox2_id_scaled,  pred_i ])
                        else:
                            print('Wrong bbox pruned')
            
        data_rel['rel_categories']=self.pred_name
        
        self.__write_json(data_rel, 'rel.json')
             
             
    def create_bbox(self):
        """creates a bounding box
        """
        
        data_train = {'images': [], 'annotations': [], 'categories': []}
        data_val = {'images': [], 'annotations': [], 'categories': []}
        data_test = {'images': [], 'annotations': [], 'categories': []}
        
        print(f'Generating bounding box labels')
        for file_name in tqdm(list(sorted(os.listdir(self.img_dir), key=lambda s: int(s[:s.index('_')])))):
   
            if file_name.endswith('.png') or file_name.endswith('.jpg'):
                # remove the '_' character to have only the number of the file
                
                if self.is_tg:
                    img_id = int(file_name[:file_name.index('_')])
                    
                else:
                    img_id =  int(file_name.removesuffix('.jpg'))
                    
                img_path =  os.path.join(self.img_dir, file_name)
                
                img =cv2.imread( img_path )
                
                # ugly trick to handle corrupted images in visual genome
                try:
                    height, width = img.shape[:2]
                    # sanity check to be sure to avoid imgg with bbox
                    if img_id< self.num_img and  self.fst_bbox[img_id] >=0 and self.last_bbox[img_id]>=0:
                      
                        self.corrupted_bbox[img_id] =set()
                        item = self.__create_coco_item(file_name, img_id, height, width, img)
            
                        if  img_id in self.img_val:
                            data_val['images'].extend(item['images'])
                            data_val['annotations'].extend(item['annotations'])
                            
                                        # check if categories id not already present
                            cc_id =[ cc['id'] for cc in data_val['categories']  ]
                            item_filtered= filter(lambda ii: ii['id'] not in cc_id  ,item['categories'])
                            for ii in  item_filtered:
                                data_val['categories'].append(ii)
                            
                        elif img_id in self.img_test:
                            data_test['images'].extend(item['images'])
                            data_test['annotations'].extend(item['annotations'])
                        
                            
                            cc_id =[ cc['id'] for cc in  data_test['categories']  ]
                            item_filtered= filter(lambda ii: ii['id'] not in cc_id  ,item['categories'])
                            for ii in  item_filtered:
                                data_test['categories'].append(ii)
                            
                        else:
                            data_train['images'].extend(item['images'])
                            data_train['annotations'].extend(item['annotations'])
            
    
                            cc_id =[ cc['id'] for cc in  data_train['categories']  ]
                            item_filtered= filter(lambda ii: ii['id'] not in cc_id  ,item['categories'])
                            for ii in  item_filtered:
                                data_train['categories'].append(ii)
                
                except AttributeError:
                    self.corrupted_img.add(img_id)
                    
        print(f'number of corrupted img is = {len(self.corrupted_img)}')
        self.__write_json(data_val, 'val.json')
        self.__write_json(data_test, 'test.json')
        self.__write_json(data_train, 'train.json')
             

    def __write_json(self,ddict: Dict, filename:str):
        """generates the json file 

        Args:
            ddict (Dict): dictionary to write
            filename (str): file name
        """
        with open(self.root_dir + '/' + filename, 'w') as f:
             json.dump(ddict, f)
           
             

    def __create_coco_item(self,  path:str, img_id:int, height:int, width:int, img:np.array) ->Dict[str, Any]:
        """creates a dataset item

        Args:
            path (str): path of the item
            img_id (int): image id
            height (int): height of the image
            width (int): width of the image
            img (np.array): image array

        Returns:
            Dict[str, Any]: the item processed
        """
         
        item = {'images': [], 'annotations': [], 'categories': []}
        images_dict ={'file_name':path, 'height': int(height), 'width': int(width), 'id':img_id }
        corrupted_offset = 0
        
        # debug
        # if img_id == 313:
        #     print()

        for bbox_id in range(self.fst_bbox[img_id],self.last_bbox[img_id]+1):
            # for our purpose iscrows is irrelevant (it refers to segmentation task), thus it is setted to 0
            # recover original box from BOX_SCALE
            cur_bbox = self.bbox_xyxy[bbox_id] #  self.box_scale * max(width, height)


            # bbox in traffic genome is xyxy
            xl = int(cur_bbox[0])
            yt = int(cur_bbox[1])
            xr = int(cur_bbox[2])
            yb = int(cur_bbox[3])
            
            w = int(cur_bbox[2]- cur_bbox[0])
            h = int(cur_bbox[3]- cur_bbox[1])
            

            
            area = float(h*w)

            
            # Black color in BGR
            if self.show_img==True:
                color = (randint(0,255),randint(0,255),randint(0,255) )
                thickness = 1
                img = cv2.rectangle(img, (xl, yt), (xr, yb), color, thickness)
                
                
                cv2.imshow('bbox',img)
                cv2.waitKey(5)
                cv2.destroyAllWindows()
            

            if area>0:
                offset = -1 if self.v2 else 0
                cat_id = int(self.bbox_name[bbox_id]) + offset
                assert cat_id>=0, 'entity label cannot be negative'
                # remove scale offset
                bbox_id_scaled = int(bbox_id - self.fst_bbox[img_id] -corrupted_offset)
                
                if img_id not in self.bbox_id_scaled.keys():
                    self.bbox_id_scaled[img_id] = {}
                    
                self.bbox_id_scaled[img_id][bbox_id] =  bbox_id_scaled
      
                # coco dataset needs unique id for every bbox in the entire dataset
                ann_dict = {'segmentation': None, 'area':  area, 'bbox': [xl, yt, xr, yb], 
                            'iscrowd': 0, 'image_id': img_id, 
                            'id':  int(bbox_id), 'category_id': cat_id
                        }

                cat_name = str(self.bbox_legend[str(cat_id)])
                cat_dict = {'supercategory': cat_name, 'id': cat_id, 'name': cat_name }
                        
            
                item['annotations'].append( ann_dict)
                
                # check if categories id not already present
                list_cat_id = [ k['id']  for k in item['categories'] ]
                if cat_dict['id'] not in list_cat_id:
                    item['categories'].append(cat_dict)
            else:
                # needed to avoid crash during training 
                corrupted_offset += 1
  
                    
                self.corrupted_bbox[img_id].add(bbox_id)
                # if img_id == 313:
                #     print(f'invalid bbox found! in img 313 {bbox_id}')
                # self.false_bbox.add(bbox_id)
        
        item['images'].append(images_dict)
        
        
        assert len(item['images'])!=0, 'image must exist!'
        assert len(item['annotations'])!=0, 'image must have bounding box!'
        assert len(item['categories'])!=0, 'category cannot te null!'
        return item
    
    def create_all(self):
        """creates the dataset in one-shot
        """
        self.create_bbox()
        self.create_rel()
        
        # print how many invalid bbox are found
        print(f'found {self.corrupted_bbox} corrupted bbox')
   
    def stats(self):
        """generate the statistics of the dataset
        """
        bbox4img = np.asarray([  self.last_bbox[i]- self.fst_bbox[i] +1 for i in range(self.num_img)], dtype=int)
        rel4img = np.asarray([  self.last_rel[i]- self.fst_rel[i] +1 for i in range(self.num_img)], dtype=int)
        
        print("----------------STATS---------------------")
        print(bbox4img)
        
        for i, n in enumerate(bbox4img):
            if n==131:
                print(f'index for 130 is i={i}')
        print(f'bounding box for img: max({max(bbox4img)}) min({min(bbox4img)}) mean({np.mean(bbox4img)})')
        print(f'relationship for img: max({max(rel4img)}) min({min(rel4img)}) mean({np.mean(rel4img)})')
     
        
def stats_vg(ann_bbox_file: str, ann_rel_file:str, set_choice:str):
    """generates the statistics of compatible label from visual genome dataset

    Args:
        ann_bbox_file (str): file of boundong box labels
        ann_rel_file (str): file of relationship labels
        set_choice (str): set of choice
    """
    with open(ann_rel_file) as f: 
        tmp= json.load(f)
        
        rel_set_img = tmp[set_choice]
    with open(ann_bbox_file) as f: 
        tmp= json.load(f)
        
     
        image_ids = set()
        offset_bb = {}
        
    
        
        for ann in  tqdm(tmp['annotations']):
    
            if ann['image_id'] not in  offset_bb.keys() :
                offset_bb[ann['image_id']] = ann['id'] 
                
            elif  offset_bb[ann['image_id']]< ann['id'] :
                offset_bb[ann['image_id']] = ann['id']  
                
                      
        for ann in  tqdm(tmp['annotations']):
            
            
            if  ann['category_id'] in AV_CLASS.keys():
                
                bb_id = offset_bb[ann['image_id']] - ann['id'] 
                count = sum([ 1 for _ in filter( lambda rel_bb: ( (rel_bb[0]==bb_id and rel_bb[1] in MAP_CLASS.keys())  or (rel_bb[1]==bb_id and rel_bb[0] in MAP_CLASS.keys()) )and rel_bb[2] in MAP_REL.keys(), rel_set_img[str(ann['image_id'])])])
                
                if count>0:
                  image_ids.add(ann['image_id'])
                
        
        print(f'number of image that have the class label is {len(image_ids)}')
                  

                  
def add_vg2tg(bbox_offset:int, src_bbox_file: str, src_rel_file:str, set_choice:str,
              dst_train_file:str, dst_rel_file:str,  valtest_set:bool=False,
              dst_val_file:Optional[str]=None, dst_test_file:Optional[str]=None,
              offset_id_img:bool=0,v2:bool=False) -> Tuple[int,Set[str], int]:
    """extends  traffic genome with compatible visual genome images

    Args:
        bbox_offset (int): bounding box offset.
        src_bbox_file (str): source bounding box file path
        src_rel_file (str) : source relationship file path
        set_choice (str) : set to be processed
        dst_train_file (str): new bounding box file path
        dst_rel_file (str): new relatiomship file path
        valtest_set (bool) is a validation set.  Default (False)
        dst_val_file (Optional[str]): path of new  label val-set file. Defaults None
        dst_test_file (Optional[str]): path of nww label test-set file  Defaults None.
        offset_id_img (bool) image offset.  Default 0
        v2 (bool) : version of dataset to be generated. Defaults False

    Returns:
        (int,Set[str], int): bbox_offset, n_img, filename_s, n_rel
    """
    filename_s= set()
    image_ids = set()
    offset_bb = {}
    data_bbox = {}
    img_metadata = {}
    n_rel = 0

    #358 are value inside we decide to use around 9% for
    split_valtest = 36 # 358 a
    iter_valtest = 0

    invalid_bbox = {}
    with open(src_bbox_file, 'r') as f: 
        vg_ann_bbox = json.load(f) 
    
    with open(src_rel_file, 'r') as f: 
        vg_ann_rel= json.load(f)
        rel_set_img =  vg_ann_rel[set_choice]
           
   
   
    if valtest_set:
        with open(dst_val_file, 'r') as f: 
            dst_val_bbox = json.load(f)
            
        with open(dst_test_file, 'r') as f:     
            dst_test_bbox = json.load(f)

    with open(dst_train_file, 'r') as f: 
            dst_train_bbox = json.load(f)
    
    with open(dst_rel_file, 'r') as f: 
        dst_ann_rel= json.load(f)
        
    n_img = offset_id_img
      
   
    print(f'number of image involved in relationship before extension:{n_img}')
        

    
    print('preprocessing')
    # n_img = 997
    # get the maximum index
    for ann in  tqdm(vg_ann_bbox['annotations']):
    
        if ann['image_id'] not in  offset_bb.keys() :
            offset_bb[ann['image_id']] = ann['id'] 
            
        elif  offset_bb[ann['image_id']]< ann['id'] :
            offset_bb[ann['image_id']] = ann['id']    
    

    
    img_info = { x['id']: x for x in  vg_ann_bbox['images']}
  
    n_invalid = 0
        
        
        
    # for easiness sort ann by bbox id
    vg_ann = vg_ann_bbox['annotations']
    vg_ann_sorted = sorted( vg_ann , key = lambda x: x['id'])
    for ann in  tqdm(vg_ann_sorted ):
        img_id = ann['image_id']
        
        
       
        
        if img_id not in image_ids:
            
            # prune only useful relationship cuz not interested in img with AV_CLASS obj without relationship
            rel_bbox = list(filter( lambda rel_bb:  rel_bb[2] in MAP_REL.keys(), rel_set_img[str(ann['image_id'])]) )
            
            #  bbox offset, bbox id from annotations, num of valid relationship in TrafficGenome
            img_metadata[img_id] = [ 0, set(), rel_bbox, {} ]

        
        
        n_rel_img = len(img_metadata[img_id][2])
        # we assume that in order to be an image related to autonomous driving scenario
        # it must contain at least one object from AV_CLASS
        
        # abs used to protect against stupid error of reltr author
        w = abs(ann['bbox'][0] - ann['bbox'][2])
        h = abs(ann['bbox'][1] - ann['bbox'][3])
        
        if img_id not in invalid_bbox.keys():
            invalid_bbox[img_id] = set()
    
        # swap misleading attribute against stupid error of reltr author
        if  ann['bbox'][0]>ann['bbox'][2]:
            ann['bbox'][0],ann['bbox'][2] = ann['bbox'][2],ann['bbox'][0]
        
        if  ann['bbox'][1]>ann['bbox'][3]:
            ann['bbox'][1],ann['bbox'][3] = ann['bbox'][3],ann['bbox'][1]
        
        if  w == 0 or h == 0 or h>img_info[img_id]['height'] or w>img_info[img_id]['width'] or \
            ann['bbox'][3]>img_info[img_id]['height'] or ann['bbox'][2]>img_info[img_id]['width'] or \
            ann['bbox'][1]>img_info[img_id]['height'] or ann['bbox'][0]>img_info[img_id]['width']:  
            # bb_id = ann['id']    
            # bb_val = ann['bbox']    
            # print(f'invalid bbox encountered whose id is {bb_id}, {bb_val}')

            bb_id = offset_bb[img_id] - ann['id'] 
            
            invalid_bbox[img_id].add( bb_id)
            
            n_invalid+=1
        # check whether the category belongs to traffic genome ontology
        elif  ann['category_id'] in AV_CLASS.keys() and  n_rel_img>0:
            
    
  
            # calculate the offset to understand the idx inside relationship json file
            bb_id = offset_bb[img_id] - ann['id'] 
            
            # convert bbox idx notation to tg and save modified ann bbox
            # only if bbox relationship is valid     
                            
            image_ids.add(img_id)
            
            tmp_dict = ann
            # tmp_dict['id'] = img_metadata[img_id][0]
            tmp_dict['id'] = bbox_offset
            
  
            img_metadata[img_id][1].add(bb_id)
            img_metadata[img_id][3][bb_id] = img_metadata[img_id][0]
                
                
                
                
            if img_id not in data_bbox.keys():
                data_bbox[img_id] = { 'bbox': [], 'rel': []}
                
            # save annotation related to bbox  i.e.  filter invalid bbox
            data_bbox[img_id]['bbox'].append(tmp_dict)
        
            img_metadata[img_id][0] += 1 
            bbox_offset += 1
                  
            # check that the second term is valid
            
            #data_bbox[img_id]['rel'].extend(rel_pruned)
    print(f'found {n_invalid} invalid bbox')
    # filter and map relationship       
    for img_id in data_bbox.keys():
        set_bbox = img_metadata[img_id][1]
        rel_bbox = img_metadata[img_id][2]
        map_bbox = img_metadata[img_id][3]
        
        
        valid_rel = list(filter(lambda r: r[0] in set_bbox and r[1] in set_bbox and
                                     r[0] not in invalid_bbox[img_id] and r[1] not in invalid_bbox[img_id]
                                     , rel_bbox ))
        # sanity check 
        if len(invalid_bbox[img_id]):
            

            
            false_rel = filter(lambda r: r[0] in set_bbox and r[1] in set_bbox and
                                     r[0]  in invalid_bbox[img_id] and r[1]  in invalid_bbox[img_id]
                                     , rel_bbox )
            assert valid_rel != false_rel, 'Saved invalid relationship'
            
        if v2:
            rel_pruned = list(map(lambda r: [ map_bbox[r[0]], map_bbox[r[1]],  MAP_REL[r[2]][1] -1 ],  valid_rel ) )   
        
        else:
            rel_pruned = list(map(lambda r: [ map_bbox[r[0]], map_bbox[r[1]],  MAP_REL[r[2]][1] ],  valid_rel ) )
        
        # respect order and 
        
        
        if len(rel_pruned)>0:
            
            data_bbox[img_id]['rel'].extend(rel_pruned)
            
            # save the corresponding filename
            filename_s.add(str(img_id) + '.jpg')
        else:
            # no useful relationship, the image is remove from the dataset
            image_ids.remove(img_id)
        
        
     
    print('write relationship')



    for img_id in tqdm(image_ids):
        cand_info = img_info[img_id]

   
        # change image id
        cand_info['id'] = n_img

        rel_bbox = data_bbox[img_id]['rel'] 
        
        # sanity check
        assert len(rel_bbox)!=0, 'each bounding box must have a relationship'
        assert len(data_bbox[img_id])!=0, 'each image must have at least one bounding box'
        
        # add relationship  category
        for bbox in data_bbox[img_id]['bbox']:
            bbox['image_id'] = n_img
            
            if v2:
                bbox['category_id'] = MAP_CLASS[bbox['category_id']][1] - 1
                
            else:
                bbox['category_id'] = MAP_CLASS[bbox['category_id']][1] 

        
        if valtest_set:
            if iter_valtest <split_valtest:    
                dst_ann_rel['val'][str(n_img)] = list(rel_bbox)
                
                dst_val_bbox['images'].append(cand_info)        
                dst_val_bbox['annotations'].extend(data_bbox[img_id]['bbox'])   
                
                
                
            else:
                dst_ann_rel['test'][str(n_img)] = list(rel_bbox)
               
                dst_test_bbox['images'].append(cand_info)           
                dst_test_bbox['annotations'].extend(data_bbox[img_id]['bbox'])     
                
            iter_valtest +=1
        else:
            dst_ann_rel['train'][str(n_img)] = list(rel_bbox)
            
            dst_train_bbox['images'].append(cand_info)
            dst_train_bbox['annotations'].extend(data_bbox[img_id]['bbox'])  

        n_rel +=1
        
        n_img += 1
    
          
    with open(dst_train_file, 'w') as f: 
        json.dump(dst_train_bbox, f)

    
    if valtest_set:
        
        with open(dst_val_file, 'w') as f: 
            json.dump(dst_val_bbox, f) 
            
        with open(dst_test_file, 'w') as f: 
            json.dump(dst_test_bbox, f)
        

        
    with open(dst_rel_file, 'w') as f: 
        json.dump(dst_ann_rel, f)
        

                
            
              
    print(f'pruned images from Visual Genome are {len(image_ids)}')
    print(f'pruned images from Visual Genome with rel are {n_rel}')


    
    return bbox_offset, n_img, filename_s, n_rel
        


def modify_set(dir:str, ann_set:str, set_file:str, conv_map:Dict[int, int], cat_name:Dict[int, str], cat_stats:Dict[int,int]={})-> int:
    """ extends the current label set

    Args:
        dir (str): directory where labels are saved
        ann_set (str): annotations set to change
        set_file (str): set file name
        conv_map (Dict[int, int]): conversion dictionary
        cat_dict (Dict[int, str]): dict of  category id to  category  name  with TrafficGenome notation

    Returns:
        int : return the catefgory statistics on the specified set
    """
    
    cat_id = set()
    cat_list = []
    

    # remove tunnel from traffic genome since present in only 1 img
    # no  remove bbox from rel since not present (hopefully)
    ann_pruned = list(filter(lambda a: int(a['category_id']) != 17, ann_set['annotations']))


    for ann in  ann_pruned:
        old_id = ann['category_id'] 
        ann['category_id'] = conv_map[ ann['category_id'] ]
        curr_id = ann['category_id']
        
        if curr_id not in cat_id:
            
            cat_id.add(curr_id)
            
            cat_list.append({'supercategory':cat_name[str(old_id)], 'id': int(curr_id), 'name':cat_name[str(old_id)] } )

        if curr_id not in cat_stats.keys():
            cat_stats[cat_name[str(curr_id)]]['id'] = 0
            
        cat_stats[curr_id]['num']+=1
    # update the category according to new labeling
    ann_set['categories']  = list(cat_list)
    ann_set['annotations'] = ann_pruned 
   
    with open(dir + set_file, 'w') as f:
        json.dump( ann_set, f)
    
    return cat_stats
    
