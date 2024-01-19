# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved

from dataset import add_vg2tg
import argparse, shutil, json
# we assume that any error is due to labeller distraction, thus we do not consider it
if __name__ == '__main__':    
    
    # there are 997 image in the trainset, but 99 overall counting from 0
    offset_img = 1000
    # start from tg img 
    n_rel_img = 1000
    
    tot_img = 0
    
    file_set = set()
    
    parser = argparse.ArgumentParser('Generate Augmented Traffic Genome dataset')
    parser.add_argument('--img_with_rel', default=True, type=bool,
                        help='whether or not to consider only image with relationship during data extraction from Visual Genome')
    parser.add_argument('--vg_folder', default='vg_label', type=str,
                        help='folder containing Visual Genome annotations')
    
    
    
    args = parser.parse_args()

    prefix_dir = args.vg_folder


    try:
        shutil.rmtree('coco_mix_dataset_v2')
    except:
        print('found already the directory!')
        
    shutil.copytree('coco_traffic_genome_v2', 'coco_mix_dataset_v2')
    
    # find the id related to the bbox 
    bb_id = [0,0,0]
    with open('coco_mix_dataset_v2/train.json', 'r') as f: 
            tmp = json.load(f)
            
            bb_id[0] =max(list(map(lambda ann: ann['id'], tmp['annotations'])))
    
    with open('coco_mix_dataset_v2/test.json', 'r') as f: 
            tmp = json.load(f)
            
            bb_id[1] =max(list(map(lambda ann: ann['id'], tmp['annotations'])))
 
    with open('coco_mix_dataset_v2/val.json', 'r') as f: 
            tmp = json.load(f)
            
            bb_id[2] =max(list(map(lambda ann: ann['id'], tmp['annotations'])))   
    
    offset_bb_id = max(bb_id) + 1
    
    
    offset_bb_id, offset_img, tmp_f, tmp_rel = add_vg2tg(offset_bb_id, prefix_dir+ '/val.json',  prefix_dir+ '/rel.json', 'val', 
                                'coco_mix_dataset_v2/train.json', 'coco_mix_dataset_v2/rel.json',
                                # valtest_set=args.img_with_rel,dst_val_file='coco_mix_dataset_v2/val.json', dst_test_file='coco_mix_dataset_v2/test.json', 
                                offset_id_img= offset_img)


    tot_img += offset_img
    n_rel_img += tmp_rel
    file_set = file_set.union(tmp_f)
    
    offset_bb_id, offset_img, tmp_f, tmp_rel = add_vg2tg(offset_bb_id, prefix_dir + '/test.json',  prefix_dir+  '/rel.json', 'test', 
                                'coco_mix_dataset_v2/train.json', 'coco_mix_dataset_v2/rel.json',  offset_id_img= offset_img)
    tot_img += offset_img
    n_rel_img += tmp_rel
    file_set = file_set.union(tmp_f)
    
    tot_img += offset_img
    offset_bb_id, offset_img, tmp_f, tmp_rel = add_vg2tg(offset_bb_id, prefix_dir +'/train.json',  prefix_dir +'/rel.json', 'train', 
                                'coco_mix_dataset_v2/train.json', 'coco_mix_dataset_v2/rel.json',  offset_id_img= offset_img)


    tot_img += offset_img
    n_rel_img += tmp_rel
    file_set =file_set.union(tmp_f)
    
    print()
    with open('coco_mix_dataset_v2/file_list.txt', 'w')  as fp:
        fp.writelines([ f + '\n' for  f in file_set ])
        
    
    print(f"number of images pruned from visual genome {len(file_set)}")
    print(f"current number of images containg relationship is {n_rel_img}")
    
    
