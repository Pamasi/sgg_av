# Copyright (c) Paolo Dimasi, Politecnico di Torino.
from dataset import H5ToCoco


if __name__ == '__main__':    
    n_img = 997
    file_set = set()
    
    converter = H5ToCoco('traffic_genome_format/traffic_genome_format.h5', 'coco_traffic_genome_v2', 
                         'traffic_genome_img', 'traffic_genome_format/traffic_genome_format.json', v2=True)
    converter.create_all()
    converter.stats()