#!/bin/bash

FILEID=1l1wL4kNMTOG3DYwW7GLms-C5mB6lu3E3
FILENAME=traffic_genome_img.zip

# download traffic genome 
curl "https://drive.usercontent.google.com/download?id={$FILEID}&confirm=xxx" -o $FILENAME

unzip traffic_genome_img.zip

# create the traffic genome label
python utils/tg_coco_v2.py

#extend  traffic genome using visual genome labeling from RelTR repo
python utils/vg2tg_v2.py

