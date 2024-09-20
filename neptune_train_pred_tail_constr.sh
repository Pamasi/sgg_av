#!/bin/bash
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNDcyMGQ5Mi05YjAzLTRjN2QtYjQ0Yi1kOTM1ZDFiZWI3ODQifQ=="
export NEPTUNE_PROJECT="russo.alessandro199/SGG-VG"
export NEPTUNE_CUSTOM_RUN_ID=`date +"%Y%m%d%H%M%s%N" | md5sum`

# cd /home/pdimasi/sgg_av

# find port
export MASTER_PORT=16200

LOG_ELASTIC_DIR=log_torchrun
dataset=tg
img_folder=coco_mix_dataset_v2/images
ann_path=coco_mix_dataset_v2/
batch=10
num_gpu=1
num_workers=2

# activate conda
eval "$(conda shell.bash hook)"
conda activate sgg_av_env
exec
# remove old file from neptune dir to save space
# neptune clear
output_dir=/home/odin/dimasi_thesis/ckpt/nesy/pred_hard_constr_phi3_log
mkdir -p $output_dir

echo $output_dir

NESY_START_EPOCH=0
#torchrun    --nproc_per_node=$num_gpu --nnodes=1   --master_port $MASTER_PORT  --log_dir $LOG_ELASTIC_DIR \
#train_reltr.py --dataset $dataset --img_folder $img_folder --in_xywh --ann_path $ann_path --batch_size $batch   --num_workers $num_workers  \
#               --epochs 210 --save_freq 10  --rel_loss_coef 1.5  --enc_layers 5 --set_cost_bbox 8 --eos_coef 0.4   --cyclic_lr  --use_neptune True \
#             --neg_constr_path constraint/tgv2_so_neg_constr.csv  --pos_constr_path constraint/tgv2_pos_constr_v3.csv --enable_ltn True \
#            --different_p --sat_loss_coef 0.1 --nms_sat_enable --rel_operator 1 --univ_step 2 --p_epoch 50 --rc_enable  --univ_p 6  --output_dir $output_dir --start_epoch $NESY_START_EPOCH  > pred_tail_constr_fst.log

output_dir=/home/odin/dimasi_thesis/ckpt/nesy/pred_tail_constr_phi3_manual
mkdir -p $output_dir

echo $output_dir

torchrun    --nproc_per_node=$num_gpu --nnodes=1   --master_port $MASTER_PORT  --log_dir $LOG_ELASTIC_DIR \
train_reltr.py --dataset $dataset --img_folder $img_folder --in_xywh --ann_path $ann_path --batch_size $batch   --num_workers $num_workers  \
               --epochs 210 --save_freq 10  --rel_loss_coef 1.5  --enc_layers 5 --set_cost_bbox 8 --eos_coef 0.4   --cyclic_lr  --use_neptune True \
             --neg_constr_path constraint/tgv2_neg_constr_pair_rel_tail.csv  --pos_constr_path constraint/tgv2_pos_constr_pair_rel_tail.csv --enable_ltn True \
            --sat_loss_coef 0.1 --nms_sat_enable --rel_operator 1 --univ_step 2 --p_epoch 50 --rc_enable  --univ_p 6  --output_dir $output_dir --start_epoch $NESY_START_EPOCH  > pred_tail_constr_fst.log



