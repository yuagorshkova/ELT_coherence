#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
TRANSFORMERS_OFFLINE=1 \
WANDB_PROJECT=tc \
python train_hierarchical.py \
--tokenized_datasets /home/jovyan/vkr/text_coherence/tokenized_data \
--gcdc /home/jovyan/vkr/training_data/gcdc \
--ellipse /home/jovyan/vkr/training_data/ellipse_train \
--eval /home/jovyan/vkr/training_data/ellipse_test \
--run_id hierarchal_base_ellipse,gcdc_lr5_freeze_ \
--gpu=0 