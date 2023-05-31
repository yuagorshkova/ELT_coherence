#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_PROJECT=tc \
python train_stl.py \
--tokenized_datasets /home/jovyan/vkr/text_coherence/tokenized_data/sentence_tokenize_function \
--gcdc /home/jovyan/vkr/training_data/gcdc \
--ellipse /home/jovyan/vkr/training_data/ellipse_train \
--eval /home/jovyan/vkr/training_data/ellipse_test \
--run_id stl_base_ellipse,gcdc_l2_freeze \
--gpu=0