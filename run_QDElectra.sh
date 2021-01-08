#!/usr/bin/env bash
python QDElectra_pretrain.py \
        --data_file './data/wiki.train.tokens' \
        --vocab './data/vocab.txt' \
        --train_cfg './config/QDElectra_pretrain.json' \
        --model_cfg 'config/QDElectra_base.json' \
        --max_pred 100 \
        --max_len 128 \
        --mask_prob 0.15 \
        --save_dir './saved_QDElectra' \
        --log_dir './logs_QDElectra' \
        --quantize True \
