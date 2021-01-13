#!/usr/bin/env bash
python electra_pretrain.py \
        --data_file './data/wiki.train.tokens' \
        --vocab './data/vocab.txt' \
        --train_cfg './config/electra_pretrain.json' \
        --model_cfg './config/electra_small.json' \
        --max_pred 100 \
        --max_len 128 \
        --mask_prob 0.15 \
        --save_dir './saved_electra' \
        --log_dir './logs_electra' \

