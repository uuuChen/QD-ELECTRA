#!/usr/bin/env bash
python electra_pretrain.py \
        --data_file './data/wiki.train.tokens' \
        --vocab './data/vocab.txt' \
        --train_cfg './config/test_electra_pretrain.json' \
        --model_cfg 'config/electra_base.json' \
        --max_pred 75 --mask_prob 0.15 \
        --save_dir './saved' \
        --log_dir './logs'