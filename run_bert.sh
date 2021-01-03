#!/usr/bin/env bash
python bert_pretrain.py \
        --data_file './data/wiki.train.tokens' \
        --vocab './data/vocab.txt' \
        --train_cfg './config/bert_pretrain.json' \
        --model_cfg 'config/bert_base.json' \
        --max_pred 75 --mask_prob 0.15 \
        --save_dir './saved_bert' \
        --log_dir './logs_bert'