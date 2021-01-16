#!/usr/bin/env bash
python classify.py \
        --task_name 'mrpc' \
        --train_cfg './config/train_mrpc.json' \
        --model_cfg './config/QDElectra_base.json' \
        --data_file './data/msr_paraphrase_train.txt' \
        --vocab './data/vocab.txt' \
        --save_dir './finetune/mrpc/' \
        --log_dir './finetune/mrpc/log_dir/' \
        --pred_distill True \
        --quantize True \
        --imitate_tinybert False