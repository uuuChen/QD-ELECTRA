#!/usr/bin/env bash
python classify.py \
        --task_name 'qqp' \
        --train_cfg './config/train_mrpc.json' \
        --model_cfg './config/QDElectra_base.json' \
        --model_file None \
        --train_data_file './GLUE/glue_data/QQP/train.tsv' \
        --eval_data_file './GLUE/glue_data/QQP/eval.tsv' \
        --vocab './data/vocab.txt' \
        --save_dir './finetune/qqp/' \
        --log_dir './finetune/qqp/log_dir/' \
        --distill True \
        --quantize True \
        --imitate_tinybert False \
        --pred_distill True