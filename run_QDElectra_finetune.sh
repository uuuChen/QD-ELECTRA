#!/usr/bin/env bash
python classify.py \
        --task_name 'mrpc' \
        --train_cfg './config/train_mrpc.json' \
        --model_cfg './config/QDElectra_base.json' \
        --train_data_file './GLUE/glue_data/QQP/train.tsv' \
        --eval_data_file './GLUE/glue_data/QQP/eval.tsv' \
        --vocab './data/vocab.txt' \
        --save_dir './finetune/mrpc/' \
        --log_dir './finetune/mrpc/log_dir/' \
        --pred_distill True \
        --quantize True \
        --imitate_tinybert False