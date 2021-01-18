#!/usr/bin/env bash
python classify.py \
        --task_name 'mrpc' \
        --train_cfg './config/train_mrpc.json' \
        --model_cfg './config/QDElectra_base.json' \
        --model_file None \
        --train_data_file './data/msr_paraphrase_train.txt' \
        --eval_data_file './data/msr_paraphrase_train.txt' \
        --vocab './data/vocab.txt' \
        --save_dir './finetune/mrpc/' \
        --log_dir './finetune/mrpc/log_dir/' \
        --distill True \
        --quantize True \
        --gradually_distill True \
        --imitate_tinybert False \
        --pred_distill True