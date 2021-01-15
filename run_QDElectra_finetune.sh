#!/usr/bin/env bash
python classify.py \
        --task_name 'mrpc' \
        --mode 'eval' \
        --train_cfg './config/train_mrpc.json' \
        --model_cfg './config/QDElectra_base.json' \
        --data_file './data/msr_paraphrase_train.txt' \
        --vocab './data/vocab.txt' \
        --save_dir './saved_QDElectra_tuned/' \
        --pred_distill True \
        --quantize True \
        --imitate_tinybert False