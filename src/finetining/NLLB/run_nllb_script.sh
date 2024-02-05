#!/bin/bash
echo "starting up the script"
python nllb.py \
     --model_name_or_path facebook/nllb-200-distilled-600M \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang  en \
    --target_lang yor \
    --dataset_name masakhane/mafand \
    --dataset_config_name en-yor \
    --output_dir ../experiments \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 5 \
    --do_aug \
    --aug_file ./test.json \
    --seed 42 \
    --save_total_limit 3 \
    --gradient_accumulation_steps 4 \