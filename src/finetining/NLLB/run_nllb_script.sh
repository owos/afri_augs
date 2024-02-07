#!/bin/bash
echo "starting up the script"

AUGMENTATION_TECHNIQUE="sentence_concat"

for language_pair in "en-yor" "en-hau"
do
    IFS='-' read -ra ADDR <<< "$language_pair"
    for SEED in {1..3}
    do
        for augmentation_percent in 10 40
        do
            echo "Running $language_pair, seed number $SEED and $augmentation_percent"
            CUDA_VISIBLE_DEVICES=0 python3 \
                nllb.py \
                --model_name_or_path facebook/nllb-200-distilled-600M \
                --do_train \
                --do_eval \
                --do_predict \
                --source_lang  ${ADDR[0]} \
                --target_lang ${ADDR[1]} \
                --dataset_name masakhane/mafand \
                --dataset_config_name $language_pair \
                --output_dir ../../../experiments/$AUGMENTATION_TECHNIQUE/$language_pair/$augmentation_percent/$SEED \
                --per_device_train_batch_size=4 \
                --per_device_eval_batch_size=4 \
                --overwrite_output_dir \
                --predict_with_generate \
                --num_train_epochs 5 \
                --do_aug \
                --aug_file ../../../data/$AUGMENTATION_TECHNIQUE/$language_pair/${AUGMENTATION_TECHNIQUE}_$augmentation_percent.json \
                --seed $SEED \
                --save_total_limit 3 \
                --gradient_accumulation_steps 4 \
                
        done
    done
done