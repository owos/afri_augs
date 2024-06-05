#!/bin/bash
AUGMENTATION_TECHNIQUE="sentence_concat"
language_mapper=(["en-yor"]="en,yo" ["en-hau"]="en,ha" ["en-tsn"]="en,ha" ["fr-fon"]="fr,yo" ["fr-wol"]="fr,wo" ["en-swa"]="en,sw")

for language_pair in "en-yor" "en-hau" "en-tsn" "fr-fon" "fr-wol" "en-swa"
do
    IFS=',' read -ra LANGS <<< "${language_mapper[$language_pair]}"
    for SEED in {1..3}
    do
        for augmentation_percent in 10 20 30 40
        do
            echo "Running $language_pair, seed number $SEED and $augmentation_percent"
            CUDA_VISIBLE_DEVICES=0 python \
                ../src/finetining/m2m100/m2m100.py \
                --model_name_or_path facebook/m2m100_418M \
                --do_train \
                --do_eval \
                --do_predict \
                --source_lang ${LANGS[0]} \
                --target_lang ${LANGS[1]} \
                --dataset_name masakhane/mafand \
                --dataset_config_name $language_pair \
                --output_dir ../experiments/$AUGMENTATION_TECHNIQUE/m2m/$language_pair/$augmentation_percent/$SEED \
                --per_device_train_batch_size=16 \
                --per_device_eval_batch_size=16 \
                --overwrite_output_dir \
                --predict_with_generate \
                --num_train_epochs 5 \
                --do_aug \
                --aug_file ../data/$AUGMENTATION_TECHNIQUE/$language_pair/${AUGMENTATION_TECHNIQUE}_$augmentation_percent.json \
                --seed $SEED \
                --save_total_limit 3 \
                --gradient_accumulation_steps 4 \
                
        done
    done
done