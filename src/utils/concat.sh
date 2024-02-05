python concatenate_sentence.py \
    --subset_name fr-wol \
    --seperator_token "</s>" \
    --do_back_translation True \
    --number_of_concatenations 1008 \
    --back_translation_model masakhane/mbart50_yor_en_news \
    --back_translation_percent 0.3 \
    --destination_filepath ../../data/sentence_concat