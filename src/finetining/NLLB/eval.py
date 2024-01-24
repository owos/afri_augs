import nltk
import unidecode
import json
import sys
import re
import string
import gc
import torch


from nltk.translate import bleu_score
from typing import List
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from tqdm import trange
from argparse import ArgumentParser

nltk.download('punkt')


lang_code = {

    "yor": "yor_Latn",
    "hau": "hau_Latn",
     "wolof": "wol_Latn",
     "fon": "swa_Latn",
     "en": "eng_Latn"

}


def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate(model, tokenizer, text, src_lang, tgt_lang, a=32, b=3 ):

    #model, tokenizer = load_model(model_name)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    model.eval()
    translation = model.generate(**inputs.to(model.device), forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang), max_new_tokens = int(a + b * inputs.input_ids.shape[1]), num_beams = 4)
    return tokenizer.batch_decode(translation, skip_special_tokens = True)
    


def batched(model_name, dataset_path: string, source_lang, target_lang, batch_size = 16):

    model, tokenizer = load_model(model_name)
    data = process_file(dataset_path=dataset_path)
    sorted_data = sorted(data, key= lambda item: len(item["translation"][source_lang]), reverse=True)

    result = []
    for i in trange(0, len(sorted_data), batch_size):

        sorted_batch = sorted_data[i: i + batch_size]
        batch = [item["translation"][source_lang] for item in sorted_batch]
        translated_batch = translate(model, tokenizer, text=batch, src_lang=lang_code[source_lang], tgt_lang=lang_code[target_lang])
        
        for j, translation in enumerate(translated_batch):
            sorted_batch[j]["translation"]["machine_translated"] = translation
        result.extend(sorted_batch)

    with open("output.json", "w", encoding="utf-8") as file:
        for item in result:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(f"{json_string}\n")
   
    bleu_per_line, average_bleu_score = calculate_bleu_score(file_path="./output.json", target_lang=target_lang)
    return bleu_per_line, average_bleu_score

    
def process_file(dataset_path: string) -> List:

    '''
    This function extract the json objects from a json file 
    '''

    with open(dataset_path, 'r', encoding='utf-8') as file:

        data = []
        for line in file:
            
            json_obj = json.loads(line)
            data.append(json_obj)

    return data



def _normalize_sentence(sentence: string):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def unidecode_str(text):
        return unidecode.unidecode(str(text))

    return white_space_fix(remove_articles(remove_punc(lower(unidecode_str(sentence)))))



def calculate_bleu_score(file_path, target_lang):


    """
    Calculate the average BLEU score for a list of BLEU scores.

    :param bleu_scores: List of BLEU scores.
    :return: bleu_per_line and Average BLEU score.
    """

    data = process_file(file_path)
    bleu_per_line = []

    for  i, item in enumerate(data):
        candidate = _normalize_sentence(item["translation"]["machine_translated"])
        reference = _normalize_sentence(item["translation"][target_lang])

        tokenize_candidate = nltk.word_tokenize(candidate)
        tokenize_reference = nltk.word_tokenize(reference)
        
        bl_score = bleu_score.sentence_bleu([tokenize_reference], tokenize_candidate, weights=(1,0,0,0))
        print(f"BLEU Score for Candidate {i + 1}: {bl_score}")
        bleu_per_line.append(bl_score)
    average_bleu_score = sum(bleu_per_line) / len(bleu_per_line)
    print(f"The Average BLEU Score is: {average_bleu_score}")
    return bleu_per_line, average_bleu_score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", help="error: no model name")
    parser.add_argument("--dataset_path", help="error: no inference file path")
    parser.add_argument("--source_lang", default="en")
    parser.add_argument("--target_lang", default="yor")

    args = parser.parse_args()
    print(args)
    batched(model_name=args.model_name, dataset_path=args.dataset_path, source_lang=args.source_lang, target_lang=args.target_lang)
   


