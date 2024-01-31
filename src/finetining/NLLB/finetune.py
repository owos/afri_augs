import argparse
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from tqdm.auto import tqdm, trange
import gc
import random
import numpy as np
import torch
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
import os
from typing import List
from torch.utils.data import DataLoader
import json
import string
import unidecode

num_gpus = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang_code_dict = {

    "yor": "yor_Latn",
    "hau": "hau_Latn",
     "wolof": "wol_Latn",
     "fon": "swa_Latn",
     "en": "eng_Latn"
}
if len(num_gpus) > 1:
    print("Let's use", num_gpus, "GPUs!")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpus)

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def adjust_tokenizer(tokenizer, new_lang_code: str):
    """
    Add a new language token to the tokenizer vocabulary
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang_code in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang_code] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang_code
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang_code not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang_code)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

# my edits for processing file
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


def main(args):

    model_path = args.modelpath
    src_lang = args.sourcelang
    tgt_lang = args.targetlang

    data_train = args.traindatapath
    data_dev = args.devdatapath
    multiway = args.ismultiway


    batch_size = 16
    epochs = 2
    steps = (len(data_train) // batch_size ) * epochs
    max_length = 256
    warmup_steps = 1_00
    training_steps = 60000
    losses = []
    #seed_value = 42
    
    train_data = process_file(data_train)
    dev_data = process_file(data_dev)

    #random.seed(seed_value)
   
   #create a List for the indexes of the training sample
    train_indices = list(range(len(train_data)))
    random.shuffle(train_indices)

    tokenizer = NllbTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if (len(train_data) < batch_size):
        raise ValueError("The number of data samples must be greater than the batch size")
        
    if len(num_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=num_gpus)
    
    model.to(device)
    model = model.module if len(num_gpus) > 1 else model

    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
        scale_parameter=False, relative_step=False, lr=1e-4, clip_threshold=1.0, 
        weight_decay=1e-3)



    
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    LANGS = [(src_lang, lang_code_dict[src_lang]), (tgt_lang, lang_code_dict[tgt_lang])]
    

    def get_batch_pairs(batch_size, current_step):

        (l1, lang1), (l2, lang2) = LANGS
        xx, yy = [], []

        #slice out the data for the batch
        data = [train_data[item] for item in train_indices[current_step % batch_size: current_step % batch_size + batch_size]]
       
        for index, item in enumerate(data):
            xx.append(item["translation"][l1].lower())
            yy.append(item["translation"][l2].lower())
        return xx, yy, lang1, lang2, l1, l2


    model.train()
    cleanup()

    FINETUNE_BASE_PATH = '../finetuned_model/'

    if not os.path.exists(FINETUNE_BASE_PATH):
        os.mkdir(FINETUNE_BASE_PATH)

    MODEL_SAVE_PATH = os.path.join(FINETUNE_BASE_PATH, 'nllb_{}_{}/'.format(src_lang, tgt_lang))

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    tq = trange(len(losses), training_steps)

    def evaluate_model(model, tokenizer, val_dataset, max_length, lang1, lang2, l1, l2):
        model.eval()  # Set the model to evaluation mode
        val_losses = []
        random.shuffle(val_dataset)
        with torch.no_grad():  # Disable gradient calculation
            for val_samp in val_dataset:
                xx, yy = val_samp["translation"][l1], val_samp["translation"][l2]
                tokenizer.src_lang = lang1
                x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

                tokenizer.tgt_lang = lang2
                y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
                y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                loss = model(**x, labels=y.input_ids).loss
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        return avg_val_loss
    
    for _ in range(epochs):
        for i in tq:

            #shuffle the indices of the data after a complete use of training data
            if i % batch_size == 0:
                random.shuffle(train_indices)

            xx, yy, lang1, lang2, l1, l2 = get_batch_pairs(batch_size, current_step=i)
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

            tokenizer.tgt_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
    
            # switch the dataset xx and yy to run the training in the other way
            if multiway:
                tokenizer.src_lang = lang2
                x = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

                tokenizer.tgt_lang = lang1
                y = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
                y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                loss += model(**x, labels=y.input_ids).loss
                

            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if i % 500 == 0:
                print('Training Step: {} with Loss: {}'.format(i+1, np.mean(losses[-500:])))

            if i % 100 == 0 and i > 0:
                model.save_pretrained(MODEL_SAVE_PATH)
                tokenizer.save_pretrained(MODEL_SAVE_PATH)
                avg_val_loss = evaluate_model(model, tokenizer, dev_data, max_length, lang1, lang2, l1, l2)
                print("Checkpoint validation loss: {}".format(avg_val_loss))
                model.train()
        cleanup()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='facebook/nllb-200-distilled-600M')
    parser.add_argument("--sourcelang", type=str, default='en')
    parser.add_argument("--targetlang", type=str, default='yor')
    parser.add_argument("--traindatapath", type=str, default='./test.json')
    parser.add_argument("--devdatapath", type=str, default='./test.json')
    parser.add_argument("--ismultiway", type=bool, default=False)
    
    args = parser.parse_args()
    main(args)
 