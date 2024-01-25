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

num_gpus = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang_code_dict = {'en': 'eng_Latn', 'zul': 'zul_Latn'}

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


def main(args):

    model_path = args.modelpath
    src_lang = args.sourcelang
    tgt_lang = args.targetlang

    data_train = args.traindatapath
    data_dev = args.devdatapath
    data_test = args.testdatapath

    # load train, dev test data
    df_train = pd.read_csv(data_train, sep='\t')
    df_dev = pd.read_csv(data_dev, sep='\t')
    df_test = pd.read_csv(data_test, sep='\t')

    tokenizer = NllbTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


    if len(num_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=num_gpus)
    
    model.to(device)
    model = model.module if len(num_gpus) > 1 else model

    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
        scale_parameter=False, relative_step=False, lr=1e-4, clip_threshold=1.0, 
        weight_decay=1e-3)


    batch_size = 20
    epochs = 5
    steps = (len(df_train) // batch_size ) * epochs
    max_length = 256
    warmup_steps = 1_000
    training_steps = 2000000
    losses = []
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    LANGS = [(src_lang, lang_code_dict[src_lang]), (tgt_lang, lang_code_dict[tgt_lang])]

    def get_batch_pairs(batch_size, data=df_train):
        (l1, lang1), (l2, lang2) = random.sample(LANGS, 2)
        xx, yy = [], []
        for _ in range(batch_size):
            item = data.iloc[random.randint(0, len(data)-1)]
            xx.append(item[l1].lower())
            yy.append(item[l2].lower())
        return xx, yy, lang1, lang2

    model.train()
    cleanup()

    FINETUNE_BASE_PATH = '../finetuned_model/'

    if not os.path.exists(FINETUNE_BASE_PATH):
        os.mkdir(FINETUNE_BASE_PATH)

    MODEL_SAVE_PATH = os.path.join(FINETUNE_BASE_PATH, 'nllb_{}_{}/'.format(src_lang, tgt_lang))

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    tq = trange(len(losses), training_steps)
    def evaluate_model(model, tokenizer, val_dataset, max_length):
        model.eval()  # Set the model to evaluation mode
        val_losses = []
        with torch.no_grad():  # Disable gradient calculation
            for xx, yy, lang1, lang2 in val_dataset:
                tokenizer.src_lang = lang1
                x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

                tokenizer.tgt_lang = lang2
                y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
                y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                loss = model(**x, labels=y.input_ids).loss
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        return avg_val_loss
    
    for epoch in epochs:
        for i in tq:
            xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

            tokenizer.tgt_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
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
                avg_val_loss = evaluate_model(model, tokenizer, val_dataset, max_length)
                model.train()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='facebook/nllb-200-distilled-600M')
    parser.add_argument("--sourcelang", type=str, default='en')
    parser.add_argument("--targetlang", type=str, default='zul')
    parser.add_argument("--traindatapath", type=str, default='../lafand-mt-data/en-zul/train.tsv')
    parser.add_argument("--devdatapath", type=str, default='../lafand-mt-data/en-zul/dev.tsv')
    parser.add_argument("--testdatapath", type=str, default='../lafand-mt-data/en-zul/test.tsv')
    
    args = parser.parse_args()
    main(args)