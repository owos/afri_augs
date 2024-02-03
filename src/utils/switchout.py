# !pip install datasets -q
from torch.autograd import Variable
import torch
import numpy as np
import random, json
import argparse
from transformers import AutoTokenizer, set_seed
from datasets import load_dataset
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from torch.autograd import Variable

#set_seed(6500)

flores_codes = {
    "bemba" : "bem_Latn",
    "bem" : "bem_Latn",
    "fon" : "fon_Latn",
    "hausa" : "hau_Latn",
    "hau" : "hau_Latn",
    "igbo" : "ibo_Latn",
    "ibo" : "ibo_Latn",
    "kinyarwanda" : "kin_Latn",
    "kin" : "kin_Latn",
    "twi" : "twi_Latn",
    "yoruba" : "yor_Latn",
    "yor" : "yor_Latn",
    "swahili" : "swh_Latn",
    "swa" : "swh_Latn",
    "wolof" : "wol_Latn",
    "wol" : "wol_Latn",
    "zulu" : "zul_Latn",
    "zul" : "zul_Latn",
    "french" : "fra_Latn",
    "fr" : "fra_Latn",
    "english" : "eng_Latn",  
    "en" : "eng_Latn",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Causal Language Inference Script")
    
    parser.add_argument("--data_path", "-d", type=str, default="masakhane/mafand",
                        help="path to the data on huggingface.")
    
    parser.add_argument("--tokenizer", "-tok", type=str, default="masakhane/mafand",
                        help="path to the tokenizer on huggingface.")
    
    parser.add_argument("--switch_type", "-t", type=str, choices=['in_lang', 'out_lang'], required=True,
                        help="Type of switch to do")
    
    parser.add_argument("--tau", "-ta", type=float, required=True, default=0.2,
                        help="tau value.")

    parser.add_argument("--src_lang", "-src", type=str, required=True,
                         help="Specify the source language(s), e.g., 'en'")

    parser.add_argument("--tgt_lang", "-tgt", type=str, required=True,
                        help="Specify the target language(s), e.g., 'yor'")
    
    parser.add_argument("--output_file", "-out", type=str, required=True,
                        help="Specify the directory you want to save the output dataset to, e.g., 'en-yor_sw-data'")
    args = parser.parse_args()
    return args


def hamming_distance_sample(sents: torch.Tensor, tau: int, bos_id: int, eos_id: int, pad_id: int, switch_type: str, vocab_size = None) -> torch.Tensor:
    """
    Sample a batch of corrupted examples from sents.
    Args:
        sents: Tensor [batch_size, n_steps]. The input sentences.
        tau: Temperature (int) (0 < tau < 1).
        vocab_size: to create valid samples (int).
        bos_id: id of the beginning of sentence token (int).
        eos_id: id of the end of sentence token (int).
        pad_id: id of the padding token (int).
    Returns:
        sampled_sents: Tensor [batch_size, n_steps]. The corrupted sentences.
    """

    # Existing code for mask and logits
    batch_size, n_steps = sents.size()
    mask = torch.eq(sents, bos_id) | torch.eq(sents, eos_id) | torch.eq(sents, pad_id)
    lengths = mask.mul(-1).add(1).float().sum(dim=1)
    
    logits = torch.arange(n_steps).mul(-1).unsqueeze(0).expand_as(sents).contiguous().masked_fill_(mask, np.iinfo(np.int64).min)
    logits = Variable(logits)

    probs = F.softmax(logits.float().mul_(tau), dim=1)
    base_corruption = torch.ones(batch_size, dtype=torch.long) * 2
    additional_corruptions = Categorical(probs).sample()
    num_words = base_corruption + additional_corruptions

    # sample the corrupted positions.
    corrupt_pos = num_words.data.float().div(lengths).unsqueeze(
        1).expand_as(sents).contiguous().masked_fill_(mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).bool()
    total_words = int(corrupt_pos.sum())

    # sample the corrupted values, which will be added to sents
    corrupt_val = torch.LongTensor(total_words)

    if switch_type == "in_lang":
        corrupt_val = torch.tensor(random.choices(vocab_size, k=len(corrupt_val)))
        corrupts = torch.zeros(batch_size, n_steps).long()
        corrupts = corrupts.masked_scatter(corrupt_pos, corrupt_val)
        mask = corrupts != 0

        sents[mask] = corrupts[mask]
        return sents

    corrupt_val = corrupt_val.random_(1, vocab_size)
    corrupts = torch.zeros(batch_size, n_steps).long()
    corrupts = corrupts.masked_scatter(corrupt_pos, corrupt_val)
    sampled_sents = sents.add(Variable(corrupts)).remainder(vocab_size)
    return sampled_sents


def apply_switchout(examples, src_tokenizer, trg_tokenizer, switch_type, source_lang, target_lang, tau, bos_id, eos_id, pad_id, padding, vocab_size_src, vocab_size_tgt):

    inputs = examples['translation'][source_lang]
    model_inputs = src_tokenizer(
        inputs, padding=padding, truncation=True, return_tensors="pt"
    )
    model_inputs["input_ids"] = [
        hamming_distance_sample(
            inp.reshape(1, -1), tau, bos_id, eos_id, pad_id, switch_type, vocab_size=vocab_size_src
        ).squeeze()
        for inp in model_inputs["input_ids"]
    ]
    prediction_model = src_tokenizer.batch_decode(
                model_inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    targets = examples['translation'][target_lang]
    labels = trg_tokenizer(targets, padding=padding, truncation=True, return_tensors="pt")
    labels["input_ids"] = [
        hamming_distance_sample(
            trgt.reshape(1, -1), tau, bos_id, eos_id, pad_id, switch_type, vocab_size=vocab_size_tgt
        ).squeeze()
        for trgt in labels["input_ids"]
    ]
    prediction_label = trg_tokenizer.batch_decode(
                labels["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    model_inputs["labels"] = labels["input_ids"]
    examples["translation"][source_lang] = prediction_model[0]
    examples["translation"][target_lang] = prediction_label[0]
                                    #(change model input to decoded value, look at model input and take only the input id)
    return examples['translation']      #change model input to list before decoding

def main ():
    args = parse_args()
    data = load_dataset(args.data_path, f"{args.src_lang}-{args.tgt_lang}")
    data = data.shuffle()
    data = data['train']
    source_lang = args.src_lang
    target_lang = args.tgt_lang
    src_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=flores_codes[source_lang])
    trg_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=flores_codes[target_lang])

    tau = args.tau
    bos_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.bos_token)
    eos_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.eos_token)
    pad_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.pad_token)
    padding = "max_length"
    output_file = args.output_file
    switch_type = args.switch_type
    #dynamic
    if args.switch_type== "out_lang":
        vocab_size = src_tokenizer.vocab_size
        sw_args = {"src_tokenizer":src_tokenizer, "trg_tokenizer":trg_tokenizer, "switch_type":switch_type, "source_lang":source_lang, "target_lang":target_lang, "tau":tau, "bos_id":bos_id, "eos_id":eos_id, "pad_id":pad_id, "padding":padding, "vocab_size_src":vocab_size, "vocab_size_tgt":vocab_size}
    elif  args.switch_type== "in_lang":
        all_text_src = ' '.join([ex[source_lang] for ex in data["translation"]])
        all_text_tgt = ' '.join([ex[target_lang] for ex in data["translation"]])
        tokens_src = src_tokenizer(all_text_src, truncation=True)
        tokens_tgt = trg_tokenizer(all_text_tgt, truncation=True)
        vocab_size_src = list(set(tokens_src['input_ids']))
        vocab_size_tgt = list(set(tokens_tgt['input_ids']))
        #1 combine dataset to one giant string - do seperately for the source and target lang
        #2 tokenize it and remove duplicates by combining to a set and back to a list
        # pass the list here - you will have 2 vocab size variable, one for source lang, the otheer for target lag
        sw_args = {"src_tokenizer":src_tokenizer, "trg_tokenizer":trg_tokenizer, "switch_type":switch_type, "source_lang":source_lang, "target_lang":target_lang, "tau":tau, "bos_id":bos_id, "eos_id":eos_id, "pad_id":pad_id, "padding":padding, "vocab_size_src":vocab_size_src, "vocab_size_tgt":vocab_size_tgt}
    
    sw_data = data.select(range(int(len(data) * tau)))
    sw_data = sw_data.map(apply_switchout, fn_kwargs=sw_args)

    #1. save sw_data to json format and push to hub
    with open(output_file, "w") as file:
        json.dump(sw_data, file)

    print("Data stored successfully!")
#add logging statements to code
if __name__ == "__main__":
    main()