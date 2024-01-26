from torch.autograd import Variable
import torch
import numpy as np
import random
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Causal Language Inference Script")
    
    parser.add_argument("--data_path", "-d", type=str, default="masakhane/mafand",
                        help="path to the data on huggingface.")
    
    parser.add_argument("--tokenizer", "-tok", type=str, default="masakhane/mafand",
                        help="path to the tokenizer on huggingface.")
    
    parser.add_argument("--switch_type", "-t", type=str, choices=['in_lang', 'out_lang'], required=True,
                        help="Type of switch to do")
    
    parser.add_argument("--switch_percent", "-p", type=str, required=True, default=0.2,
                        help="tau value.")
    parser.add_argument("--tau", "-ta", type=str, required=True, default=0.2,
                        help="tau value.")

    parser.add_argument("--src_lang", "-src", type=str, required=True,
                        help="Specify the source language(s), e.g., 'en'")

    parser.add_argument("--tgt_lang", "-tgt", type=str, required=True,
                        help="Specify the target language(s), e.g., 'yor'")


    args = parser.parse_args()
    return args


def hamming_distance_sample(sents: torch.Tensor, tau: int, bos_id: int, eos_id: int, pad_id: int, vocab_size: int, method: str) -> torch.Tensor:
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

    mask = torch.eq(sents, bos_id) | torch.eq(
        sents, eos_id) | torch.eq(sents, pad_id)
    lengths = mask.mul(-1).add(1).float().sum(dim=1)
    batch_size, n_steps = sents.size()

    # first, sample the number of words to corrupt for each sentence
    logits = torch.arange(n_steps)
    logits = logits.mul_(-1).unsqueeze(0).expand_as(
        sents).contiguous().masked_fill_(mask, np.iinfo(np.int64).min)
    logits = Variable(logits)
    probs = torch.nn.functional.softmax(logits.float().mul_(tau), dim=1)
    num_words = torch.distributions.Categorical(probs).sample()

    # sample the corrupted positions.
    corrupt_pos = num_words.data.float().div(lengths).unsqueeze(
        1).expand_as(sents).contiguous().masked_fill_(mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).bool()
    total_words = int(corrupt_pos.sum())

    # sample the corrupted values, which will be added to sents
    breakpoint()
    corrupt_val = torch.LongTensor(total_words)
    if method == "in-lang":
        corrupt_val = torch.tensor(random.choices(vocab_size, k=len(corrupt_val)))
        corrupts = torch.zeros(batch_size, n_steps).long()
        corrupts = corrupts.masked_scatter(corrupt_pos, corrupt_val)
        sampled_sents = sents.add(Variable(corrupts)).remainder(len(vocab_size))  # this link has not been verified
        return sampled_sents
   
    corrupt_val = corrupt_val.random_(1, vocab_size)
    corrupts = torch.zeros(batch_size, n_steps).long()
    corrupts = corrupts.masked_scatter(corrupt_pos, corrupt_val)
    sampled_sents = sents.add(Variable(corrupts)).remainder(vocab_size)
    return sampled_sents




def apply_switchout(examples, tokenizer, switch_type, switch_percent, source_lang, target_lang):
    # update the args to taking all the necc. parameters 

    inputs = [ex[source_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, padding=padding, truncation=True, return_tensors="pt"
    )
    model_inputs["input_ids"] = [
        hamming_distance_sample(
            inp.reshape(1, -1), tau, bos_id, eos_id, pad_id, vocab_size
        ).squeeze()
        for inp in model_inputs["input_ids"]
    ]
    targets = [ex[target_lang] for ex in examples["translation"]]
    labels = tokenizer(targets, padding=padding, truncation=True, return_tensors="pt")
    labels["input_ids"] = [
        hamming_distance_sample(
            trgt.reshape(1, -1), tau, bos_id, eos_id, pad_id, vocab_size
        ).squeeze()
        for trgt in labels["input_ids"]
    ]
    # what you will do moving forward from here will depend on the model you are working with, for this example I used mbart-50,
    if padding == "max_length" and False:  # data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main ():
    #import necessary pacakages
    args = parse_args()
    data = load_dataset(args.data_path, f"{args.src_lang}'-'{args.src_lang}")
    data = data.shuffle()
    data = data['train']
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    
    tau = 0.2
    bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    padding = "max_length"
    #dynamic
    if args.switch_type== "out_lang":
        vocab_size = tokenizer.vocab_size
    elif  args.switch_type== "in_lang":
        #1 combine dataset to one giant string 
        #2 tokenize it and remove duplicates by combining to a set and back to a list
        vocab_size = None  # pass the list here
    
    
    sw_args = {"..."}
    
    sw_data = data.select(range[len(args.switch_percent)]) #select the percentage of the data to be used 
    sw_data = sw_data.map(apply_switchout, fn_kwargs=sw_args)
    
    #1. save sw_data to json format and push to hub
    




