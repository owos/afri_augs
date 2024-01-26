import torch
import argparse

from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device_map(model_name, device, do_int8):
    if device == "a100-40g":
        return "auto"

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {0: "18GiB"}
    for i in range(1, 6):
        d[i] = "26GiB"
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"]
    )
    print(device_map)
    del model
    return device_map

if __name__ == "__main__":
    # CLI argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bigscience/bloomz-7b1-mt", choices=["meta-llama/Llama-2-7b-chat-hf", "bigscience/bloomz-7b1-mt", "mistralai/Mistral-7B-Instruct-v0.2"])
    parser.add_argument("--device", type=str, choices=["a100-40g", "v100-32g"], default="a100-40g")
    parser.add_argument("--do_int8", action="store_true")
    parser.add_argument("--template_prompt", type=str, default="Puma is a {}")
    args = parser.parse_args()

    # Model Initialization
    model_id = f"{args.model_name}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        repo_type="hf",  # Specify the repository type explicitly
        device_map=get_device_map(model_id, args.device, args.do_int8),
        torch_dtype=torch.int8 if args.do_int8 else torch.float16
    )

    # Tokenizer Initialization
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Text generation
    generate_kwargs = {
        "max_new_tokens": 200,
        "min_new_tokens": 100,
        "temperature": 0.1,
        "do_sample": False,
        "top_k": 4,
        "penalty_alpha": 0.6,
    }

    prompt = args.template_prompt.format(args.model_name.capitalize())
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if input_ids[0][-1] == 2:
            input_ids = input_ids[:, :-1]
        input_ids = input_ids.to(0)
        generated_ids = model.generate(input_ids, **generate_kwargs)
        result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        print(result)