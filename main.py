import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def perform_translation(model, tokenizer, batch_texts):
    formatted_batch_texts = [f"{text}" for text in batch_texts]
    model_inputs = tokenizer(formatted_batch_texts, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**model_inputs)
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_texts

def perform_back_translation(original_texts, original_model, original_tokenizer, back_translation_model, back_translation_tokenizer):
    temp_translated_batch = perform_translation(original_model, original_tokenizer, original_texts)
    back_translated_batch = perform_translation(back_translation_model, back_translation_tokenizer, temp_translated_batch)
    return list(set(original_texts) | set(back_translated_batch))

def main():
    parser = argparse.ArgumentParser(description="Perform translation and back-translation with Seq2Seq models.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for tokenizer and model loading.")
    parser.add_argument("--back_translation_model_name", type=str, required=True, help="Model name for back translation.")
    parser.add_argument("--source_lang", type=str, required=True, help="Source language code.")
    parser.add_argument("--target_lang", type=str, required=True, help="Target language code.")
    parser.add_argument("--original_texts", nargs="+", required=True, help="Original texts for translation.")
    
    args = parser.parse_args()

    # Load models and tokenizer
    original_model, original_tokenizer = load_model_and_tokenizer(args.model_name)
    back_translation_model, back_translation_tokenizer = load_model_and_tokenizer(args.back_translation_model_name)

    # Perform translation
    translated_texts = perform_translation(original_model, original_tokenizer, args.original_texts)
    print("Translated Texts:", translated_texts)

    # Perform back-translation
    back_translated_texts = perform_back_translation(args.original_texts, original_model, original_tokenizer, back_translation_model, back_translation_tokenizer)
    print("Back-Translated Texts:", back_translated_texts)

if __name__ == "__main__":
    main()
