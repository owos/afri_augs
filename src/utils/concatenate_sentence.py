"""
Author: Oluwatosin Olajide
This is a script that implement sentence concantenation as described in Sentence Concatenation 
Approach to Data Augmentation for Neural Machine Translation by Kondo et al 
(https://aclanthology.org/2021.naacl-srw.18)
Other References:
Improving Neural Machine Translation Models with Monolingual Data by Sennrich et al
(https://aclanthology.org/P16-1009)
"""
import math

# pylint: disable=fixme,too-many-arguments,too-many-locals
import random
from typing import List, Tuple

from datasets import load_dataset
from transformers import pipeline


def back_translate(
    origin_target_data: list,
    model: str,
    sample_percent: float,
) -> List[str]:
    """
    Function to back translate a list of sentences using a model from huggingface.

    Args:
        origin_target_data (list): Data to be back translated.
        model (str): Model to use for back translation.
        sample_percent (float): percent of data to back_tranlate.

    Returns:
        List[str]: List of back translated sentences.
    """
    number_of_samples = math.ceil(sample_percent * len(origin_target_data))
    target_sample_indexes = random.sample(
        range(len(origin_target_data)), number_of_samples
    )
    target_samples = [origin_target_data[index] for index in target_sample_indexes]
    translator = pipeline("text2text-generation", model=model)
    translated_source_data = [result["generated_text"] for result in translator(target_samples)]
    return translated_source_data, target_sample_indexes


def concantenate_sentences(
    subset_name: str,
    seperator_token: str = "[SEP]",
    sentence_length_threshold: int = 25,
    sequential: bool = False,
    do_back_translation: bool = True,
    number_of_concatenations: int = 100,
    back_translation_model: str = None,
    back_translation_percent: float = None
) -> Tuple[List[str], List[str]]:
    """
    Function to concantenate sentences from a source and target file as described
    by Kondo et al in Sentence Concatenation Approach to Data Augmentation for Neural.

    Args:
        subset_name (str): Name of Mafand subset.
        seperator_token (str, optional): Token to use as separator token. Defaults to
        "[SEP]".
        sentence_length_threshold (int, optional): Threshold required for length of
        sentences from the pseudo data to make it to the final dataset. Defaults to 25.
        sequential (bool, optional): Concatenates the sentences sequentially if True
        else it does it randomly. Defaults to False.
        do_back_translation (bool, optional): Whether to back-translate or not. Defaults
        to True.
        number_of_concatenations (int, optional): Number of concatenation to do, this
        parameter is only needed if `sequential` is False, if `do_back_translation` 
        is True. Defaults to 100.
        back_translation_model (str, optional): name of model to use for back translations.
        This parameter is only needed if `do_back_translation` is True. Defaults to None.
        back_translation_percent (float, optional): percent of target data to back translate.
        Should take any value between 0 and 1. This parameter is only needed if 
        `do_back_translation` is True. Defaults to None.

    Returns:
        Tuple[List[str], List[str]]: Returns of a tuple of the source and target sentences
    """
    # load data
    data = load_dataset("masakhane/mafand", subset_name)
    source_lang = subset_name.split("-")[0]
    target_lang = subset_name.split("-")[1]
    origin_source_data = [pair[source_lang] for pair in data["train"]["translation"]]
    origin_target_data = [pair[target_lang] for pair in data["train"]["translation"]]
    print(len(origin_source_data))

    # if back translate is true, back translate the origin data else
    # set the pseudo data to be the same as the origin data
    if do_back_translation:
        pseudo_source_data, index_translated = back_translate(
            origin_target_data,
            back_translation_model,
            back_translation_percent,
        )
        pseudo_target_data = [origin_target_data[index] for index in index_translated]
    else:
        pseudo_source_data = origin_source_data
        pseudo_target_data = origin_target_data

    # If sequential, concantenate all sentences in the source data with the sentence at
    # the next index in the pseudo data. If it's not sequential then randomly select
    # sentences to concatenate
    if sequential:
        origin_indexes = list(range(len(origin_source_data)))
        pseudo_indexes = list(range(1, len(pseudo_source_data)))
    else:
        origin_indexes = random.sample(
            range(len(origin_source_data)), number_of_concatenations
        )
        pseudo_indexes = random.sample(
            range(len(pseudo_source_data)), number_of_concatenations
        )

    concantenated_source = []
    concantenated_target = []

    for origin_index, pseudo_index in list(zip(origin_indexes, pseudo_indexes)):
        new_source = (
            origin_source_data[origin_index]
            + f" {seperator_token} "
            + pseudo_source_data[pseudo_index]
        )
        # check if the length of the concatenated sentence meets the threshold
        if len(new_source) >= sentence_length_threshold:
            concantenated_source.append(new_source)
            concantenated_target.append(
                origin_target_data[origin_index]
                + f" {seperator_token} "
                + pseudo_target_data[pseudo_index]
            )
    new_source_data = origin_source_data + concantenated_source
    new_target_data = origin_target_data + concantenated_target

    return new_source_data, new_target_data


if __name__ == "__main__":
    # This the name of the subset of the dataset to use
    # in the case of mafand, it's the language pair
    SUBSET = "en-yor"
    # separator token. This should ideally be the separator token used by
    # the model you would train with the output data, not the back translation model.
    SEPARATOR_TOKEN = "</s>"
    SENTENCE_LENGTH_THRESHOLD = 10
    SEQUENTIAL = False
    DO_BACK_TRANSLATION = True
    # In a case where you are doing back translation, this parameter should be less
    # that back_translation_percent * <length of your data>, in this case 33.22(6644 * 0.005)
    NUMBER_OF_CONCATENATION = 10
    BACK_TRANSLATION_MODEL = "masakhane/mbart50_yor_en_news"
    BACK_TRANSLATION_PERCENT = 0.005

    source_data, target_data = concantenate_sentences(
        SUBSET,
        SEPARATOR_TOKEN,
        SENTENCE_LENGTH_THRESHOLD,
        SEQUENTIAL,
        DO_BACK_TRANSLATION,
        NUMBER_OF_CONCATENATION,
        BACK_TRANSLATION_MODEL,
        BACK_TRANSLATION_PERCENT
    )
    print(len(source_data))
