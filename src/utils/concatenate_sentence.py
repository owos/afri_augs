"""
Author: Oluwatosin Olajide

This is a script that implement sentence concantenation as described in Sentence Concatenation 
Approach to Data Augmentation for Neural Machine Translation by Kondo et al 
(https://aclanthology.org/2021.naacl-srw.18)

Other References:
Improving Neural Machine Translation Models with Monolingual Data by Sennrich et al
(https://aclanthology.org/P16-1009)
"""
import argparse
import logging
import math

# pylint: disable=too-many-arguments,too-many-locals
import os
import random
from typing import List, Tuple

from datasets import Dataset, load_dataset
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("sentencer concatenator")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEVICE = "mps"


def back_translate(
    origin_target_data: list,
    model: str,
    sample_percent: float,
) -> Tuple[List[str], List[int]]:
    """
    Function to back translate a list of sentences using a model from huggingface.

    Args:
        origin_target_data (list): Data to be back translated.
        model (str): Model to use for back translation.
        sample_percent (float): percent of data to back_tranlate.

    Returns:
        Tuple[List[str], List[int]]: Dataset of back translated sentences.
    """
    number_of_samples = math.ceil(sample_percent * len(origin_target_data))
    target_sample_indexes = random.sample(
        range(len(origin_target_data)), number_of_samples
    )
    target_samples = [origin_target_data[index] for index in target_sample_indexes]
    translator = pipeline("text2text-generation", model=model, device=DEVICE)
    translated_source_data = [
        result["generated_text"] for result in translator(target_samples)
    ]
    return translated_source_data, target_sample_indexes


def concantenate_sentences(
    subset_name: str,
    seperator_token: str = "[SEP]",
    sentence_length_threshold: int = 25,
    sequential: bool = False,
    do_back_translation: bool = True,
    number_of_concatenations: int = 100,
    back_translation_model: str = None,
    back_translation_percent: float = None,
    dataset: str = "masakhane/mafand",
) -> Dataset:
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
        dataset (str, optional): Dataset to augment. Defaults to `masakhane/mafand`.

    Returns:
        Dataset: Returns of a tuple of the source and target sentences

    Example Usage:
        ```
        # This the name of the subset of the dataset to use in the case of mafand, it's the
        # language pair
        subset = "en-yor"
        # separator token should ideally be the separator token used by
        # the model you would train with the output data, not the back translation model.
        separator_token = "</s>"
        sentence_length_threshold = 10
        sequential = False
        do_back_translation = True
        # In a case where you are doing back translation, this parameter should be less
        # that back_translation_percent * <length of your data>, in this case 33.22(6644 * 0.005)
        number_of_concatenation = 10
        back_translation_model = "masakhane/mbart50_yor_en_news"
        back_translation_percent = 0.005
        dataset = "masakhane/mafand"

        source_data, target_data = concantenate_sentences(
            subset,
            separator_token,
            sentence_length_threshold,
            SEQUENTIAL,
            do_back_translation,
            number_of_concatenation,
            back_translation_model,
            back_translation_percent,
            dataset
        )
        ```
    """
    # load data
    data = load_dataset(dataset, subset_name, split="train")
    logger.info("Dataset loaded")
    source_lang = subset_name.split("-")[0]
    target_lang = subset_name.split("-")[1]
    origin_source_data = [pair[source_lang] for pair in data["translation"]]
    origin_target_data = [pair[target_lang] for pair in data["translation"]]
    logger.info("Source and target data extracted.")

    # if back translate is true, back translate the origin data else
    # set the pseudo data to be the same as the origin data
    if do_back_translation:
        logger.info("Doing back translation")
        pseudo_source_data, index_translated = back_translate(
            origin_target_data,
            back_translation_model,
            back_translation_percent,
        )
        pseudo_target_data = [origin_target_data[index] for index in index_translated]
    else:
        logger.info("No back translation")
        pseudo_source_data = origin_source_data
        pseudo_target_data = origin_target_data

    # If sequential, concantenate all sentences in the source data with the sentence at
    # the next index in the pseudo data. If it's not sequential then randomly select
    # sentences to concatenate
    if sequential:
        logger.info("Getting sequential indexes to be concatenated.")
        origin_indexes = list(range(len(origin_source_data)))
        pseudo_indexes = list(range(1, len(pseudo_source_data)))
    else:
        logger.info("Getting random indexes to be concatenated.")
        origin_indexes = random.sample(
            range(len(origin_source_data)), number_of_concatenations
        )
        pseudo_indexes = random.sample(
            range(len(pseudo_source_data)), number_of_concatenations
        )

    concantenated_source = []
    concantenated_target = []

    logger.info("Concatenating sentences")
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
        else:
            logger.info("Dropped : %s", new_source)

    logger.info("Putting augmented data into huggingface dataset.")
    for source, target in list(zip(concantenated_source, concantenated_target)):
        data = data.add_item({"translation": {"en": source, "yor": target}})

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for sentence concatenation")
    parser.add_argument(
        "--subset_name",
        type=str,
        help="Name of the dataset subset to concatenate sentences from",
        required=True,
    )
    parser.add_argument(
        "--seperator_token",
        type=str,
        default="[SEP]",
        help="Token used to separate concatenated sentences",
    )
    parser.add_argument(
        "--sentence_length_threshold",
        type=int,
        default=25,
        help="Threshold for sentence length to be considered for concatenation",
    )
    parser.add_argument(
        "--sequential",
        type=bool,
        help="Flag to indicate whether sentences should be concatenated sequentially",
    )
    parser.add_argument(
        "--do_back_translation",
        type=bool,
        help="Flag to indicate whether to perform back translation",
    )
    parser.add_argument(
        "--number_of_concatenations",
        type=int,
        default=100,
        help="Number of sentence concatenations to perform",
    )
    parser.add_argument(
        "--back_translation_model",
        type=str,
        help="Model to use for back translation",
    )
    parser.add_argument(
        "--back_translation_percent",
        type=float,
        help="Percentage of sentences to back translate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="masakhane/mafand",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--destination_filepath",
        type=str,
        help="File path to directory to save data to.",
    )

    args = parser.parse_args()
    augmented_data = concantenate_sentences(
        args.subset_name,
        args.seperator_token,
        args.sentence_length_threshold,
        args.sequential,
        args.do_back_translation,
        args.number_of_concatenations,
        args.back_translation_model,
        args.back_translation_percent,
        args.dataset,
    )
    file_path = (
        f"{args.destination_filepath}/{args.subset_name}"
        f"/sentence_concat_{int(args.back_translation_percent * 100)}.json"
    )
    logger.info("Saving data to %s", file_path)
    augmented_data.to_json(
        file_path,
        orient="records",
    )
    logger.info("Adios!")
