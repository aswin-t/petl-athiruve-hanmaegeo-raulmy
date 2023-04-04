import os
import random

import numpy as np
from utils import constants
from utils.data import PrepDataset
from utils.log import create_logger
from transformers import AutoTokenizer


def check_tokenizer_lengths(checkpoint='t5-small'):
    """

    Args:
        checkpoint:

    Returns:

    """

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=constants.ENCODER_MAX_LEN)
    pairs = [('true', 'false'), ('positive', 'negative'),
             ('entailment', 'not_entailment'), ('entailment', 'neutral or contradiction'),
             ('unacceptable', 'acceptable'),
             ('equivalent', 'not_equivalent'), ('equivalent', 'different'),
             ('duplicate', 'not_duplicate'), ('duplicate', 'different'),
             ('entailment', 'not_entailment', 'contradiction'),
             ('absolutely positive', 'terribly negative'),
             ('absolute truth', 'terrible lie'),
             ('entailment', 'neutral', 'contradiction'),
             ('implies', 'neutral', 'contradiction'),
             ('follows', 'neutral', 'contradiction'),
             ('similar', 'different'),
             ('zero', 'one', 'two', 'three')
             ]

    for pair in pairs:
        tokens = tokenizer(list(pair), padding=False).input_ids
        print(tokens)
        lengths = [len(x) for x in tokens]
        print([[tokenizer.decode([x, ]) for x in y] for y in tokens])
        max_ = max(lengths)
        print(pair, max_, lengths)


def text_encode_and_save():
    output_path = os.path.join(os.path.dirname(__file__), "../mycheckpoints")
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    model_checkpoint = 't5-small'

    # Create a log object
    logger = create_logger(output_path, filename='data_text_encode.log')

    # Prepare the Dataset
    dprep = PrepDataset(logger=logger, checkpoint=model_checkpoint)

    for task in constants.Tasks()['glue'] + constants.Tasks()['super_glue']:
        for token_equalize in [True, False]:
            for is_fft in [True, False]:
                dprep.encode_and_save(task, cache_path=cache_path, token_equalize=token_equalize, is_fft=is_fft)


def compare_predictions():
    predictions = [1, -1, 3]
    references = [1, 2, 3]
    if np.any(np.array(predictions) == -1):
        pred_ = []
        classes = set(references)
        for p, r in zip(predictions, references):
            if p == -1:
                # Pick any class other than the reference class
                pred_.append(list(classes.difference({r}))[random.randint(0, len(classes)-2)])
            else:
                pred_.append(r)
        predictions = pred_

    print(predictions)
    return predictions


if __name__ == '__main__':
    # compare_predictions()
    check_tokenizer_lengths(checkpoint='google/t5-base-lm-adapt')
    # text_encode_and_save()
