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
    pairs = [('entailment', 'not_entailment'),
             ('true', 'false'), ('positive', 'negative'),
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


def histo_normalize(token_counts: dict):

    # Count the frequency of tokens
    independent_counts = {}
    for tokens, count in token_counts.items():
        for token in tokens:
            try:
                independent_counts[token] += 1
            except KeyError:
                independent_counts[token] = 1

    # More frequent tokens get a lower weight
    independent_counts = {k: 1/v for k, v in independent_counts.items()}

    # Token mapping
    mapped_weights = {}
    for tokens in token_counts.keys():
        out = []
        for token in tokens:
            out.append(independent_counts[token])
        mapped_weights[tokens] = out

    max_counts = max(list(token_counts.values()))
    # Now normalize by class
    for tokens, weights in mapped_weights.items():
        mapped_weights[tokens] = [x * max_counts/token_counts[tokens] for x in weights]

    return mapped_weights


if __name__ == '__main__':
    histo_normalize(token_counts={(7532, 4, 12, 7): 2249, (5756, 0, 0, 0, 0): 1149, (412, 4, 12, 7): 2249})
    # compare_predictions()
    # check_tokenizer_lengths(checkpoint='google/t5-base-lm-adapt')
    # text_encode_and_save()
