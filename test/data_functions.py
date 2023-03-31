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
    pairs = [('true', 'false'), ('positive', 'negative'), ('entailment', 'not_entailment'),
             ('unacceptable', 'acceptable'), ('equivalent', 'not_equivalent'),
             ('duplicate', 'not_duplicate'), ('entailment', 'not_entailment', 'contradiction'),
             ('absolutely positive', 'terribly negative'),
             ('absolute truth', 'terrible lie'),
             ('entailment', 'neutral', 'contradiction'),
             ('similar', 'different')
             ]

    for pair in pairs:
        tokens = tokenizer(list(pair), padding=False).input_ids
        print(tokens)
        lengths = [len(x) for x in tokens]
        max_ = max(lengths)
        print(pair, max_, lengths)


def text_encode_and_save():
    output_path = os.path.join(os.path.dirname(__file__), "../checkpoints")
    cache_path = os.path.join(os.path.dirname(__file__), "../cache")
    model_checkpoint = 't5-small'

    # Create a log object
    logger = create_logger(output_path, filename='data_text_encode.log')

    # Prepare the Dataset
    dprep = PrepDataset(logger=logger, checkpoint=model_checkpoint)

    # out = dprep.get(which=which_d, batch_size=100, cache_path=cp)
    # whiches = (('super_glue', 'boolq'), ('super_glue', 'rte'), ('super_glue', 'wic'), ('super_glue', 'wsc.fixed'),
    #            ('super_glue', 'record'),  ('super_glue', 'multirc'),
    #            ('super_glue', 'cb'), ('super_glue', 'copa'))
    whiches = (('glue', 'cola'), ('glue', 'mrpc'), ('glue', 'qnli'), ('glue', 'qqp'),
               ('glue', 'rte'), ('glue', 'sst2'), ('glue', 'stsb'), ('glue', 'wnli'))
    for wo in whiches:
        dprep.encode_and_save(wo, cache_path=cache_path)


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
