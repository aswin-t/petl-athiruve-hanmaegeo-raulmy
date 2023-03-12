import torch
import evaluate
import tensorflow as tf
from functools import partial
from constants import ENCODER_MAX_LEN
from transformers import AutoTokenizer
from utils.data import LabelEncodeDecode, PrepDataset


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def evaluate_metric(which, checkpoint, model, val_split, batch_size=100):
    """

    Args:
        which: Which dataset are we evaluating
        model: Model object
        checkpoint: Tokenizer to use
        val_split: Validation split
        batch_size: Size of batch to generate results

    Returns:

    """

    # Get the tokenizer for this data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=ENCODER_MAX_LEN)
    tokenize = partial(PrepDataset.tokenize, tokenizer, True)
    led = LabelEncodeDecode(which)

    if which[0] in ['super_glue', 'glue']:
        metric = evaluate.load(*which)
    else:
        raise ValueError(f'Unsupported metric type: {which}')

    # These are the questions we want answered
    idx = [x['idx'] for x in val_split]
    questions = [x['question'] for x in val_split]
    tokens = [tokenize(x) for x in val_split]
    references = [x['label'] for x in val_split]

    #  These are the predictions from the model
    text_predictions = []
    for batch in chunks(tokens, batch_size):
        answers = model.generate(tf.concat(batch, axis=0))
        for answer in answers:
            text_predictions.append(tokenizer.decode(answer, skip_special_tokens=True))

    # The answers are text, now convert the answers back to labels
    predictions = [led[x] for x in text_predictions]
    results = metric.compute(predictions=predictions, references=references)

    print(results)
    return results
