import random
import evaluate
import numpy as np
import tensorflow as tf
from functools import partial
from transformers import AutoTokenizer
from utils import constants
from utils.data import LabelEncodeDecode, PrepDataset
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score


class SelectiveSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='multiclass_true_positives', skip_zero=False, **kwargs):
        self.skip_zero = skip_zero
        super(SelectiveSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # tf.print(tf.shape(y_true), tf.shape(y_pred))
        # tf.print(tf.reduce_mean(tf.abs(y_true - tf.math.argmax(y_pred, axis=-1))))
        # tf.print('\n')
        # tf.print(y_true[0, :], tf.math.argmax(y_pred, axis=-1)[0, :])
        # Here we eliminate the last index because the last index is the end of sequence marker
        # By eliminating it we give credit for the actual word predicted
        # super().update_state(y_true[:, :-1], y_pred[:, :-1, :], sample_weight)

        if self.skip_zero:
            # Sometimes the token end up in such a way that some classes have a bunch of 0's
            # We need to weigh the samples as 0
            sample_weight = tf.cast(y_true > 0, dtype=y_pred.dtype)

        # Counting accuracy with the zeros mak
        super().update_state(y_true[:, :], y_pred[:, :, :], sample_weight)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def cleanup_predictions(predictions, references):
    if np.any(np.array(predictions) == -1):
        pred_ = []
        classes = set(references)
        for p, r in zip(predictions, references):
            if p == -1:
                # Pick any class other than the reference class
                pred_.append(list(classes.difference({r}))[random.randint(0, len(classes)-2)])
            else:
                # Concatenate prediction
                pred_.append(p)
        predictions = pred_

    return predictions


def evaluate_metric(logger, tag, dprep, checkpoint, model, val_split, is_fft, batch_size=100):
    """

    Args:
        logger: Logger object
        tag: Evaluation tag string
        dprep: Which dataset are we evaluating
        model: Model object
        checkpoint: Tokenizer to use
        val_split: Validation split
        dprep:
        batch_size: Size of batch to generate results
        is_fft: Is this a FFt run
    Returns:

    """

    # Get the tokenizer for this data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.replace('_-_', '/'),
                                              model_max_length=constants.ENCODER_MAX_LEN)
    tokenize = partial(PrepDataset.tokenize, tokenizer, True, is_fft)
    led = dprep.led

    if led.which[0] in ['super_glue', 'glue']:
        metric = evaluate.load(*led.which)
    else:
        raise ValueError(f'Unsupported metric type: {led.which}')

    # These are the questions we want answered
    # idx = [x['idx'] for x in val_split]
    # questions = [x['question'] for x in val_split]
    tokens = [tokenize(x).numpy().tolist()[0] for x in val_split]
    references = [x['label'] for x in val_split]

    #  These are the predictions from the model
    text_predictions = []
    for batch in chunks(tokens, batch_size):
        answers = model.generate(tf.stack(batch, axis=0), max_length=constants.DECODER_MAX_LEN+1)
        # answers = model(tf.stack(batch, axis=0), training=False)
        for answer in answers.numpy().tolist():
            text_predictions.append(tokenizer.decode(answer, skip_special_tokens=True))

    # The answers are text, now convert the answers back to labels
    predictions = [led[x] for x in text_predictions]

    # For the ones with an answer not in the answer set, replace with opposite of reference
    # This is to ensure we do not get credit for anything that is not real
    predictions = cleanup_predictions(predictions, references)

    # Log the prediction and the reference
    logger.info('reference,prediction,predict_text')
    for r, p, t in zip(references, predictions, text_predictions):
        logger.info(f'{r},{p},{t}')

    try:
        # Most metric follow this format
        results = metric.compute(predictions=predictions, references=references)
    except ValueError:
        # multirc follows a different format
        idx = [x['idx'] for x in val_split]
        # It needs  an index which is a dictionary and the value prediction
        dict_predictions = [{'idx': eval(x), 'prediction': y} for x, y in zip(idx, predictions)]
        results = metric.compute(predictions=dict_predictions, references=references)

    # Unique values
    classes = list(set(references))
    labels = [led(x) for x in classes]

    # Print out the confusion matrix
    logger.info('Confusion matrix')
    try:
        matrix = confusion_matrix(references, predictions)
        strng = "".join(f'{x:>15}' for x in [' ', ] + labels)
        logger.info(strng)
        for label, row in zip(labels, matrix):
            row_str = "".join(f'{x:15d}' for x in row)
            logger.info(f'{label:>15}{row_str}')
    except ValueError:
        logger.info('Bad predictions. Confusion matrix cannot be printed')

    # simple sklearn results
    acc = accuracy_score(references, predictions)
    bas = balanced_accuracy_score(references, predictions)
    abas = balanced_accuracy_score(references, predictions, adjusted=True)
    f1s = f1_score(references, predictions, average='weighted')
    logger.info(f'Results:{tag},acc,{acc},bas,{bas},abas,{abas},f1s,{f1s}')

    # Then the final results
    res_str = "".join(f'{k},{v},' for k, v in results.items())
    logger.info(f'Results_benchmark:{tag},{res_str[:-1]}')
    results.update({'bas': bas, 'abas': abas, 'f1s': f1s, 'acc': acc})
    return results
