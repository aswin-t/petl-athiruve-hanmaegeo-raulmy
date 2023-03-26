import evaluate
import tensorflow as tf
from functools import partial
from transformers import AutoTokenizer
from utils import constants
from utils.data import LabelEncodeDecode, PrepDataset
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score


class SelectiveSparseTopKCategoricalAccuracy(tf.keras.metrics.SparseTopKCategoricalAccuracy):

    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(SelectiveSparseTopKCategoricalAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # tf.print(tf.shape(y_true), tf.shape(y_pred))
        # tf.print(tf.reduce_mean(tf.abs(y_true - tf.math.argmax(y_pred, axis=-1))))
        # tf.print(y_true[0, :], tf.math.argmax(y_pred, axis=-1)[0, :])
        # Here we eliminate the last index because the last index is the end of sequence marker
        # By eliminating it we give credit for the actual word predicted
        # super().update_state(y_true[:, :-1], y_pred[:, :-1, :], sample_weight)
        super().update_state(y_true[:, :], y_pred[:, :, :], sample_weight)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def evaluate_metric(logger, tag, which, checkpoint, model, val_split, batch_size=100):
    """

    Args:
        logger: Logger object
        tag: Evaluation tag string
        which: Which dataset are we evaluating
        model: Model object
        checkpoint: Tokenizer to use
        val_split: Validation split
        batch_size: Size of batch to generate results
    Returns:

    """

    # Get the tokenizer for this data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.replace('_-_', '/'),
                                              model_max_length=constants.ENCODER_MAX_LEN)
    tokenize = partial(PrepDataset.tokenize, tokenizer, True)
    led = LabelEncodeDecode(which)

    if which[0] in ['super_glue', 'glue']:
        metric = evaluate.load(*which)
    else:
        raise ValueError(f'Unsupported metric type: {which}')

    # These are the questions we want answered
    # idx = [x['idx'] for x in val_split]
    # questions = [x['question'] for x in val_split]
    tokens = [tokenize(x).numpy().tolist()[0] for x in val_split]
    references = [x['label'] for x in val_split]

    #  These are the predictions from the model
    text_predictions = []
    for batch in chunks(tokens, batch_size):
        answers = model.generate(tf.stack(batch, axis=0), max_length=constants.DECODER_MAX_LEN)
        # answers = model(tf.stack(batch, axis=0), training=False)
        for answer in answers.numpy().tolist():
            text_predictions.append(tokenizer.decode(answer, skip_special_tokens=True))

    # The answers are text, now convert the answers back to labels
    predictions = [led[x] for x in text_predictions]

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
