import evaluate
import tensorflow as tf
from functools import partial
from transformers import AutoTokenizer
from utils.constants import ENCODER_MAX_LEN
from utils.data import LabelEncodeDecode, PrepDataset


class SelectiveSparseTopKCategoricalAccuracy(tf.keras.metrics.SparseTopKCategoricalAccuracy):

    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(SelectiveSparseTopKCategoricalAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Here we eliminate the last index because the last index is the end of sequence marker
        # By eliminating it we give credit for teh actual word predicted
        super().update_state(y_true[:, :-1], y_pred[:, :-1, :], sample_weight)


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
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=ENCODER_MAX_LEN)
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
        answers = model.generate(tf.stack(batch, axis=0))
        for answer in answers.numpy().tolist():
            text_predictions.append(tokenizer.decode(answer, skip_special_tokens=True))

    # The answers are text, now convert the answers back to labels
    predictions = [led[x] for x in text_predictions]
    results = metric.compute(predictions=predictions, references=references)

    res_str = "".join(f'{k},{v}' for k, v in results.items())
    logger.info(f'Results:{tag},{res_str}')
    return results
