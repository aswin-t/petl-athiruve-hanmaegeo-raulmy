import evaluate
from transformers import AutoTokenizer
from utils.data import LabelEncodeDecode


def evaluate_metric(which, checkpoint, model, val_split):
    """

    Args:
        which: Which dataset are we evaluating
        model: Model object
        checkpoint: Tokenizer to use
        val_split: Validation split

    Returns:

    """

    # Get the tokenizer for this data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=256)
    led = LabelEncodeDecode(which)

    if which[0] in ['super_glue', 'glue']:
        metric = evaluate.load(*which)
    else:
        raise ValueError(f'Unsupported metric type: {which}')

    # These are the questions we want answered
    idx = [x['idx'] for x in val_split]
    questions = [x['question'] for x in val_split]
    references = [x['label'] for x in val_split]
    answers = model.generate(tokenizer(questions, return_tensors="pt", padding=True).input_ids)

    # The answers are text, now convert the answers back to labels
    predictions = [led[x] for x in answers]
    results = metric.compute(predictions=predictions, references=references)

    print(results)
    return results
