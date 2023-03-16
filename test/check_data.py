from utils import constants
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
             ('absolute truth', 'terrible lie')
             ]

    for pair in pairs:
        tokens = tokenizer(list(pair), padding=False).input_ids
        print(tokens)
        lengths = [len(x) for x in tokens]
        max_ = max(lengths)
        print(pair, max_)


if __name__ == '__main__':
    check_tokenizer_lengths(checkpoint='t5-small')
