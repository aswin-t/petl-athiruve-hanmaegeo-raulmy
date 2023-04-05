import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Union
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from utils import constants


class LabelEncodeDecode:

    def __init__(self, which, do_equal=False):
        self.which = which
        self.do_equal = do_equal
        self.has_versions = False
        if which[0] == 'super_glue':
            if which[1] in ['axb', 'axg', 'rte']:
                if not self.do_equal:
                    self.lookup = {0: 'entailment', 1: 'not_entailment', -1: 'test'}
                    # Entailment is 4, not_entailment is 5
                    constants.DECODER_MAX_LEN = 6
                else:
                    # self.lookup = {0: 'entailment', 1: 'neutral or contradiction', -1: 'test'}
                    # Entailment is 4, neutral or contradiction is 3
                    # constants.DECODER_MAX_LEN = 5
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            elif which[1] in ['boolq', 'wic', 'wsc.fixed', 'multirc']:
                if not self.do_equal:
                    self.lookup = {0: 'false', 1: 'true', -1: 'test'}
                    # True and False are both 1
                    constants.DECODER_MAX_LEN = 2
                    self.has_versions = False
                else:
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            elif which[1] == 'cb':
                if not self.do_equal:
                    self.lookup = {0: 'entailment', 1: 'contradiction', 2: 'neutral', -1: 'test'}
                    # Entailment is 4, contradiction is 1 and neutral is 1
                    constants.DECODER_MAX_LEN = 5
                else:
                    # Make the three the same length
                    # self.lookup = {0: 'implies', 1: 'contradiction', 2: 'neutral', -1: 'test'}
                    # Implies is 1, contradiction is 1 and neutral is 1
                    # constants.DECODER_MAX_LEN = 2
                    self.lookup = {0: 'zero', 1: 'one', 2: 'two', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            elif which[1] == 'copa':
                if not self.do_equal:
                    self.lookup = {0: 'choice1', 1: 'choice2', -1: 'test'}
                    # choice1 and choice2  are 2 tokens long
                    constants.DECODER_MAX_LEN = 3
                else:
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            else:
                self.lookup = {}
                # This is a longer answer and requires 50
                constants.DECODER_MAX_LEN = 20
                self.has_versions = False
        elif which[0] == 'glue':
            if which[1] in ['qnli', 'rte']:
                if not self.do_equal:
                    self.lookup = {0: 'entailment', 1: 'not_entailment', -1: 'test'}
                    # Entailment is 4, not_entailment is 5
                    constants.DECODER_MAX_LEN = 6
                else:
                    # self.lookup = {0: 'entailment', 1: 'neutral or contradiction', -1: 'test'}
                    # Entailment is 4, neutral or contradiction is 3
                    # constants.DECODER_MAX_LEN = 5
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            elif which[1] in ['wnli', ]:
                if not self.do_equal:
                    self.lookup = {0: 'not_entailment', 1: 'entailment', -1: 'test'}
                    # Entailment is 4, not_entailment is 5
                    constants.DECODER_MAX_LEN = 6
                else:
                    # self.lookup = {0: 'neutral or contradiction', 1: 'entailment', -1: 'test'}
                    # Entailment is 4, neutral or contradiction is 3
                    # constants.DECODER_MAX_LEN = 5
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            elif which[1] in ['sst2', ]:
                if not self.do_equal:
                    self.lookup = {0: 'negative', 1: 'positive', -1: 'test'}
                    # Negative and positive are both 1
                    constants.DECODER_MAX_LEN = 2
                else:
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2

                self.has_versions = True
            elif which[1] in ['cola', ]:
                if not self.do_equal:
                    self.lookup = {0: 'unacceptable', 1: 'acceptable', -1: 'test'}
                    # Acceptable and unacceptable are both 1
                    constants.DECODER_MAX_LEN = 2
                else:
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2

                self.has_versions = True
            elif which[1] in ['mrpc', ]:
                if not self.do_equal:
                    self.lookup = {0: 'equivalent', 1: 'not_equivalent', -1: 'test'}
                    # Equivalent is 1, not_equivalent is 5
                    constants.DECODER_MAX_LEN = 6
                else:
                    # self.lookup = {0: 'equivalent', 1: 'different', -1: 'test'}
                    # Equivalent and different are 1
                    # constants.DECODER_MAX_LEN = 2
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            elif which[1] in ['qqp', ]:
                if not self.do_equal:
                    self.lookup = {0: 'not_duplicate', 1: 'duplicate', -1: 'test'}
                    # Duplicate is 1 and not duplicate is 5
                    constants.DECODER_MAX_LEN = 6
                else:
                    # self.lookup = {0: 'different', 1: 'duplicate', -1: 'test'}
                    # Duplicate and different are 1
                    # constants.DECODER_MAX_LEN = 2
                    self.lookup = {0: 'zero', 1: 'one', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            elif which[1] in ['mnli', ]:
                if not self.do_equal:
                    self.lookup = {0: 'entailment', 1: 'neutral', 2: 'contradiction', -1: 'test'}
                    # entailment is 4, contradiction is 1 and neutral is 1
                    constants.DECODER_MAX_LEN = 5
                else:
                    # self.lookup = {0: 'implies', 1: 'neutral', 2: 'contradiction', -1: 'test'}
                    # implies is 1, contradiction is 1 and neutral is 1
                    # constants.DECODER_MAX_LEN = 2
                    self.lookup = {0: 'zero', 1: 'one', 2: 'two', -1: 'test'}
                    constants.DECODER_MAX_LEN = 2
                self.has_versions = True
            else:
                self.lookup = {}
                constants.DECODER_MAX_LEN = 50
                self.has_versions = False
        else:
            self.lookup = {}
            # This is a longer answer and requires 50
            constants.DECODER_MAX_LEN = 50

    def __call__(self, inlabel, *args, **kwargs):
        return str(self.from_label(inlabel))

    def __getitem__(self, item):
        return self.to_label(item)

    def get_tag(self):
        """

        Returns:

        """
        tag = "".join(f'{x}-' for x in self.which)
        # Also add the answer format
        if self.has_versions:
            tag += 'equal-' if self.do_equal else 'unequal-'

        return tag[:-1]

    def from_label(self, inlabel):

        if self.lookup:
            # There is a translation that can be done
            return self.lookup[inlabel]
        else:
            # There is no translation involved
            return str(inlabel)

    def to_label(self, inpred):

        if self.lookup:
            for k, v in self.lookup.items():
                # Found the label value
                if v == inpred:
                    return k

            # Did not find the predicted values in the value field
            return -1
        else:
            # There is no translation involved
            return inpred


def _cleanup_str(strng):
    strng = strng.rstrip()
    if strng[-1] != '.':
        strng += '.'

    return strng


class PrepDataset:

    def __init__(self, logger, checkpoint: str, encoder_max_len=constants.ENCODER_MAX_LEN,
                 decoder_max_len=constants.DECODER_MAX_LEN, num_proc=1):
        """

        Args:
            logger: Object of python logging class
            checkpoint: Checkpoint from which to load the model
            encoder_max_len: Maximum token length for encoder
            decoder_max_len: Maximum length for decoder
        """

        self.logger = logger
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint.replace('_-_', '/'),
                                                       model_max_length=self.encoder_max_len)
        self.supported_type = [('squad',), ('super_glue', 'axb')]

        self.train_dataset = None
        self.valid_dataset = None
        self.num_proc = num_proc
        self.words = None
        self.counts = None
        self.led = None

    def __getstate__(self):
        state = self.__dict__
        del state['tokenizer']

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, model_max_length=self.encoder_max_len)

    @staticmethod
    def _to_tf_dataset(dataset):
        """

        Args:
            dataset: Hugging Face dataset

        Returns:

        """
        columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
        dataset.set_format(type='tensorflow', columns=columns)
        return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32,
                        'labels': tf.int32, 'decoder_attention_mask': tf.int32, }
        return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
                         'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}
        ds = tf.data.Dataset.from_generator(lambda: dataset, return_types, return_shapes)
        return ds

    @staticmethod
    def encode_squad(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:

        """

        if led:
            pass

        # Context for answering the question
        context = example['context']

        # The question
        question = example['question']

        # Answer for the question, required for training
        answer = example['answers']['text']

        # Adding prompt
        question_plus = f"answer_me: {str(question)}"
        question_plus += f" context: {str(context)}"

        # This is the answer we are trying to generate
        answer_plus = ', '.join([i for i in list(answer)])
        answer_plus = f"{answer_plus} "

        outputs = {'question': question_plus, 'answer': answer_plus}
        return outputs

    @staticmethod
    def encode_super_glue(led, add_taskname, example):
        """

        Args:
            led:
            add_taskname: Add the taskname to the task
            example:

        Returns:
        """
        if led.which[1] == 'axb':
            # Context for answering the question
            first = example['sentence1']
            second = example['sentence2']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            first = _cleanup_str(first)
            second = _cleanup_str(second)
            question_plus = f"sentence1: {first} sentence2: {second}"
            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus
            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in ['axg', 'cb', 'rte']:
            # Context for answering the question
            premise = example['premise']
            hypothesis = example['hypothesis']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            hypothesis = _cleanup_str(hypothesis)
            premise = _cleanup_str(premise)
            question_plus = f"hypothesis: {hypothesis} premise: {premise}"

            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'boolq':
            # Context for answering the question
            question = example['question']
            passage = example['passage']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question = _cleanup_str(question)
            passage = _cleanup_str(passage)
            question_plus = f" passage: {passage} question: {question}"
            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus
            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'copa':
            # Context for answering the question
            question = example['question']
            c1 = example['choice1']
            c2 = example['choice2']
            premise = example['premise']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question = _cleanup_str(question)
            premise = _cleanup_str(premise)
            question_plus = f"choice1: {c1} choice2: {c2} "
            question_plus += f"premise: {premise} question: {question}"

            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'multirc':
            # Context for answering the question
            question = example['question']
            para = example['paragraph']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question = _cleanup_str(question)
            para = _cleanup_str(para)
            question_plus = f"question: {question} paragraph: {para}"

            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'record':
            # Context for answering the question
            passage = example['passage']

            # Does replacing this with extra id work better?
            # Do not know but it might be worth trying
            query = example['query']
            # Convert integer to text
            answer = ', '.join([i for i in list(example['answers'])])

            # Adding prompt
            question_plus = f"query: {query} "
            question_plus += f"passage: {passage}"

            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'rte':
            # Context for answering the question
            premise = example['premise']
            hypothesis = example['hypothesis']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            premise = _cleanup_str(premise)
            hypothesis = _cleanup_str(hypothesis)
            question_plus = f"hypothesis: {hypothesis} premise: {premise}"
            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'wic':
            # Context for answering the question
            sen1 = example['sentence1']
            sen2 = example['sentence2']
            word = example['word']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            sen1 = _cleanup_str(sen1)
            sen2 = _cleanup_str(sen2)
            question_plus = f"sentence1: {sen1} sentence2: {sen2} word: {word}"
            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus
            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'wsc.fixed':
            # Context for answering the question
            para = example['text']
            span1 = example['span1_text']
            span2 = example['span2_text']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            para = para.replace(span1, f'*{span1}*')
            para = para.replace(span2, f'*{span2}*')
            para = _cleanup_str(para)
            question_plus = f"paragraph: {para}"

            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        else:
            raise KeyError('Unknown datatype for translation')

    @staticmethod
    def encode_glue(led, add_taskname, example):
        """

        Args:
            led:
            add_taskname: Add the taskname to the task
            example:

        Returns:
        """
        if led.which[1] in ['cola', 'sst2']:
            sentence = example['sentence']

            sentence = _cleanup_str(sentence)
            question_plus = f"sentence: {sentence}"

            # Convert the numeric label to text
            answer = led(example['label'])

            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'mnli':
            # Context for answering the question
            premise = example['premise']
            hypothesis = example['hypothesis']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            hypothesis = _cleanup_str(hypothesis)
            premise = _cleanup_str(premise)
            question_plus = f"hypothesis: {hypothesis} premise: {premise}"
            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in ['mrpc', 'rte', 'stsb', 'wnli']:
            # Context for answering the question
            sen1 = example['sentence1']
            sen2 = example['sentence2']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            sen1 = _cleanup_str(sen1)
            sen2 = _cleanup_str(sen2)
            question_plus = f"sentence1: {sen1} sentence2: {sen2}"
            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in 'qnli':
            # Context for answering the question
            question = example['question']
            sentence = example['sentence']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            sentence = _cleanup_str(sentence)
            question_plus = f"sentence: {sentence} "
            question_plus += f"question: {question}"

            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in 'qqp':
            # Context for answering the question
            q1 = example['question1']
            q2 = example['question2']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"question1: {q1} question2: {q2}"
            if add_taskname:
                question_plus = f'{led.which[1]} {question_plus}'
            else:
                question_plus = 'absence ' * constants.NUM_SOFT_TOKENS + question_plus

            outputs = {'question': question_plus, 'answer': answer}
            return outputs
        else:
            raise KeyError('Unknown datatype for translation')

    def _get_encode(self, which, add_taskname, do_equal=False):
        """

        Args:
            which: Which dataset
            add_taskname: add the taskname to the task
            do_equal: Make task token lengths equal?
        Returns:

        """

        # Create a label encoder decoder for converting from labels to text and back
        led = LabelEncodeDecode(which, do_equal=do_equal)

        if which[0] == 'super_glue':
            return partial(self.encode_super_glue, led, add_taskname), led
        elif which[0] == 'glue':
            return partial(self.encode_glue, led, add_taskname), led
        elif which[0] == 'squad':
            return self.encode_squad, led
        else:
            raise KeyError(f'Unsupported dataset {which[0]}')

    @staticmethod
    def create_dataset(dataset, batch_size=4, buffer_size=1000, shuffling=True, cache_path=None):
        """

        Args:
            dataset: Dataset to convert
            batch_size:
            buffer_size:
            shuffling:
            cache_path:

        Returns:

        """
        if cache_path is not None:
            dataset.cache(cache_path)

        if shuffling:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(batch_size * 10)

        return dataset

    @staticmethod
    def _shorten_to_length(tokenizer, example):
        """

        Args:
            tokenizer: T5 tokenizer instance
            example: Example to process

        Returns:

        """

        # This is the max length we would like to tokenize
        max_length = constants.ENCODER_MAX_LEN - constants.NUM_SOFT_TOKENS

        # This is the number of tokens it will produce without restrictions
        tokens = tokenizer(example['question'])
        len_tokens = len(tokens['input_ids'])
        if len_tokens <= max_length:
            return {'question': example['question']}

        # Each word is at least one token so start with as many words
        new_example = example['question'].split(' ')[:max_length]

        while len_tokens > max_length:
            # Join the example back with space and remove the extra space that was added
            new_example = ''.join(f"{x} " for x in new_example)[:-1]
            tokens = tokenizer(new_example)
            len_tokens = len(tokens['input_ids'])

            if len_tokens > max_length:
                new_example = new_example.split(' ')[:-1]

        return {'question': new_example}

    @staticmethod
    def tokenize(tokenizer, is_test, is_fft, token_counter, example):
        """
        Our objective is not only to tokenize the input but also to ensure that the after soft prompt is included, the
        embeddings that are removed are all paddings only

        Args:
            is_fft:
            tokenizer: Instance of Autotokenizer initialized appropriately for the checkpoint
            is_test: Whether this is a test set we are working with
            example: Example to be tokenized
            token_counter: List for counting tokens

        Returns:
        """
        text = example['question'].rstrip()

        # Now encode the tokens, this time we can be sure that there are at least NUM_SOFT_TOKENS worth of paddings
        encode_length = constants.ENCODER_MAX_LEN if is_fft else constants.ENCODER_MAX_LEN + constants.NUM_SOFT_TOKENS
        encoder_inputs = tokenizer(text, truncation=True, max_length=encode_length, padding="max_length",
                                   return_tensors="tf")

        if encoder_inputs['input_ids'][0][-1] != 0:
            truncated = True
        else:
            truncated = False

        if is_test:
            return encoder_inputs['input_ids']

        if not isinstance(example['answer'], str):
            tmp = str(example['answer']).lower()
        else:
            tmp = str(example['answer'])
        decoder_inputs = tokenizer(tmp, truncation=True, max_length=constants.DECODER_MAX_LEN)['input_ids']

        # Set up to return
        input_ids = encoder_inputs['input_ids'][0]
        input_attention = encoder_inputs['attention_mask'][0]

        target_ids = decoder_inputs[:-1] + [0, ] * (constants.DECODER_MAX_LEN - len(decoder_inputs))

        try:
            token_counter[tuple(target_ids)] += 1
        except KeyError:
            token_counter[tuple(target_ids)] = 1

        target_attention = [1, ] * len(target_ids)
        # target_ids = tf.convert_to_tensor(target_ids, dtype=encoder_inputs['input_ids'].dtype)
        # target_attention = tf.convert_to_tensor(target_attention, dtype=encoder_inputs['input_ids'].dtype)

        return {'input_ids': input_ids, 'attention_mask': input_attention, 'labels': target_ids,
                'decoder_attention_mask': target_attention, 'truncated': truncated}

    def encode_and_save(self, which: Union[str, tuple] = 'squad', cache_path: str = None, is_fft: bool = False,
                        token_equalize: bool = False):
        """

        Args:
            which: Which dataset is it processing. ex; ('glue', 'cola')
            cache_path: Path for the cache file
            is_fft: Is this full fine tuning or sot prompt
            token_equalize: Make token length sequal for the answer?

        Returns:

        """

        if not isinstance(which, tuple):
            which = (which,)

        self.logger.info(f'Encoding dataset {which}')

        # Encode it into a question answer format
        # It also set the value for teh DECODE_MAX_LENGTH based on the answer type
        encoder, led = self._get_encode(which, add_taskname=is_fft, do_equal=token_equalize)
        self.logger.info(f'Using function {encoder} with answer lookup {led.lookup}')

        # The foldername name will be  the task name
        foldername = ''.join(f'{x}-' for x in which)
        # Also add the answer format
        if led.has_versions:
            foldername += 'equal' if led.do_equal else 'unequal'

        sub_folder = 'processed/not_fft' if not is_fft else 'processed/is_fft'
        processed_save_path = os.path.join(cache_path, sub_folder, foldername)
        os.makedirs(processed_save_path, exist_ok=True)

        if not os.path.exists(os.path.join(processed_save_path, "val.csv")):

            # Now we have the text encoded to tokens
            train_dataset = load_dataset(*which, split='train', cache_dir=cache_path)
            try:
                valid_dataset = load_dataset(*which, split='validation', cache_dir=cache_path)

            # For  Glue, MNLI the validaiton set is called validation_matched or validation_mismatched
            # We load the validation mismatched here
            except ValueError:
                valid_dataset = load_dataset(*which, split='validation_mismatched', cache_dir=cache_path)

            # Convert it into a question answer format
            remove_columns = [x for x in train_dataset.column_names if x not in ['id', 'idx', 'label']]
            train_dataset = train_dataset.map(encoder, remove_columns=remove_columns, num_proc=self.num_proc)
            valid_dataset = valid_dataset.map(encoder, remove_columns=remove_columns, num_proc=self.num_proc)

            self.logger.info(f'Data sample for train: {train_dataset[0]}')
            self.logger.info(f'Data sample for validation: {valid_dataset[0]}')

            train_dataset.to_csv(os.path.join(processed_save_path, f"train.csv"))
            valid_dataset.to_csv(os.path.join(processed_save_path, f"val.csv"))
        else:
            self.logger.info(f'Decoding was not performed as cache was found')

        # This is the completely unseen final test
        if not os.path.exists(os.path.join(processed_save_path, f"ftest.csv")):
            try:
                test_dataset = load_dataset(*which, split='test', cache_dir=cache_path)
            except ValueError:
                test_dataset = load_dataset(*which, split='test_mismatched', cache_dir=cache_path)

            # Remove the columns we do not need
            remove_columns = [x for x in test_dataset.column_names if x not in ['id', 'idx', 'label']]
            test_dataset = test_dataset.map(encoder, remove_columns=remove_columns, num_proc=self.num_proc)
            test_dataset.to_csv(os.path.join(processed_save_path, "ftest.csv"))

        return led, processed_save_path

    def count_words(self, filepath):

        df = pd.read_csv(filepath)
        q_list = df.question.to_list()
        out_list = []
        for q in q_list:
            out_list += q.split()

        words, counts = np.unique(out_list, return_counts=True)
        idx = np.argsort(counts)[-1::-1]

        # Words and counts in descending order
        words = words[idx]
        counts = counts[idx]

        # remove absence if it is the most common word
        if words[0] == 'absence':
            words = words[1:]
            counts = counts[1:]

        # Convert counts to probability
        counts = counts / np.sum(counts)
        self.words = words
        self.counts = counts

    @staticmethod
    def _histo_normalize(token_counts: dict):

        # Count the frequency of tokens
        independent_counts = {}
        for tokens, count in token_counts.items():
            for token in tokens:
                try:
                    independent_counts[token] += 1
                except KeyError:
                    independent_counts[token] = 1

        # More frequent tokens get a lower weight
        independent_counts = {k: 1 / v for k, v in independent_counts.items()}

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
            mapped_weights[tokens] = [x * max_counts / token_counts[tokens] for x in weights]

        return mapped_weights

    def load_to_memory(self, which: Union[str, tuple] = 'squad', batch_size: int = 10, cache_path: str = None,
                       is_fft: bool = False, folderpath: str = ''):
        """

        Returns:
        """
        self.logger.info(f'Loading {which} to memory')

        # Load the dataset from CSV
        tfsplits = {}
        splits = {}
        counts = {}
        token_counts = {'train': {}, 'val': {}}
        # weight_map = {}

        # Create a partial function with tokenize.
        for split in ['train', 'val']:
            # Create a tokenization function
            tokenize = partial(self.tokenize, self.tokenizer, False, is_fft, token_counts[split])

            # Load the data from CSV and tokenize
            splits[split] = Dataset.from_csv(os.path.join(folderpath, f"{split}.csv"), cache_dir=cache_path)
            self.logger.info(f'Data sample for {split}: {splits[split][0]}')

            # Convert text to tokens
            splits[split] = splits[split].map(tokenize, num_proc=self.num_proc, load_from_cache_file=True)
            before_filter = len(splits[split])

            # Filter truncated examples
            splits[split] = splits[split].filter(lambda example: not example['truncated'], load_from_cache_file=True)
            after_filter = len(splits[split])
            self.logger.info(f'Filter with token length {constants.ENCODER_MAX_LEN} before {before_filter} '
                             f'after {after_filter} lost {((before_filter - after_filter) / before_filter) * 100}%')
            counts[split] = len(splits[split])

            # For the train split, keep a count of token
            if split == 'train':
                # weight_map = self._histo_normalize(token_counts['train'])
                pass

            # Seed for splitting validation and test
            self.logger.info(f'Splitting with seed {constants.SEED}')
            if split == 'val':
                print(f'Splitting with seed {constants.SEED}')
                tts = splits[split].train_test_split(test_size=0.4, seed=constants.SEED)
                splits[split] = tts['train']
                splits['test'] = tts['test']

            # We have a dictionary of weigths, create sample_weights using that
            # splits[split] = \
            #     splits[split].map(lambda example:
            #                       {'sample_weights': weight_map[tuple(example['labels'])]},
            #                       load_from_cache_file=False)

            # Convert to TensorFlow dataset
            tfsplits[split] = splits[split].to_tf_dataset(
                batch_size=batch_size,
                columns=['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask'],
                shuffle=True)

        self.count_words(os.path.join(folderpath, "val.csv"))
        return tfsplits, splits, counts

    def load(self, which: Union[str, tuple], batch_size: int = 10, as_batches: bool = False, cache_path: str = None,
             is_fft: bool = False, encoder_max_length: int = 250, token_equalize: bool = False):
        """

        Args:
            which: Which dataset to use Ex: ('squad') or ('super_glue', 'boolq')
            as_batches: Return as batches
            batch_size: Size of each batch
            cache_path: Location to save/load cached files
            is_fft: Add taskname to task
            encoder_max_length: Max token length for encoder
            token_equalize: Equalize token lengths?
        Returns:

        """

        # Ensure the dataset exists and is processed
        led, folderpath = self.encode_and_save(which, cache_path, is_fft=is_fft, token_equalize=token_equalize)

        constants.ENCODER_MAX_LEN = encoder_max_length
        if as_batches:
            raise NotImplementedError('Loading as batches is not implemented')
        else:
            tfsplits, splits, counts = \
                self.load_to_memory(which, batch_size, cache_path, is_fft=is_fft, folderpath=folderpath)
        self.led = led
        return tfsplits, splits, counts
