import os
import random
import tensorflow as tf
from typing import Union
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from utils import constants


class LabelEncodeDecode:

    def __init__(self, which):
        self.which = which
        if which[0] == 'super_glue':
            if which[1] in ['axb', 'axg']:
                self.lookup = {0: 'entailment', 1: 'not_entailment', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 6
            elif which[1] in ['boolq', 'rte', 'wic', 'wsc', 'multirc']:
                self.lookup = {0: 'false', 1: 'true', -1: 'test'}
                # self.lookup = {0: 'absolute truth', 1: 'terrible lie', -1: 'test1 test2'}
                # True and False are both one token each
                constants.DECODER_MAX_LEN = 2
            elif which[1] == 'cb':
                self.lookup = {0: 'entailment', 1: 'contradiction', 2: 'neutral', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 6
            elif which[1] == 'copa':
                self.lookup = {0: 'choice1', 1: 'choice2', -1: 'test'}
                # choice1 and choice2  are 2 tokens long plus one end of sequence is 3
                constants.DECODER_MAX_LEN = 3
            else:
                self.lookup = {}
                # This is a longer answer and requires 50
                constants.DECODER_MAX_LEN = 20
        elif which[0] == 'glue':
            if which[1] in ['qnli', 'rte', 'wnli']:
                self.lookup = {0: 'entailment', 1: 'not_entailment', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 6
            elif which[1] in ['sst2', ]:
                self.lookup = {0: 'negative', 1: 'positive', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 2
            elif which[1] in ['cola', ]:
                self.lookup = {0: 'unacceptable', 1: 'acceptable', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 2
            elif which[1] in ['mrpc', ]:
                self.lookup = {0: 'equivalent', 1: 'not_equivalent', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 6
            elif which[1] in ['qqp', ]:
                self.lookup = {0: 'duplicate', 1: 'not_duplicate', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 6
            elif which[1] in ['mnli', ]:
                self.lookup = {0: 'entailment', 1: 'contradiction', 2: 'neutral', -1: 'test'}
                # not_entailment is 5 tokens long plus one end of sequence is 6
                constants.DECODER_MAX_LEN = 6
            else:
                self.lookup = {}
                # This is a longer answer and requires 50
                constants.DECODER_MAX_LEN = 50
        else:
            self.lookup = {}
            # This is a longer answer and requires 50
            constants.DECODER_MAX_LEN = 50

    def __call__(self, inlabel, *args, **kwargs):
        return str(self.from_label(inlabel))

    def __getitem__(self, item):
        return self.to_label(item)

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
            keys = list(self.lookup.keys())
            keys = [x for x in keys if x != -1]
            random.shuffle(keys)
            return keys[0]
        else:
            # There is no translation involved
            return inpred


class PrepDataset:

    def __init__(self, logger, checkpoint: str, encoder_max_len=constants.ENCODER_MAX_LEN,
                 decoder_max_len=constants.DECODER_MAX_LEN):
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
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=self.encoder_max_len)
        self.supported_type = [('squad', ), ('super_glue', 'axb')]

        self.train_dataset = None
        self.valid_dataset = None

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

        outputs = {'q': question_plus, 'answer': answer_plus}
        return outputs

    @staticmethod
    def encode_super_glue(led, example):
        """

        Args:
            led:
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
            question_plus = f"sentence1: {first}"
            question_plus += f" sentence2: {second}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in ['axg', 'cb', 'rte']:
            # Context for answering the question
            premise = example['premise']
            hypothesis = example['hypothesis']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"hypothesis: {hypothesis}"
            question_plus += f" premise: {premise}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'boolq':
            # Context for answering the question
            first = example['question']
            second = example['passage']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"question: {first}"
            question_plus += f" passage: {second}"

            outputs = {'q': question_plus, 'answer': answer}
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
            question_plus = f"question: {question}"
            question_plus += f" choice1: {c1} choice2: {c2}"
            question_plus += f" premise: {premise}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'multirc':
            # Context for answering the question
            question = example['question']
            para = example['paragraph']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"question: {question}"
            question_plus += f" paragraph: {para}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'record':
            # Context for answering the question
            passage = example['passage']

            # Does replacing this with extra id work better?
            # Do not know but it might be worth trying
            query = example['query']
            query = query.replace('@placeholder', '<extra_id_0>')

            # Convert integer to text
            # shortest_answer = [len(x) for x in example['answers']]
            # idx = shortest_answer.index(min(shortest_answer))
            # answer = example['answers'][idx]
            answer = ', '.join([i for i in list(example['answers'])])

            # Adding prompt
            question_plus = f"query: {query}"
            question_plus += f" passage: {passage}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'rte':
            # Context for answering the question
            premise = example['premise']
            hypothesis = example['hypothesis']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"hypothesis: {hypothesis}"
            question_plus += f" premise: {premise}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'wic':
            # Context for answering the question
            sen1 = example['sentence1']
            sen2 = example['sentence2']
            word = example['word']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"word: {word} sentence1: {sen1} sentence2: {sen2}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'wsc.fixed':
            # Context for answering the question
            para = example['text']
            span1 = example['span1_text']
            span2 = example['span2_text']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"span1: {span1} span2: {span2} paragraph: {para}"
            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        else:
            raise KeyError('Unknown datatype for translation')

    @staticmethod
    def encode_glue(led, example):
        """

        Args:
            led:
            example:

        Returns:
        """
        if led.which[1] in ['cola', 'sst2']:
            sentence = example['sentence']
            question_plus = f"sentence: {sentence}"

            # Convert the numeric label to text
            answer = led(example['label'])

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] == 'mnli':
            # Context for answering the question
            premise = example['premise']
            hypothesis = example['hypothesis']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"hypothesis: {hypothesis}"
            question_plus += f" premise: {premise}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in ['mrpc', 'rte', 'stsb', 'wnli']:
            # Context for answering the question
            first = example['sentence1']
            second = example['sentence2']

            # Convert the numeric label to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"sentence1: {first}"
            question_plus += f" sentence2: {second}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in 'qnli':
            # Context for answering the question
            first = example['question']
            second = example['sentence']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"question: {first}"
            question_plus += f" sentence: {second}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        elif led.which[1] in 'qqp':
            # Context for answering the question
            first = example['question1']
            second = example['question2']

            # Convert integer to text
            answer = led(example['label'])

            # Adding prompt
            question_plus = f"question1: {first}"
            question_plus += f" question2: {second}"

            outputs = {'q': question_plus, 'answer': answer}
            return outputs
        else:
            raise KeyError('Unknown datatype for translation')

    def _get_encode(self, which):
        """

        Args:
            which: Which dataset

        Returns:

        """

        # Create a label encoder decoder for converting from labels to text and back
        led = LabelEncodeDecode(which)

        if which[0] == 'super_glue':
            return partial(self.encode_super_glue, led), led
        elif which[0] == 'glue':
            return partial(self.encode_glue, led), led
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
        dataset = dataset.prefetch(batch_size*10)

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
        tokens = tokenizer(example['q'])
        len_tokens = len(tokens['input_ids'])
        if len_tokens <= max_length:
            return {'question': example['q']}

        # Each word is at least one token so start with as many words
        new_example = example['q'].split(' ')[:max_length]

        while len_tokens > max_length:
            # Join the example back with space and remove the extra space that was added
            new_example = ''.join(f"{x} " for x in new_example)[:-1]
            tokens = tokenizer(new_example)
            len_tokens = len(tokens['input_ids'])

            if len_tokens > max_length:
                new_example = new_example.split(' ')[:-1]

        return {'question': new_example}

    @staticmethod
    def tokenize(tokenizer, is_test, example):
        """
        Our objective is not only to tokenize the input but also to ensure that the after soft prompt is included, the
        embeddings that are removed are all paddings only

        Args:
            tokenizer: Instance of Autotokenizer initialized appropriately for the checkpoint
            is_test: Whether this is a test set we are working with
            example: Example to be tokenized

        Returns:
        """

        # Now encode the tokens, this time we can be sure that there are at least NUM_SOFT_TOKENS worth of paddings
        encoder_inputs = tokenizer(example['question'], truncation=True, max_length=constants.ENCODER_MAX_LEN,
                                   padding="max_length", return_tensors="tf")

        if is_test:
            return encoder_inputs['input_ids']

        if not isinstance(example['answer'], str):
            tmp = str(example['answer']).lower()
        else:
            tmp = str(example['answer'])
        decoder_inputs = tokenizer(tmp, truncation=True, max_length=constants.DECODER_MAX_LEN, padding="max_length",
                                   return_tensors="tf")

        # Set up to return
        input_ids = encoder_inputs['input_ids'][0]
        input_attention = encoder_inputs['attention_mask'][0]
        target_ids = decoder_inputs['input_ids'][0]
        target_attention = decoder_inputs['attention_mask'][0]

        return {'input_ids': input_ids, 'attention_mask': input_attention, 'labels': target_ids,
                'decoder_attention_mask': target_attention}

    def encode_and_save(self, which: Union[str, tuple] = 'squad', cache_path: str = None):
        """

        Returns:
        """
        if not isinstance(which, tuple):
            which = (which,)

        self.logger.info(f'Encoding dataset {which}')

        # Encode it into a question answer format
        # It also set the value for teh DECODE_MAX_LENGTH based on the answer type
        encoder, led = self._get_encode(which)
        self.logger.info(f'Using function {encoder} with answer lookup {led.lookup}')

        # Shorten the text to a length so that there is no truncation from the soft prompt
        stl = partial(self._shorten_to_length, self.tokenizer)

        which_str = ''.join(f'{x}-' for x in which)
        foldername = which_str
        processed_save_path = os.path.join(cache_path, "processed")

        if not os.path.exists(os.path.join(processed_save_path, f"{foldername}/val.csv")):
            os.makedirs(os.path.join(processed_save_path, f"{foldername}"), exist_ok=True)

            # Now we have the text encoded to tokens
            train_dataset = load_dataset(*which, split='train', cache_dir=cache_path)
            try:
                valid_dataset = load_dataset(*which, split='validation', cache_dir=cache_path)
            # For  Glue, MNLI the validaiton set is called validation_matched or validation_mismatched
            # We load the validation mismatched here
            except ValueError:
                valid_dataset = load_dataset(*which, split='validation_mismatched', cache_dir=cache_path)

            tts = valid_dataset.train_test_split(test_size=0.3)
            valid_dataset = tts['train']
            test_dataset = tts['test']

            # Convert it into a question answer format
            remove_columns = [x for x in train_dataset.column_names if x not in ['id', 'idx', 'label']]
            train_dataset = train_dataset.map(encoder, remove_columns=remove_columns)
            valid_dataset = valid_dataset.map(encoder, remove_columns=remove_columns)
            test_dataset = test_dataset.map(encoder, remove_columns=remove_columns)

            remove_columns = ['q']
            train_dataset = train_dataset.map(stl, remove_columns=remove_columns)
            valid_dataset = valid_dataset.map(stl, remove_columns=remove_columns)
            test_dataset = test_dataset.map(stl, remove_columns=remove_columns)

            self.logger.info(f'Data sample for train: {train_dataset[0]}')
            self.logger.info(f'Data sample for validation: {valid_dataset[0]}')
            self.logger.info(f'Data sample for set: {test_dataset[0]}')

            train_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/train.csv"))
            valid_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/val.csv"))
            test_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/test.csv"))
        else:
            self.logger.info(f'Decoding was not performed as cache was found')

        # This is the completely unseen final test
        if not os.path.exists(os.path.join(processed_save_path, f"{foldername}/ftest.csv")):
            try:
                test_dataset = load_dataset(*which, split='test', cache_dir=cache_path)

                # Remove the columns we do not need
                remove_columns = [x for x in test_dataset.column_names if x not in ['id', 'idx', 'label']]
                test_dataset = test_dataset.map(encoder, remove_columns=remove_columns)

                remove_columns = ['q']
                test_dataset = test_dataset.map(stl, remove_columns=remove_columns)
                test_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/ftest.csv"))
            except ValueError:
                pass

        return True

    def load_to_memory(self, which: Union[str, tuple] = 'squad', batch_size: int = 10, cache_path: str = None):
        """

        Returns:
        """
        self.logger.info(f'Loading {which} to memory')

        if not isinstance(which, tuple):
            which = (which,)

        # This is the folder where the data will go
        which_str = ''.join(f'{x}-' for x in which)
        foldername = which_str
        processed_save_path = os.path.join(cache_path, "processed")

        # Load the dataset from CSV
        tfsplits = {}
        splits = {}
        counts = {}

        # Create a partial function with tokenize.
        tokenize = partial(self.tokenize, self.tokenizer, False)
        for split in ['train', 'val', 'test']:
            # Load the data from CSV and tokenize
            splits[split] = Dataset.from_csv(os.path.join(processed_save_path, f"{foldername}/{split}.csv"),
                                             cache_dir=cache_path)
            self.logger.info(f'Data sample for {split}: {splits[split]}')

            # Convert text to tokens
            tfsplits[split] = splits[split].map(tokenize)
            counts[split] = len(splits[split])

            # Convert to TensorFlow dataset
            tfsplits[split] = tfsplits[split].to_tf_dataset(
                batch_size=batch_size, columns=['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask'])

        try:
            splits['ftest'] = Dataset.from_csv(os.path.join(processed_save_path, f"{foldername}/test.csv"))
            counts['ftest'] = len(splits['test'])
        except FileNotFoundError:
            splits['ftest'] = []
            counts['ftest'] = 0

        return tfsplits, splits, counts

    def load(self, which: Union[str, tuple], batch_size: int = 10, as_batches: bool = False, cache_path: str = None,
             **kwargs):
        """

        Args:
            which: Which dataset to use Ex: ('squad') or ('super_glue', 'boolq')
            as_batches: Return as batches
            batch_size: Size of each batch
            cache_path: Location to save/load cached files
        Returns:

        """
        if kwargs:
            pass

        # Ensure the dataset exists and is processed
        self.encode_and_save(which, cache_path)

        if as_batches:
            raise NotImplementedError('Loading as batches is not implemented')
        else:
            tfsplits, splits, counts = self.load_to_memory(which, batch_size, cache_path)

        return tfsplits, splits, counts
