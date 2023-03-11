import os
import random
import tensorflow as tf
from typing import Union
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

ENCODER_MAX_LEN = 250
DECODER_MAX_LEN = 54
NUM_TOKENS = 20


class LabelEncodeDecode:

    def __init__(self, which):
        if which[0] == 'super_glue':
            if which[1] in ['axb', 'axg']:
                self.lookup = {0: 'entailment', 1: 'not_entailment', -1: 'test'}
            elif which[1] in ['boolq', 'rte', 'wic', 'wsc', 'multirc']:
                self.lookup = {0: 'false', 1: 'true', -1: 'test'}
            elif which[1] == 'cb':
                self.lookup = {0: 'entailment', 1: 'contradiction', 2: 'neutral', -1: 'test'}
            elif which[1] == 'copa':
                self.lookup = {0: 'choice1', 1: 'choice2', -1: 'test'}
            else:
                self.lookup = {}
        elif which[0] == 'glue':
            raise KeyError('Not implemented')
        else:
            self.lookup = {}

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
            return inlabel

    def to_label(self, inpred):

        if self.lookup:
            for k, v in self.lookup.items():
                # Found the label value
                if v == inpred:
                    return k

            # Did not find teh predicted values in the value field
            keys = list(self.lookup.keys())
            random.shuffle(keys)
            return keys[0]
        else:
            # There is no translation involved
            return inpred


class PrepDataset:

    def __init__(self, checkpoint: str, encoder_max_len=ENCODER_MAX_LEN, decoder_max_len=DECODER_MAX_LEN):
        """

        Args:
            checkpoint: Checkpoint from which to load the model
            encoder_max_len: Maximum token length for encoder
            decoder_max_len: Maximum length for decoder
        """

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

        outputs = {'question': question_plus, 'answer': answer_plus}
        return outputs

    @staticmethod
    def encode_super_glue_axb(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

        # Context for answering the question
        first = example['sentence1']
        second = example['sentence2']

        # Convert the numeric label to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"sentence1: {first}"
        question_plus += f" sentence2: {second}"

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_axg(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

        # Context for answering the question
        premise = example['premise']
        hypothesis = example['hypothesis']

        # Convert the numeric label to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"premise: {premise}"
        question_plus += f" hypothesis: {hypothesis}"

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_boolq(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """
        # Context for answering the question
        first = example['question']
        second = example['passage']

        # Convert integer to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"question: {first}"
        question_plus += f" passage: {second}"

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_cb(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

        # Context for answering the question
        first = example['premise']
        second = example['hypothesis']

        # Convert integer to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"question: {first}"
        question_plus += f" passage: {second}"

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_copa(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

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

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_multirc(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

        # Context for answering the question
        question = example['question']
        para = example['paragraph']

        # Convert integer to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"question: {question}"
        question_plus += f" paragraph: {para}"

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_record(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """
        if led:
            pass

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

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_rte(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

        # Context for answering the question
        premise = example['premise']
        hypothesis = example['hypothesis']

        # Convert the numeric label to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"premise: {premise}"
        question_plus += f" hypothesis: {hypothesis}"

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_wic(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

        # Context for answering the question
        sen1 = example['sentence1']
        sen2 = example['sentence2']
        word = example['word']

        # Convert the numeric label to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"word: {word} sentence1: {sen1} sentence2: {sen2}"

        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    @staticmethod
    def encode_super_glue_wsc(led, example):
        """

        Args:
            led: Object of LabelEncodeDecode
            example: The example to encode

        Returns:
        """

        # Context for answering the question
        para = example['text']
        word1 = example['span1_text']
        word2 = example['span2_text']

        # Convert the numeric label to text
        answer = led(example['label'])

        # Adding prompt
        question_plus = f"word1: {word1} word2: {word2} paragraph: {para}"
        outputs = {'question': question_plus, 'answer': answer}
        return outputs

    def _get_encode(self, which):
        """

        Args:
            which: Which dataset

        Returns:

        """

        # Create a label encoder decoder for converting from labels to text and back
        led = LabelEncodeDecode(which)

        if which[0] == 'super_glue':
            if which[1] in 'axb':
                f = self.encode_super_glue_axb
            elif which[1] == 'axg':
                f = self.encode_super_glue_axg
            elif which[1] in 'boolq':
                f = self.encode_super_glue_boolq
            elif which[1] in 'rte':
                f = self.encode_super_glue_rte
            elif which[1] in 'wic':
                f = self.encode_super_glue_wic
            elif which[1] in 'wsc':
                f = self.encode_super_glue_wsc
            elif which[1] in 'multirc':
                f = self.encode_super_glue_multirc
            elif which[1] == 'cb':
                f = self.encode_super_glue_cb
            elif which[1] == 'copa':
                f = self.encode_super_glue_copa
            elif which[1] == 'record':
                f = self.encode_super_glue_record
            else:
                raise KeyError(f'Unexpected super glue dataset {which[1]}')
        elif which[0] == 'squad':
            f = self.encode_squad
        elif which[0] == 'glue':
            raise KeyError('Glue not implemented')
        else:
            raise KeyError(f'Unsupported dataset {which[0]}')

        # Return
        return partial(f, led), led

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
    def tokenize(tokenizer, encoder_max_len, decoder_max_len, example):

        encoder_inputs = tokenizer(example['question'], truncation=True, max_length=encoder_max_len,
                                   padding="max_length")
        if not isinstance(example['answer'], str):
            tmp = str(example['answer']).lower()
        else:
            tmp = str(example['answer'])
        decoder_inputs = tokenizer(tmp, truncation=True, max_length=decoder_max_len, padding="max_length")

        # Set up to return
        input_ids = encoder_inputs['input_ids']
        input_attention = encoder_inputs['attention_mask']
        target_ids = decoder_inputs['input_ids']
        target_attention = decoder_inputs['attention_mask']

        return {'input_ids': input_ids, 'attention_mask': input_attention, 'labels': target_ids,
                'decoder_attention_mask': target_attention}

    def encode_and_save(self, which: Union[str, tuple] = 'squad', cache_path: str = None):
        """

        Returns:
        """
        if not isinstance(which, tuple):
            which = (which,)

        # This is the folder where the data will go
        which_str = ''.join(f'{x}-' for x in which)
        foldername = which_str
        processed_save_path = os.path.join(cache_path, "processed")

        if not os.path.exists(os.path.join(processed_save_path, f"{foldername}/val.csv")):
            os.makedirs(os.path.join(processed_save_path, f"{foldername}"), exist_ok=True)

            # Now we have the text encoded to tokens
            train_dataset = load_dataset(*which, split='train', cache_dir=cache_path)
            valid_dataset = load_dataset(*which, split='validation', cache_dir=cache_path)

            # Convert it into a question answer format
            encoder, led = self._get_encode(which)
            remove_columns = [x for x in train_dataset.column_names if x not in ['id', 'idx', 'label']]
            train_dataset = train_dataset.map(encoder, remove_columns=remove_columns)
            valid_dataset = valid_dataset.map(encoder, remove_columns=remove_columns)

            train_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/train.csv"))
            valid_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/val.csv"))

        # Get the test dataset if available
        if not os.path.exists(os.path.join(processed_save_path, f"{foldername}/test.csv")):
            try:
                encoder, _ = self._get_encode(which)
                test_dataset = load_dataset(*which, split='test', cache_dir=cache_path)

                # Remove the columns we do not need
                remove_columns = [x for x in test_dataset.column_names if x not in ['id', 'idx', 'label']]
                test_dataset = test_dataset.map(encoder, remove_columns=remove_columns)
                test_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/test.csv"))
            except ValueError:
                pass

        return True

    def load_to_memory(self, which: Union[str, tuple] = 'squad', batch_size: int = 10, cache_path: str = None):
        """

        Returns:
        """
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
        tokenize = partial(self.tokenize, self.tokenizer, self.encoder_max_len, self.decoder_max_len)
        for split in ['train', 'val']:
            # Load the data from CSV and tokenize
            splits[split] = Dataset.from_csv(os.path.join(processed_save_path, f"{foldername}/{split}.csv"),
                                             cache_dir=cache_path)

            # Convert text to tokens
            tfsplits[split] = splits[split].map(tokenize)
            counts[split] = len(splits[split])

            # Convert to TensorFlow dataset
            tfsplits[split] = self._to_tf_dataset(tfsplits[split])

            # Convert to a dataset
            shuffling = True if split == 'train' else False
            tfsplits[split] = self.create_dataset(tfsplits[split], batch_size=batch_size, shuffling=shuffling,
                                                cache_path=cache_path)

        try:
            splits['test'] = Dataset.from_csv(os.path.join(processed_save_path, f"{foldername}/test.csv"))
            counts['test'] = len(splits['test'])
        except FileNotFoundError:
            splits['test'] = []
            counts['test'] = 0

        return tfsplits, splits, counts

    def load(self, which: Union[str, tuple], batch_size: int = 10, as_batches: bool = False, cache_path: str = None):
        """

        Args:
            which: Which dataset to use Ex: ('squad') or ('super_glue', 'boolq')
            as_batches: Return as batches
            batch_size: Size of each batch
            cache_path: Location to save/load cached files
        Returns:

        """
        # Ensure the dataset exists and is processed
        self.encode_and_save(which, cache_path)

        if as_batches:
            raise ValueError('Not implemented')
            #
            # train_dataset, val_datset, test_datset = self.load_as_batches()
        else:
            tfsplits, splits, counts = self.load_to_memory(which, batch_size, cache_path)

        return tfsplits, splits, counts


if __name__ == '__main__':
    # Prepare the Dataset
    cp = 't5-small'
    dprep = PrepDataset(checkpoint=cp)
    which_d = 'super_glue'
    cp = os.path.join(os.path.dirname(__file__), "../cache")

    # out = dprep.get(which=which_d, batch_size=100, cache_path=cp)
    # ('super_glue', 'axb'), ('super_glue', 'axg'),
    whiches = ('squad',  ('super_glue', 'boolq'),
               ('super_glue', 'record'), ('super_glue', 'rte'), ('super_glue', 'wic'), ('super_glue', 'wsc'),
               ('super_glue', 'multirc'), ('super_glue', 'cb'), ('super_glue', 'copa'))

    # for wo in whiches:
    #     dprep.encode_and_save(wo, cache_path=cp)
    dprep.load_to_memory(('super_glue', 'boolq'), batch_size=10, cache_path=cp)

