import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Union
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from constants import ENCODER_MAX_LEN, DECODER_MAX_LEN, NUM_SOFT_TOKENS


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


def preprocess_data(text_pairs, tokenizer, model):
    """

    Args:
        text_pairs: Pairs of input and output texts
        tokenizer:
        model:
    Returns:

    """
    orig_text = text_pairs[0]
    orig_encoded = tokenizer.batch_encode_plus(orig_text, max_length=ENCODER_MAX_LEN, padding='max_length',
                                               truncation=True, return_attention_mask=True, return_tensors='tf')
    orig_input_ids = np.array(orig_encoded["input_ids"], dtype="int32")
    orig_attention_masks = np.array(orig_encoded["attention_mask"], dtype="int32")

    target_text = text_pairs[1]
    target_encoded = tokenizer.batch_encode_plus(target_text, max_length=DECODER_MAX_LEN, padding='max_length',
                                                 truncation=True, return_tensors='tf')

    # Decoder needs it shifted
    label_ids = np.array(target_encoded['input_ids'])
    decoder_input_ids = model.layers[-1]._shift_right(label_ids)
    return [orig_input_ids, orig_attention_masks, decoder_input_ids], label_ids


class BatchDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 tokenizer,
                 model,
                 data_filename,
                 n_examples,
                 batch_size=16,
                 shuffle=True,
                 ):

        self.tokenizer = tokenizer
        self.model = model
        self.n_examples = n_examples
        self.data_filename = data_filename
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Initialize row order, call on_epoch_end to shuffle row indices
        self.row_order = np.arange(1, self.n_examples + 1)
        self.on_epoch_end()

    def __len__(self):
        # Return the number of batches in the full dataset
        return self.n_examples // self.batch_size

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size

        # Indices to skip are the ones in the shuffled row_order before and
        # after the chunk we'll use for this batch
        batch_idx_skip = self.row_order[:batch_start] + self.row_order[batch_end:]
        df = pd.read_csv(self.data_filename, skiprows=batch_idx_skip)

        text_pairs = (df['question'].values.astype(str).tolist(), df['answer'].values.astype(str).tolist())
        batch_data = preprocess_data(text_pairs, self.tokenizer, self.model)

        return batch_data

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.row_order = list(np.random.permutation(self.row_order))


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

        outputs = {'q': question_plus, 'answer': answer_plus}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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

        outputs = {'q': question_plus, 'answer': answer}
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
        outputs = {'q': question_plus, 'answer': answer}
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
    def _shorten_to_length(tokenizer, example):
        """

        Args:
            tokenizer: T5 tokenizer instance
            example: Example to process

        Returns:

        """

        # This is the max length we would like to tokenize
        max_length = ENCODER_MAX_LEN - NUM_SOFT_TOKENS

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

        # During testing, only the inputs_ids of the tokens are required
        if is_test:
            # Now encode the tokens, this time we can be sure that there are at least NUM_SOFT_TOKENS worth of paddings
            encoder_inputs = tokenizer(example['question'], truncation=True, max_length=ENCODER_MAX_LEN,
                                       padding="max_length", return_tensors="tf")

            return encoder_inputs['input_ids']

        # Now encode the tokens, this time we can be sure that there are at least NUM_SOFT_TOKENS worth of paddings
        encoder_inputs = tokenizer(example['question'], truncation=True, max_length=ENCODER_MAX_LEN,
                                   padding="max_length", return_tensors="tf")

        if not isinstance(example['answer'], str):
            tmp = str(example['answer']).lower()
        else:
            tmp = str(example['answer'])
        decoder_inputs = tokenizer(tmp, truncation=True, max_length=ENCODER_MAX_LEN, padding="max_length")

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

        # Encode it into a question answer format
        encoder, led = self._get_encode(which)

        # Shorten the text to a length so that there is no truncation from the soft prompt
        stl = partial(self._shorten_to_length, self.tokenizer)

        which_str = ''.join(f'{x}-' for x in which)
        foldername = which_str
        processed_save_path = os.path.join(cache_path, "processed")

        if not os.path.exists(os.path.join(processed_save_path, f"{foldername}/val.csv")):
            os.makedirs(os.path.join(processed_save_path, f"{foldername}"), exist_ok=True)

            # Now we have the text encoded to tokens
            train_dataset = load_dataset(*which, split='train', cache_dir=cache_path)
            valid_dataset = load_dataset(*which, split='validation', cache_dir=cache_path)

            # Convert it into a question answer format
            remove_columns = [x for x in train_dataset.column_names if x not in ['id', 'idx', 'label']]
            train_dataset = train_dataset.map(encoder, remove_columns=remove_columns)
            valid_dataset = valid_dataset.map(encoder, remove_columns=remove_columns)

            remove_columns = ['q']
            train_dataset = train_dataset.map(stl, remove_columns=remove_columns)
            valid_dataset = valid_dataset.map(stl, remove_columns=remove_columns)

            train_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/train.csv"))
            valid_dataset.to_csv(os.path.join(processed_save_path, f"{foldername}/val.csv"))

        # Get the test dataset if available
        if not os.path.exists(os.path.join(processed_save_path, f"{foldername}/test.csv")):
            try:
                test_dataset = load_dataset(*which, split='test', cache_dir=cache_path)

                # Remove the columns we do not need
                remove_columns = [x for x in test_dataset.column_names if x not in ['id', 'idx', 'label']]
                test_dataset = test_dataset.map(encoder, remove_columns=remove_columns)

                remove_columns = ['q']
                test_dataset = test_dataset.map(stl, remove_columns=remove_columns)
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

        # Create a partial function with tokenize.
        tokenize = partial(self.tokenize, self.tokenizer, False)
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

    def load_as_batches(self, which: Union[str, tuple] = 'squad', batch_size: int = 10, cache_path: str = None,
                        **kwargs):
        """

        Returns:
        """
        if not isinstance(which, tuple):
            which = (which,)

        model = kwargs['model']

        # This is the folder where the data will go
        which_str = ''.join(f'{x}-' for x in which)
        foldername = which_str
        processed_save_path = os.path.join(cache_path, "processed")

        # Load the dataset from CSV
        tfsplits = {}
        splits = {}
        counts = {}

        # Create a partial function with tokenize.
        for split in ['train', 'val']:
            filename = os.path.join(processed_save_path, f"{foldername}/{split}.csv")

            num_samples = pd.read_csv(filename).shape[0]

            # Load the data from CSV and tokenize
            tfsplits[split] = BatchDataGenerator(tokenizer=self.tokenizer, model=model, n_examples=num_samples,
                                                 data_filename=filename, batch_size=batch_size)
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
        # Ensure the dataset exists and is processed
        self.encode_and_save(which, cache_path)

        if as_batches:
            tfsplits, splits, counts = self.load_as_batches(which, batch_size, cache_path, **kwargs)
        else:
            tfsplits, splits, counts = self.load_to_memory(which, batch_size, cache_path)

        return tfsplits, splits, counts


if __name__ == '__main__':
    # Prepare the Dataset
    cp = 't5-small'
    dprep = PrepDataset(checkpoint=cp)
    which_d = 'super_glue'
    cache_p = os.path.join(os.path.dirname(__file__), "../cache")

    # out = dprep.get(which=which_d, batch_size=100, cache_path=cp)
    # ('super_glue', 'axb'), ('super_glue', 'axg'),
    whiches = ('squad',  ('super_glue', 'boolq'),
               ('super_glue', 'record'), ('super_glue', 'rte'), ('super_glue', 'wic'), ('super_glue', 'wsc'),
               ('super_glue', 'multirc'), ('super_glue', 'cb'), ('super_glue', 'copa'))

    for wo in whiches:
        dprep.encode_and_save(wo, cache_path=cp)
