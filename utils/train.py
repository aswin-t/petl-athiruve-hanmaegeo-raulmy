# Credit:https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-Datasets%20Training.ipynb
import os
import abc
import tensorflow as tf
from functools import partial
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from utils.model import TFPromptT5ForConditionalGeneration

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'
ENCODER_MAX_LEN = 250
DECODER_MAX_LEN = 54
NUM_TOKENS = 20


class FineTune(TFPromptT5ForConditionalGeneration, abc.ABC):
    def __init__(self, *args, log_dir=None, cache_dir=None, **kwargs):
        if log_dir or cache_dir:
            pass

        # Initialize the TFPromptT5ForConditionalGeneration class
        super().__init__(*args, **kwargs)

        # This tracks the mean loss
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    @tf.function
    def train_step(self, data):
        """

        Args:
            data: X and y data for training

        Returns:

        """

        # What does X look like?
        x = data
        # Extract the Y as labels
        y = x["labels"]

        # Flatening the y, but why?
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            # There must be a __call__ method that has the forward pass
            outputs = self(x, training=True)

            # The calculated loss and the
            loss = outputs[0]
            logits = outputs[1]

            # Mean loss
            loss = tf.reduce_mean(loss)

            # Get the gradient for the trainable weights
            grads = tape.gradient(loss, self.trainable_variables)

        # Apply the calcaulated gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Get the current learning rate
        # noinspection PyProtectedMember
        lr = 0.0
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})
        return metrics

    def test_step(self, data):
        """

        Args:
            data:

        Returns:

        """

        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)

        # tf.Gradient.Tape() is not set here as we do nto want gradient calculations
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        # Track the loss here
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


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
        self.supported_type = ['squad', ]

        self.train_dataset = None
        self.valid_dataset = None

    def __getstate__(self):
        state = self.__dict__
        del state['tokenizer']

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, model_max_length=self.encoder_max_len)

    def _get_tags(self, which):
        """

        Args:
            which: Which dataset to use?

        Returns:

        """

        # These are the library of prompts associated with each of these datasets
        if which == 'squad':
            self.prompt_tag = 'answer_me:'
            self.context_tag = ' context:'
        else:
            raise ValueError('Unsupported which ')

        # Add a space if a context is provided
        if self.context_tag:
            self.context_tag += ' '

    @staticmethod
    def _to_tf_dataset(dataset):
        columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
        dataset.set_format(type='tensorflow', columns=columns)
        return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32,
                        'labels': tf.int32, 'decoder_attention_mask': tf.int32, }
        return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
                         'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}
        ds = tf.data.Dataset.from_generator(lambda: dataset, return_types, return_shapes)
        return ds

    @staticmethod
    def create_dataset(dataset, cache_path=None, batch_size=4, buffer_size=1000, shuffling=True):
        """

        Args:
            dataset: Dataset to convert
            cache_path: Path
            batch_size:
            buffer_size:
            shuffling:

        Returns:

        """

        if cache_path is not None:
            dataset = dataset.cache(cache_path)

        if shuffling:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size*10)

        return dataset

    @staticmethod
    def encode(prompt_tag, context_tag, example):
        """

        Args:
            prompt_tag: Tag to add to the prompt
            context_tag: Tag to add to context
            example: The example to encode

        Returns:

        """

        # Context for answering the question
        context = example['context']
        # The question
        question = example['question']
        # Answer for the question, required for training
        answer = example['answers']['text']

        # Adding prompt
        question_plus = f"{prompt_tag} {str(question)}"
        question_plus += f"{context_tag}{str(context)} "

        # This is the answer we are trying to generate
        answer_plus = ', '.join([i for i in list(answer)])
        answer_plus = f"{answer_plus} "

        outputs = {'question': question_plus, 'answer': answer_plus}
        return outputs

    @staticmethod
    def tokenize(tokenizer, encoder_max_len, decoder_max_len, example):

        encoder_inputs = tokenizer(example['question'], truncation=True, max_length=encoder_max_len,
                                   padding="max_length")
        decoder_inputs = tokenizer(example['answer'], truncation=True, max_length=decoder_max_len,
                                   padding="max_length")

        # Set up to return
        input_ids = encoder_inputs['input_ids']
        input_attention = encoder_inputs['attention_mask']
        target_ids = decoder_inputs['input_ids']
        target_attention = decoder_inputs['attention_mask']

        return {'input_ids': input_ids, 'attention_mask': input_attention, 'labels': target_ids,
                'decoder_attention_mask': target_attention}

    def get(self, which: str, batch_size=4, cache_path=None):
        """

        Returns:
        """

        which = which.lower()

        # These are the library of prompts associated with each of these datasets
        if which not in self.supported_type:
            raise ValueError(f'Unsupported which type {which}, currently supported ones are  ')

        # These get the prompts and context tags
        self._get_tags(which)

        processed_save_path = os.path.join(cache_path, "processed")

        try:
            train_dataset = load_from_disk(os.path.join(processed_save_path, f"{which}/train"))
            valid_dataset = load_from_disk(os.path.join(processed_save_path, f"{which}/val"))

        except OSError:
            # Now we have the text encoded to tokens
            train_dataset = load_dataset(which, split='train', cache_dir=cache_path)
            valid_dataset = load_dataset(which, split='validation', cache_dir=cache_path)

            # Convert it into a question answer format
            encode = partial(self.encode, self.prompt_tag, self.context_tag)
            train_dataset = train_dataset.map(encode)
            valid_dataset = valid_dataset.map(encode)

            # Tokenize the input
            tokenize = partial(self.tokenize, self.tokenizer, self.encoder_max_len, self.decoder_max_len)
            train_dataset = train_dataset.map(tokenize)
            valid_dataset = valid_dataset.map(tokenize)

            # # Save the datasets to the processed data folder
            train_dataset.save_to_disk(os.path.join(processed_save_path, f"{which}/train"))
            valid_dataset.save_to_disk(os.path.join(processed_save_path, f"{which}/val"))

        ntrain = len(train_dataset)
        nval = len(valid_dataset)
        steps = ntrain // batch_size
        val_steps = nval // batch_size

        # Next we need to convert it into a Tensorflow dataset
        train_dataset = self._to_tf_dataset(train_dataset)
        valid_dataset = self._to_tf_dataset(valid_dataset)

        # Convert into an TensorFlow iterator
        tf_train_ds = self.create_dataset(train_dataset, batch_size=batch_size, shuffling=True,
                                          cache_path=None)
        tf_valid_ds = self.create_dataset(valid_dataset, batch_size=batch_size, shuffling=False,
                                          cache_path=None)

        return tf_train_ds, tf_valid_ds, steps, val_steps


def main_flow(checkpoint="t5-small", which='squad', batch_size=4, epochs=10, cache_path=None, debug=False):
    # learning_rate = 0.001  # Instead set a static learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    # Prepare the Dataset
    dprep = PrepDataset(checkpoint=checkpoint)
    train_ds, val_ds, steps, val_steps = dprep.get(which=which, batch_size=batch_size, cache_path=cache_path)

    # Create a model instance
    model = FineTune.from_pretrained(checkpoint)

    # The prompt part of the code requires we create an array that is equal to the batch size. Need to rebuild it here
    model.encoder.prompt.build((batch_size, ENCODER_MAX_LEN, model.encoder.prompt.soft_prompt.shape[-1]))

    # Print a few elements of the sof prompt
    tf.print(model.encoder.prompt.soft_prompt[:10, :10])

    # Compile the model with Categorical accuracy metric
    model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy'), ],
                  run_eagerly=debug)
    print("Total Steps: ", steps)
    print("Total Validation Steps: ", val_steps)

    # Ready to train the model
    # model.fit(train_ds, epochs=epochs, steps_per_epoch=steps-1, callbacks=[],
    #           validation_data=val_ds, validation_steps=val_steps-1, initial_epoch=0)
    model.fit(train_ds, epochs=epochs, callbacks=[], validation_data=val_ds, initial_epoch=0)

    # Print a few elements of the sof prompt
    tf.print(model.encoder.prompt.soft_prompt[:10, :10])

    return model


if __name__ == '__main__':
    cp = os.path.join(os.path.dirname(__file__), "../cache")
    model_o = main_flow(checkpoint='t5-small', which='squad', batch_size=100, cache_path=cp, debug=False, epochs=1)
