import abc
import copy
import warnings
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Tuple
from transformers.models.t5 import TFT5PreTrainedModel, TFT5Block, TFT5LayerNorm, T5Config, TFT5ForConditionalGeneration
from transformers.modeling_tf_utils import TFCausalLanguageModelingLoss, get_initializer, shape_list, TFModelInputType
from transformers.modeling_tf_utils import unpack_inputs, keras_serializable
from transformers.utils import ContextManagers
from transformers.modeling_tf_outputs import TFSeq2SeqLMOutput, TFBaseModelOutput, \
    TFBaseModelOutputWithPastAndCrossAttentions
from utils.metric import SelectiveSparseTopKCategoricalAccuracy
from keras.optimizers.optimizer_experimental.adafactor import Adafactor

_HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = tf.ones((num_layers,
num_heads))`.
"""


class PromptDenseLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = 20
        self.soft_prompt = None
        # self.ones_array = None

    def build(self, input_shape):
        """
        This function is called when a batch so that the appropriate sized arrays can be created
        Args:
            input_shape: Size of the input embedding array

        Returns:

        """

        # Create a prompt that
        self.soft_prompt = self.add_weight(name='prompt-weight', shape=[self.num_tokens, input_shape[2]],
                                           initializer="glorot_normal")

        # This is added at the end to let super*() know that build is complete
        super().build(input_shape)

    def call(self, input_embeds, *args, **kwargs):
        """

        Args:
            input_embeds: Input embeddings
            *args: unused
            **kwargs: unused

        Returns:

        """

        # This scales the prompt to the input batch size
        # 1. create an ones array of batch size (batch_size, )
        ones_array = tf.ones((tf.shape(input_embeds)[0], ), dtype=self.dtype)

        # 2. tensordot with soft prompt
        # Gives the shape (batch_size, num_tokens, model_d)
        scaled = tf.tensordot(ones_array, self.soft_prompt, axes=[[], []])

        # Now concat the input embedding to the output embeddings
        input_embeds = tf.concat((scaled, input_embeds), axis=1)

        # TODO athiruve - What about the end of sequence?
        return input_embeds[:, :-self.num_tokens, :]

    def compute_output_shape(self, input_shape):
        """

        Args:
            input_shape: Shape of batch input

        Returns: List with shape of batch output

        """

        return input_shape


####################################################
# The full model without a specific pretrained or finetuning head is
# provided as a tf.keras.layers.Layer usually called "TFT5MainLayer"
####################################################
@keras_serializable
class PromptTFT5MainLayer(tf.keras.layers.Layer):
    config_class = T5Config

    def __init__(self, config, embed_tokens=None, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.config = config
        self.num_hidden_layers = config.num_layers

        self.block = [
            TFT5Block(config, has_relative_attention_bias=bool(i == 0), name=f"block_._{i}")
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="final_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

        self.prompt = PromptDenseLayer(name='prompt')

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ) -> Tuple:

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            # if `self.embed_tokens.load_weight_prefix` is set, runs the embedding operation with the correct name
            # scope, so that its weights are registered with the desired name for loading/storing. When `tf.name_scope`
            # is used with a name ending in `/`, that name replaces the current name scope.
            # (embeddings with tf.name_scope: self.embed_tokens.load_weight_prefix/self.embed_tokens.name/embeddings:0)
            context = []
            if hasattr(self.embed_tokens, "load_weight_prefix"):
                context.append(tf.name_scope(self.embed_tokens.load_weight_prefix + "/"))
            with ContextManagers(context):
                # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
                # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
                tf.debugging.assert_less(
                    input_ids,
                    tf.cast(self.embed_tokens.input_dim, dtype=input_ids.dtype),
                    message=(
                        "input_ids must be smaller than the embedding layer's input dimension (got"
                        f" {tf.math.reduce_max(input_ids)} >= {self.embed_tokens.input_dim})"
                    ),
                )

                # Added the soft prompt layer for only the encoder
                inputs_embeds = self.embed_tokens(input_ids)
                if not self.is_decoder:
                    inputs_embeds = self.prompt(inputs_embeds)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            shape_list(past_key_values[0][0])[2] + seq_length if past_key_values is not None else seq_length
        )

        if attention_mask is None:
            attention_mask = tf.fill((batch_size, mask_seq_length), 1)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = shape_list(encoder_hidden_states)[1]
            encoder_attention_mask = tf.fill((batch_size, encoder_seq_length), 1)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        attention_mask = tf.cast(attention_mask, dtype=inputs_embeds.dtype)
        num_dims_attention_mask = len(shape_list(attention_mask))
        if num_dims_attention_mask == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif num_dims_attention_mask == 2:
            # Provided a padding mask of dimensions [batch_size, mask_seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length,
            #  mask_seq_length]
            if self.is_decoder:
                seq_ids = tf.range(mask_seq_length)
                causal_mask = tf.less_equal(
                    tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                    seq_ids[None, :, None],
                )
                causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                if past_key_values[0] is not None:
                    extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and  -1e9 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
        # extended_attention_mask = tf.math.equal(extended_attention_mask,
        #                                         tf.transpose(extended_attention_mask, perm=(-1, -2)))
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=extended_attention_mask.dtype)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        present_key_value_states = () if use_cache and self.is_decoder else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds, training=training)

        for idx, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                encoder_layer_head_mask=encoder_head_mask[idx] if encoder_head_mask is not None else None,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                training=training,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias),
            # (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, past_key_values, (self-attention weights),
            # (self-attention position bias), (cross-attention position bias), (cross-attention weights),
            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            # append next layer key value states
            if present_key_value_state is not None and use_cache and self.is_decoder:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            # need to check if is decoder here as well for special cases when using keras compile
            if use_cache and self.is_decoder:
                outputs = outputs + (present_key_value_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
                if self.is_decoder:
                    outputs + (all_cross_attentions,)
            return outputs  # last-layer hidden state, (past_key_values), (all hidden states), (all attentions), (all_cross_attentions)

        if self.is_decoder:
            return TFBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
        else:
            return TFBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )


class TFPromptT5ForConditionalGeneration(TFT5PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model_dim = config.d_model
        with tf.device('cpu:0'):
            self.shared = tf.keras.layers.Embedding(
                config.vocab_size,
                config.d_model,
                name="shared",
                embeddings_initializer=get_initializer(self.config.initializer_factor),
            )
        # Additional attribute to specify the expected name scope of the layer (for loading/storing weights)
        self.shared.load_weight_prefix = "shared"

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = PromptTFT5MainLayer(encoder_config, self.shared, name="encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = PromptTFT5MainLayer(decoder_config, self.shared, name="decoder")

        if not config.tie_word_embeddings:
            lm_head_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor)
            self.lm_head = tf.keras.layers.Dense(
                config.vocab_size, use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # Update init weights as in flax

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.get_input_embeddings()
        else:
            # in a dense layer the kernel has a shape (last_dim, units), for us (dim, num_tokens)
            # value has a shape (num_tokens, dim) then needs to be transposed
            return tf.transpose(self.lm_head.kernel)

    def set_output_embeddings(self, value):
        if self.config.tie_word_embeddings:
            self.set_input_embeddings(value)
        else:
            lm_head_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=self.config.initializer_factor)
            self.lm_head = tf.keras.layers.Dense(
                shape_list(value)[0], use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # Update init weights as in flax
            # in a dense layer the kernel has a shape (last_dim, units), for us (dim, num_tokens)
            # value has a shape (num_tokens, dim) then needs to be transposed
            transposed_value = tf.transpose(value)
            self.lm_head.kernel = transposed_value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @unpack_inputs
    def call(
            self,
            input_ids: Optional[TFModelInputType] = None,
            attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            decoder_input_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
            decoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            decoder_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            encoder_outputs: Optional[Union[np.ndarray, tf.Tensor]] = None,
            past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
            inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
            decoder_inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
            labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            training: Optional[bool] = False,
    ) -> Union[Tuple, TFSeq2SeqLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> inputs = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="tf").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="tf").input_ids
        >>> outputs = model(inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> inputs = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(inputs)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you
        ```"""
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = decoder_outputs[0]

        # T5v1.1 does not tie output word embeddings and thus does not require downscaling
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            logits = tf.matmul(sequence_output, self.shared.weights, transpose_b=True)
        else:
            logits = self.lm_head(sequence_output)

        logits = tf.cast(logits, tf.float32)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        past = decoder_outputs[1] if use_cache else None
        if not return_dict:
            if past_key_values is not None:
                decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        # If the user passed a tuple for encoder_outputs, we wrap it in a TFBaseModelOutput when return_dict=True
        elif isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
            hidden_states = None
            attentions = None
            idx = 0
            if output_hidden_states:
                idx += 1
                hidden_states = encoder_outputs[idx]
            if output_attentions:
                idx += 1
                attentions = encoder_outputs[idx]

            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=hidden_states,
                attentions=attentions,
            )

        return TFSeq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values[1:]) if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": None,  # needs to be passed to make Keras.layer.__call__ happy
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return self._shift_right(labels)


class PETLSoftPrompt(TFPromptT5ForConditionalGeneration, abc.ABC):
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
        output = self(x, training=False)

        # tf.Gradient.Tape() is not set here as we do dont want gradient calculations
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        # Track the loss here
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


class FullFineTune(TFT5ForConditionalGeneration):
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

        # Apply the calculated gradient
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
        output = self(x, training=False)

        # tf.Gradient.Tape() is not set here as we do nto want gradient calculations
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        # Track the loss here
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


def model_history_to_dlog(logger, history, model_name):
    """

    Args:
        logger: Logger object
        history: Model fit history
        model_name: Name of model to summarize

    Returns:
    """
    strng = f'Model {model_name} history:'
    logger.info(strng)

    logger.info(f'iteration,loss,validation loss,accuracy,validation accuracy')
    for iteration, (loss, val_loss, accuracy, val_accuracy) in \
            enumerate(zip(history['loss'], history['val_loss'],  history['accuracy'], history['val_accuracy'])):
        logger.info(f'{iteration + 1},{loss},{val_loss},{accuracy},{val_accuracy}')


def _model_structure_to_dlog(logger, model, model_name):
    """

    Args:
        logger: Logger object
        model: Model to summarize
        model_name: Name of model to summarize

    Returns:
    """
    strng = f'Model {model_name} summary:'
    logger.info(strng)

    stringlist = []
    model.summary(print_fn=lambda st: stringlist.append(st))
    for strng in stringlist:
        logger.info(strng)


def get_model(which_model, checkpoint, debug, optimizer_params, logger=None, ):
    """

    Args:
        which_model: Which model to use, FullFineTune or SoftPrompt or ...
        checkpoint: Which model checkpoint to use
        optimizer_params: Optimizer parameters

        debug: If debug is True then model is run in eager model otherwise in graph mode
        logger: Logger for logging progress
    Returns:
    """

    # Parameters other than the 'algo' are meant to go into the
    params = optimizer_params['params']
    if optimizer_params['algo'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(**params)
    elif optimizer_params['algo'] == 'adafactor':
        optimizer = Adafactor(**params)
    elif optimizer_params['algo'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(**params)
    else:
        raise KeyError(f'Unsupported optimizer algo type {optimizer_params["algo"]}')

    if which_model.lower() in ['sp', 'soft_prompt', 'softprompt', 'soft', 'petl']:
        logger.info(f'Loading PETLSoftPrompt model')

        model_name = 'PETLSoftPrompt'

        # Create a model instance
        model = PETLSoftPrompt.from_pretrained(checkpoint)

        # This makes the embedding layer non-trainable
        # The layer is called shared because it is shared between the encoder and decoder
        model.shared.trainable = False

        # We want the soft prompt to be trainable but all other weights must not be trainable
        for b in model.encoder.block:
            b.trainable = False
        model.encoder.final_layer_norm.trainable = False

        # We don't want any trainable parameters in the decode layer
        model.layers[2].trainable = False
        _model_structure_to_dlog(logger, model, model_name='PETLSoftPrompt')

    elif which_model.lower() in ['full', 'full_fine_tune', 'fullfinetune', 'fft']:
        logger.info(f'Loading FullFineTune model')
        model_name = 'FullFineTune'

        # Create a model instance
        model = FullFineTune.from_pretrained(checkpoint)

        # Save the model structure to the datalog
        _model_structure_to_dlog(logger, model, model_name='FullFineTune')
    else:
        raise KeyError(f'Model {which_model} is not supported')

    # Compile the model with Categorical accuracy metric
    model.compile(optimizer=optimizer,
                  metrics=SelectiveSparseTopKCategoricalAccuracy(name='accuracy', k=1),
                  run_eagerly=debug)
    return model, model_name
