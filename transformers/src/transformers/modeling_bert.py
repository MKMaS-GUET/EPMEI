# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

import logging
import math
import os
import random
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.nn.parameter import Parameter
from .activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer
import torch.nn.functional as F
import numpy as np
logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        try:
            self.use_full_layer = config.use_full_layer
        except:
            self.use_full_layer = -1

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            full_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if i == self.use_full_layer:
                attention_mask = full_attention_mask

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            full_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )
        if full_attention_mask is not None:
            if full_attention_mask.dim() == 2:
                full_attention_mask = full_attention_mask[:, None, None, :]
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            full_attention_mask=full_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            next_sentence_label=None,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.


    Examples::

        from transformers import BertTokenizer, BertForPreTraining
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        prediction_scores, seq_relationship_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
                                                                 2:
                                                                 ]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            next_sentence_label=None,
    ):
        r"""
        next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`next_sentence_label` is provided):
            Next sequence prediction (classification) loss.
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForNextSentencePrediction
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        seq_relationship_scores = outputs[0]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
        # seq_length = input_ids.size(2)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
        # logits = logits.view(-1, 36)

        # if labels is not None:
        #     loss_fct = BCEWithLogitsLoss()
        #     labels = labels.view(-1, 36)
        #     loss = loss_fct(logits, labels)
        #     return loss, logits
        # else:
        #     return logits


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
        # logits = logits.view(-1, 36)

        # if labels is not None:
        #     loss_fct = BCEWithLogitsLoss()
        #     labels = labels.view(-1, 36)
        #     loss = loss_fct(logits, labels)
        #     return loss, logits
        # else:
        #     return logits


@add_start_docstrings(
    """Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor`` of shape ``(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForMultipleChoice
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]

        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, classification_scores = outputs[:2]

        """
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # # Only keep active parts of the loss
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_labels = labels.view(-1)[active_loss]
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        assert answer == "a nice puppet"

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class BertForTriviaQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTriviaQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        # assert logits.ndim == 2
        # assert target.ndim == 2
        # assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer
        bsz = logits.shape[0]

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = -10000.0  # float('-inf')

        # each batch is one example
        gathered_logits = gathered_logits.view(bsz, -1)
        logits = logits.view(bsz, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute
        # return loss[~torch.isinf(loss)].sum()
        return loss.mean()  # loss.sum() / len(loss)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_masks=None,
    ):
        bsz = input_ids.shape[0]
        max_segment = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_logits = start_logits.view(bsz, max_segment, -1)  # (bsz, max_segment, seq_length)
        end_logits = end_logits.view(bsz, max_segment, -1)  # (bsz, max_segment, seq_length)

        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
            end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)

            total_loss = (start_loss + end_loss) / 2

            outputs = (total_loss,) + outputs

        return outputs


class BertForQuestionAnsweringHotpotSeg(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringHotpotSeg, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.sent_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.sent_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        # assert logits.ndim == 2
        # assert target.ndim == 2
        # assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer
        bsz = logits.shape[0]

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = -10000.0  # float('-inf')

        # each batch is one example
        gathered_logits = gathered_logits.view(bsz, -1)
        logits = logits.view(bsz, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute
        # return loss[~torch.isinf(loss)].sum()
        return loss.mean()  # loss.sum() / len(loss)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_masks=None,
            sent_start_mapping=None,
            sent_end_mapping=None,
            sent_labels=None,
    ):
        bsz = input_ids.shape[0]
        max_segment = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        sent_start_mapping = sent_start_mapping.view(bsz * max_segment, -1, sent_start_mapping.size(
            -1)) if sent_start_mapping is not None else None
        sent_end_mapping = sent_end_mapping.view(bsz * max_segment, -1,
                                                 sent_end_mapping.size(-1)) if sent_end_mapping is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,

        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_logits = start_logits.view(bsz, max_segment, -1)  # (bsz, max_segment, seq_length)
        end_logits = end_logits.view(bsz, max_segment, -1)  # (bsz, max_segment, seq_length)

        start_rep = torch.matmul(sent_start_mapping, sequence_output)
        end_rep = torch.matmul(sent_end_mapping, sequence_output)
        sent_rep = torch.cat([start_rep, end_rep], dim=-1)

        sent_logits = gelu(self.sent_linear(sent_rep))
        sent_logits = self.sent_classifier(sent_logits).squeeze(-1)

        outputs = (start_logits, end_logits, sent_logits) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
            end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            sent_loss = loss_fct(sent_logits.view(-1, 2), sent_labels.view(-1))

            total_loss = (start_loss + end_loss) / 2 + sent_loss

            outputs = (total_loss,) + outputs

        return outputs


class BertForWikihop(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForWikihop, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            cand_positions=None,
            answer_index=None,
            sent_start_mapping=None,
            sent_end_mapping=None,
            sent_labels=None,

    ):
        # bsz = input_ids.shape[0]

        # input_ids = input_ids.view(-1, input_ids.size(-1))
        # attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output).squeeze(-1)

        ignore_index = -1
        target = cand_positions
        target = target.unsqueeze(0).expand(logits.size(0), -1)
        target_mask = target == ignore_index
        masked_target = target * (1 - target_mask.long())
        gathered_logits = logits.gather(dim=-1, index=masked_target)
        gathered_logits[target_mask] = -10000.0  # float('-inf')

        gathered_logits = torch.mean(gathered_logits, dim=0)
        gathered_logits = gathered_logits.view(1, -1)

        outputs = (gathered_logits,) + outputs[2:]

        if answer_index is not None:
            answer_index = answer_index.view(-1)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(gathered_logits, answer_index)

            outputs = (loss,) + outputs

        return outputs


class BertForWikihopMulti(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForWikihopMulti, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            cand_positions=None,
            answer_index=None,
            instance_mask=None,
    ):
        bsz = input_ids.shape[0]
        max_segment = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output).squeeze(-1)  # (bsz*max_segment, seq_length)
        logits = logits.view(bsz, max_segment, -1)  # (bsz, max_segment, seq_length)

        ignore_index = -1
        target = cand_positions  # (bsz, 79)
        target = target.unsqueeze(1).expand(-1, max_segment, -1)  # (bsz, max_segment, 79)
        target_mask = (target == ignore_index)
        masked_target = target * (1 - target_mask.long())
        gathered_logits = logits.gather(dim=-1, index=masked_target)
        gathered_logits[target_mask] = -10000.0  # (bsz, max_segment, 79)
        instance_mask = instance_mask.to(gathered_logits)
        gathered_logits = torch.sum(gathered_logits * instance_mask.unsqueeze(-1), dim=1)
        gathered_logits = gathered_logits / torch.sum(instance_mask, dim=1).unsqueeze(-1)

        outputs = (gathered_logits,) + outputs[2:]

        if answer_index is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(gathered_logits, answer_index)

            outputs = (loss,) + outputs

        return outputs


class BertForQuestionAnsweringHotpot(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringHotpot, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_classifier = nn.Linear(config.hidden_size, 3)

        self.sent_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.sent_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            switch_labels=None,
            sent_start_mapping=None,
            sent_end_mapping=None,
            sent_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        switch_logits = self.qa_classifier(torch.max(sequence_output, 1)[0])
        # switch_logits = self.qa_classifier(pooled_output)

        start_rep = torch.matmul(sent_start_mapping, sequence_output)
        end_rep = torch.matmul(sent_end_mapping, sequence_output)
        sent_rep = torch.cat([start_rep, end_rep], dim=-1)

        sent_logits = gelu(self.sent_linear(sent_rep))
        sent_logits = self.sent_classifier(sent_logits).squeeze(-1)

        outputs = (start_logits, end_logits, switch_logits, sent_logits) + outputs[2:]

        if switch_labels is not None:
            ignore_index = -1

            loss_fct = CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

            start_losses = [loss_fct(start_logits, _start_positions) for _start_positions in
                            torch.unbind(start_positions, dim=1)]
            end_losses = [loss_fct(end_logits, _end_positions) for _end_positions in torch.unbind(end_positions, dim=1)]

            start_loss = sum(start_losses)
            end_loss = sum(end_losses)

            ge = torch.sum(start_positions >= 0, dim=-1).to(start_loss)

            start_loss = start_loss / ge
            end_loss = end_loss / ge

            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            switch_loss = loss_fct(switch_logits, switch_labels)

            sent_loss = loss_fct(sent_logits.view(-1, 2), sent_labels.view(-1))

            loss = torch.mean((start_loss + end_loss) / 2) + switch_loss + sent_loss

            outputs = (loss,) + outputs

        return outputs


class BertForACEBothSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.re_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothOneDropoutSpanSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 4, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 4, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        m2_span_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m2_span_end_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m2_span_start_states, m2_span_end_states],
                                   dim=2)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


# prior
# class GCN(nn.Module):
#     """ A GCN/Contextualized GCN module operated on dependency graphs. """
#
#     def __init__(self, emb_dim=768, num_layers=2,gcn_dropout=0.1):
#         super(GCN, self).__init__()
#         self.layers = num_layers
#         self.emb_dim = emb_dim
#         self.out_dim = emb_dim
#         # gcn layer
#         self.W = nn.ModuleList()
#         for layer in range(self.layers):
#             input_dim = self.emb_dim if layer == 0 else self.out_dim
#             self.W.append(nn.Linear(input_dim, input_dim))
#         self.gcn_drop = nn.Dropout(gcn_dropout)
#
#
#     def forward(self, adj, inputs):
#         #print(adj.size())
#         # gcn layer
#
#         denom = adj.sum(2).unsqueeze(2) + 1
#         mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
#         # print(denom.size(),adj.size(),inputs.size())
#         # # zero out adj for ablation
#         # if self.opt.get('no_adj', False):
#         #     adj = torch.zeros_like(adj)
#         # print(adj.size(),inputs.size())
#
#         # print(adj.size(),denom.size())
#         # print(inputs.size())
#         inputs = inputs.double()
#         for l in range(self.layers):
#             Ax = adj.bmm(inputs)
#             Ax = Ax.double()
#             AxW = self.W[l](Ax)
#             AxW = Ax.double()
#             AxW = AxW + self.W[l](inputs)  # self loop
#             AxW = AxW / denom
#             gAxW = F.relu(AxW)
#             inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
#
#         return inputs,mask
class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features).double())
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features).double())
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # para1 = list(self.weight.named_parameters())
        # print(self.weight)
        text = text.double()
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            if self.bias.dtype != output.dtype:
                self.bias = Parameter(self.bias.to(output.dtype))
            return output + self.bias
        else:
            return output





class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        FocalLoss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 FocalLoss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retina net中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retina net中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)

            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[0].fill_(alpha)
            self.alpha[1:].fill_(1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        FocalLoss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1

        ###############################
        # 一、初始操作
        ###############################
        res = 0
        # 按照最后一个维度重新调整矩阵形状，因为最后一个维度是分类数
        batches = preds.size(0)
        for batch in range(batches):
            pred = preds[batch,:,:]
            count = len([x for x in labels[batch] if x != -1])
            pred =  pred[:count,:]
            pred = pred.view(-1, pred.size(-1))
            alpha = self.alpha.to(pred.device)

            ###############################
            # 二、计算预测概率Pt
            # focalLoss(pt) = -(1-pt)^γ * log(pt)
            ###############################

            # 将 preds 张量在第 1 个维度上执行 softmax 操作，过softmax之后的，就是pt
            pt = preds_softmax = F.softmax(pred, dim=1)
            # 交叉熵损失函数 CELoss(pt) = -log(pt)，这个pt，就是预估值，多分类是softmax后的概率值，二分类是sigmoid后的值
            # 在softmax后面接着log，这样算交叉熵只用前面加个负号
            log_pt = preds_logSoftmax = torch.log(pt)

            ###############################
            # 三、选真实标签对应的数据
            ###############################

            # labels.view(-1,1) 是将 labels 张量的形状调整为 (N, 1)
            # Ensure the labels are long, not float
            label = labels[batch,:]
            label = label[:count]
            labelsView = label.view(-1, 1).long()
            # 下面操作的目的就是选出真实标签对应的pt
            pt = pt.gather(1, labelsView)
            # 下面操作的目的就是选出真实标签对应的log_pt
            log_pt = log_pt.gather(1, labelsView)

            ###############################
            # 四、不带α的focal-loss
            ###############################

            # focalLoss(pt) = -(1-pt)^γ * log(pt)
            loss = -torch.mul(torch.pow((1 - pt), self.gamma), log_pt)

            ###############################
            # 五、带上α的focal-loss
            ###############################
            # labels.view(-1) 的作用是将 labels 张量的形状调整为一个一维张量
            label_flatten = labelsView.view(-1)
            # 因为算softmax的时候，会把所有的值的softmax都算出来，然而我们需要的只要真实标签的那些而已
            # 所以要进行取值操作
            # 整句话的作用就是alpha根据label值，取到每个分类对应的数值α
            alpha = alpha.gather(0, label_flatten)
            # 损失乘上α向量操作
            loss = torch.mul(alpha, loss.t())
            # 根据需求，看损失是求平均还是求和
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()

            res += loss
        if self.size_average:
            loss = res/batches
        else:
            loss = res
        return loss
class BertForACEBothOneDropoutSub(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels
        self.bert = BertModel(config)
        # self.gcn = GCN()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.args.use_span_width:

            self.ner_classifier = nn.Linear(config.hidden_size * 2+self.args.span_width_embedding_dim, self.num_ner_labels)
            self.re_classifier_m1 = nn.Linear(config.hidden_size * 2+self.args.span_width_embedding_dim, self.num_labels).float()
            self.re_classifier_m2 = nn.Linear(config.hidden_size * 2+self.args.span_width_embedding_dim, self.num_labels).float()
        else:
            self.ner_classifier = nn.Linear(config.hidden_size * 2,self.num_ner_labels)
            self.re_classifier_m1 = nn.Linear(config.hidden_size * 2,self.num_labels).float()
            self.re_classifier_m2 = nn.Linear(config.hidden_size * 2,self.num_labels).float()
        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

        self.norm_gcn = nn.LayerNorm(args.embedding_dim).double()

        self.gcn = nn.ModuleList(
            [GCN(args.embedding_dim, args.embedding_dim).to(args.device) for i in range(args.n_gcn)])



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            adj=None,
            sub_width_embedding=None,
            obj_width_embedding=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
        pool_out = outputs[1]


        if self.args.gcn:
            dep_sparse_out_previous = hidden_states
            for i in range(self.args.n_gcn):
                dep_sparse_out = self.gcn[i](dep_sparse_out_previous, adj)
                norm_input = dep_sparse_out_previous + dep_sparse_out
                norm_input = norm_input.double()
                dep_sparse_out = self.norm_gcn(norm_input)
                # dep_sparse_out = dep_sparse_out_previous + dep_sparse_out
                dep_sparse_out_previous = dep_sparse_out
            hidden_states = dep_sparse_out

        hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape  # bsz 训练batch
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        #=================
        if self.args.use_span_width:
            feature_vector = torch.cat([e1_hidden_states, e2_hidden_states,obj_width_embedding], dim=2)
        else:
            feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        feature_vector = feature_vector.float()
        ner_prediction_scores = self.ner_classifier(feature_vector)

        # 拿出了主语[s][/s]的hidden_states
        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]  # [s] 的hidden_states
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]  # [/s] 的 hidden_states

        # m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)  # 融合
        #==============
        if self.args.use_span_width:
            sub_width_embedding = torch.squeeze(sub_width_embedding,1)
            m1_states = torch.cat([m1_start_states, m1_end_states,sub_width_embedding], dim=-1)
        else:
            m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)


        m1_states = m1_states.float()
        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label  #主语在7个label(NIL,POS,NEU,NEG,对应的对称)上的分数
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label  #obj在 7个label上的分数
        re_prediction_scores = m1_scores.unsqueeze(1) + self.args.beta*m2_scores  # 每个 obj 的score 都加上了 sub的score
        # 我认为 opinion的词 在 对于情感的方面的预测的 重要程度应该大于 aspect 这里需不需要一个加权，但是由于 是sub-oriented, 后面还是会有以 opinion作为sub的时候

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # outputs[2:]是空的 # Add hidden states and attention if they are here

        if labels is not None:
            if not self.args.focalloss :
                loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
                re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
                loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
                ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))
            else:
                if self.args.data_dir.find('14lap') != -1:
                    loss_fn = FocalLoss(alpha=[8661/5735, 8661/1640, 8661/1034, 8661/552, 8661/1640,8661/1034,8661/552], gamma=2, num_classes=7)
                elif self.args.data_dir.find('14res') != -1:
                    loss_fn = FocalLoss(alpha=[16807/12131, 16807/3382, 16807/962, 16807/632, 16807/3382, 16807/962, 16807/632], gamma=2, num_classes=7)
                elif self.args.data_dir.find('15res') != -1:
                    loss_fn = FocalLoss(alpha=[6553/4527,6553/1566, 6553/410, 6553/350,6553/1566, 6553/410, 6553/350], gamma=2, num_classes=7)
                elif self.args.data_dir.find('16res') != -1:
                    loss_fn = FocalLoss(alpha=[8995/6207, 8995/2030, 8995/658, 8995/100, 8995/2030, 8995/658, 8995/100], gamma=2, num_classes=7)

                re_loss = loss_fn(re_prediction_scores, labels)
                loss_fct_ner = FocalLoss(alpha=0.2, gamma=2, num_classes=3)
                ner_loss = loss_fct_ner(ner_prediction_scores, ner_labels)

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothOneDropoutLeviPair(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier = nn.Linear(config.hidden_size * 4, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            m1_ner_labels=None,
            m2_ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 4

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len * 1: seq_len + ent_len * 2]
        e3_hidden_states = hidden_states[:, seq_len + ent_len * 2: seq_len + ent_len * 3]
        e4_hidden_states = hidden_states[:, seq_len + ent_len * 3: seq_len + ent_len * 4]

        m1_feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        m2_feature_vector = torch.cat([e3_hidden_states, e4_hidden_states], dim=2)
        feature_vector = torch.cat([m1_feature_vector, m2_feature_vector], dim=2)

        m1_ner_prediction_scores = self.ner_classifier(m1_feature_vector)
        m2_ner_prediction_scores = self.ner_classifier(m2_feature_vector)

        re_prediction_scores = self.re_classifier(feature_vector)  # bsz, ent_len, num_label

        outputs = (re_prediction_scores, m1_ner_prediction_scores, m2_ner_prediction_scores) + outputs[
                                                                                               2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            m1_ner_loss = loss_fct_ner(m1_ner_prediction_scores.view(-1, self.num_ner_labels), m1_ner_labels.view(-1))
            m2_ner_loss = loss_fct_ner(m2_ner_prediction_scores.view(-1, self.num_ner_labels), m2_ner_labels.view(-1))

            loss = re_loss + m1_ner_loss + m2_ner_loss
            outputs = (loss, re_loss, m1_ner_loss + m2_ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothOneDropout(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        # self.re_classifier_m1 = nn.Linear(config.hidden_size*2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        # m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m2_scores  # m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothOneDropoutNERSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.ner_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
            sub_ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier_m2(feature_vector)

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        ner_prediction_scores_m1 = self.ner_classifier_m1(m1_states)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))

            ner_prediction_scores = ner_prediction_scores.view(-1, self.num_ner_labels)
            ner_prediction_scores = torch.cat([ner_prediction_scores, ner_prediction_scores_m1], dim=0)
            ner_labels = torch.cat([ner_labels.view(-1), sub_ner_labels])

            ner_loss = loss_fct_ner(ner_prediction_scores, ner_labels)

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothLMSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.cls = BertOnlyMLMHeadTransform(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(self.cls(e1_hidden_states))
        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.re_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothOneDropoutLMSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.cls = BertOnlyMLMHeadTransform(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(self.cls(e1_hidden_states))
        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.re_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForMarkerQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.max_seq_length = config.max_seq_length
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - self.max_seq_length) // 2

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        logits = self.classifier(self.dropout(feature_vector))

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(logits))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BertForCorefSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        # self.num_ner_labels = 8

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)

        self.coref_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.coref_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.coref_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.coref_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        coref_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (coref_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(coref_prediction_scores))
            # loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            coref_loss = loss_fct_re(coref_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            # ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))

            # loss = re_loss + ner_loss
            outputs = (coref_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothOneDropoutSubNoNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(m1_states)  # bsz, num_label
        m2_scores = self.re_classifier_m2(feature_vector)  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = 0
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForACEBothSubNoNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.re_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = 0
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForQuestionAnsweringMultiAnswer(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2  # config.num_labels
        self.num_answers = 3  # config.num_answers

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_answers)
        self.point_outputs = nn.Linear(config.hidden_size, 1)

        head_num = config.num_attention_heads // 4

        self.coref_config = BertConfig(num_hidden_layers=1, num_attention_heads=head_num,
                                       hidden_size=config.hidden_size, intermediate_size=256 * head_num)
        self.origin_layer = BertLayer(self.coref_config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None, answer_masks=None, answer_nums=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output_0 = outputs[0]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sequence_output = self.origin_layer(sequence_output_0, extended_attention_mask)[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        switch_logits = self.qa_classifier(sequence_output[:, 0, :])

        outputs = (start_logits, end_logits, switch_logits) + outputs[2:]

        if start_positions is not None and end_positions is not None:

            if len(answer_nums.size()) > 1:
                answer_nums = answer_nums.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

            start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) \
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]
            end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                          for (_end_positions, _span_mask) \
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_masks, dim=1))]

            s_e_loss = sum(start_losses + end_losses)  # bsz
            switch_loss = loss_fct(switch_logits, answer_nums)

            total_loss = torch.mean(s_e_loss + switch_loss)  # + 0.1 * mention_loss

            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            example_L=None,
            # mention_pos=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            if example_L is None:
                loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores))
                ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores),
                                                reduce=False)
                bsz, num_predict, num_label = ner_prediction_scores.shape
                example_L = example_L.unsqueeze(1).expand((-1, num_predict))
                ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
                ner_loss = ner_loss / example_L.to(ner_loss).view(-1)

                # ner_loss = torch.mean(ner_loss, dim=0)
                ner_loss = torch.sum(ner_loss, dim=-1) / bsz

            outputs = (ner_loss,) + outputs

        return outputs


class BertForSpanNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)
        # seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        # ent_len = (tot_seq_len-seq_len) // 2

        # e1_hidden_states = hidden_states[:, seq_len:seq_len+ent_len]
        # e2_hidden_states = hidden_states[:, seq_len+ent_len: ]

        m1_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_end_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]

        feature_vector = torch.cat([m1_start_states, m1_end_states], dim=2)
        if not self.onedropout:
            feature_vector = self.dropout(feature_vector)
        ner_prediction_scores = self.ner_classifier(feature_vector)

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class BertForSpanMarkerNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.args = args

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if args.span_width_embedding_dim != 0:
            self.ner_classifier = nn.Linear(config.hidden_size * 5+args.span_width_embedding_dim, self.num_labels)
        else:
            self.ner_classifier = nn.Linear(config.hidden_size * 5, self.num_labels)
        # self.gcn = GCN()
        self.gcn = nn.ModuleList(
            [GCN(args.embedding_dim, args.embedding_dim).to(args.device) for i in range(args.n_gcn)])

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout
        self.norm_gcn = nn.LayerNorm(args.embedding_dim)
        self.init_weights()

        self.linear1 = nn.Linear(config.hidden_size * 5+args.span_width_embedding_dim, 150)
        self.linear2 = nn.Linear(150, 150)
        self.linear3 = nn.Linear(150, self.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4, inplace=False)

        def predict(self, ner_scores, spans, span_mask, metadata):
            # TODO: Make sure the iteration works in documents with a single sentence.
            # Zipping up and iterating iterates over the zeroth dimension of each tensor; this
            # corresponds to iterating over sentences.
            predictions = []
            zipped = zip(ner_scores, spans, span_mask, metadata)
            for ner_scores_sent, spans_sent, span_mask_sent, sentence in zipped:
                predicted_scores_raw, predicted_labels = ner_scores_sent.max(dim=1)
                softmax_scores = F.softmax(ner_scores_sent, dim=1)
                predicted_scores_softmax, _ = softmax_scores.max(dim=1)
                ix = (predicted_labels != 0) & span_mask_sent.bool()  # 预测nerlabel 不为none 且mask为true的index

                predictions_sent = []
                zip_pred = zip(
                    predicted_labels[ix],
                    predicted_scores_raw[ix],
                    predicted_scores_softmax[ix],
                    spans_sent[ix],
                )
                for label, label_score_raw, label_score_softmax, label_span in zip_pred:
                    label_str = self.vocab.get_token_from_index(
                        label.item(), self._active_namespace
                    )
                    span_start, span_end = label_span.tolist()
                    ner = [
                        span_start,
                        span_end,
                        label_str,
                        label_score_raw.item(),
                        label_score_softmax.item(),
                    ]
                    prediction = document.PredictedNER(ner, sentence, sentence_offsets=True)
                    predictions_sent.append(prediction)

                predictions.append(predictions_sent)

            return predictions

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            full_attention_mask=None,
            sparse_graph=None,
            span_width_embeddings=None,
            related_spans=None,
            spans=None,
            spans_masks=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]

        # dep_sparse_out_previous = hidden_states
        # for i in range(self.args.n_gcn):
        #     dep_sparse_out = self.gcn[i](dep_sparse_out_previous, sparse_graph)
        #     dep_sparse_out = self.norm_gcn(dep_sparse_out_previous + dep_sparse_out)
        #     # dep_sparse_out = dep_sparse_out_previous + dep_sparse_out
        #     dep_sparse_out_previous = dep_sparse_out
        # hidden_states = dep_sparse_out
        entity_ctx = self.get_token(hidden_states, input_ids, 101)
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]  # 拿出的是后面的部分 即marker的部分
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        m1_start_states = hidden_states[
            torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]  # 所有span 的开始representation  # 这里拿出的是 句子中的部分
        m1_end_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]  # end representation
        # ==============================
        # spans_embedding = torch.cat([m1_start_states, m1_end_states, span_width_embeddings], dim=2)
        # spans_embedding_dict = self.linear4width(spans_embedding)
        # spans_embedding = spans_embedding_dict
        features = torch.stack([m1_start_states, m1_end_states])
        spans_embedding = torch.max(features, dim=0, keepdim=True)[0]
        spans_embedding = torch.squeeze(spans_embedding)
        # =================================
        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states], dim=2)
        if self.args.span_width_embedding_dim != 0:
            feature_vector = torch.cat(
                [feature_vector, span_width_embeddings], dim=2)
        # =======
        feature_vector = torch.cat([entity_ctx.unsqueeze(1).repeat(1, feature_vector.shape[1], 1),
                                    feature_vector], dim=2)
        if not self.onedropout:
            feature_vector = self.dropout(feature_vector)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here




        if labels is not None:
            if not self.args.focalloss :
                loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores))
                ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            else:
                if self.args.data_dir.find('14lap') != -1:
                    loss_fct_ner = FocalLoss(alpha=[113624/(113624-2549), 113624/1281,113624/1268], gamma=2, num_classes=3)
                elif self.args.data_dir.find('14res') != -1:
                    loss_fct_ner = FocalLoss(alpha=[140166/(140166-4137),140166/2051,140166/2086], gamma=2, num_classes=3)
                # elif self.args.data_dir.find('15res') != -1:
                #     loss_fct_ner = FocalLoss(alpha=[54995/(54995-1803),54995/862,54995/941], gamma=2, num_classes=3)
                elif self.args.data_dir.find('15res') != -1:
                    loss_fct_ner = FocalLoss(alpha=[54995/(54995-1803),54995/1062,54995/1141], gamma=2, num_classes=3)
                elif self.args.data_dir.find('16res') != -1:
                    loss_fct_ner = FocalLoss(alpha=[80441/(80441-2505),80441/1198,80441/1307], gamma=2, num_classes=3)
                ner_loss = loss_fct_ner(ner_prediction_scores, labels)

            outputs = (ner_loss,) + outputs


        # if labels is not None:
        #     loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores))
        #     ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
        #     # if self.args.whether_kl_loss:
        #     #     kl_loss = self.shape_span_embedding(self.args, feature_vector, feature_vector, related_spans,
        #     #                                         spans_masks)
        #     #     print(kl_loss)
        #     #     ner_loss = ner_loss + self.args.kl_loss_weight * kl_loss
        #     outputs = (ner_loss,) + outputs


        return outputs
    def shape_span_embedding(self,args, p, q, pad_mask, span_mask):
        kl_loss = 0
        input_size = p.size()
        assert input_size == q.size()
        for i in range(input_size[0]):
            span_mask_index = torch.nonzero(span_mask[i, :]).squeeze()
            lucky_squence = random.choice(span_mask_index)
            P = p[i, lucky_squence, :]
            mask_index = torch.nonzero(pad_mask[i, lucky_squence, :])
            q_tensor = None
            for idx in mask_index:
                if idx == lucky_squence:
                    continue
                if q_tensor is None:
                    q_tensor = p[i, idx]
                else:
                    q_tensor = torch.cat((q_tensor, p[i, idx]), dim=0)
            if q_tensor is None:
                continue
            expan_P = P.expand_as(q_tensor)
            kl_loss += self.compute_kl_loss(args, expan_P, q_tensor)
        return kl_loss

    def compute_kl_loss(self,args, p, q, pad_mask=None,kl_loss_mode='KLLoss'):
        if kl_loss_mode == "KLLoss":
            p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
            q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

            if pad_mask is not None:
                p_loss.masked_fill(pad_mask, 0.)
                q_loss.masked_fill(pad_mask, 0.)
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()
            total_loss = math.log(1 + 5 / ((p_loss + q_loss) / 2))
        elif kl_loss_mode == "JSLoss":
            m = (p + q) / 2
            m_loss = 0.5 * F.kl_div(F.log_softmax(p, dim=-1), F.softmax(m, dim=-1), reduction="none") + 0.5 * F.kl_div(
                F.log_softmax(q, dim=-1), F.softmax(m, dim=-1), reduction="none")
            if pad_mask is not None:
                m_loss.masked_fill(pad_mask, 0.)
            m_loss = m_loss.sum()
            # test = -math.log(2*m_loss)-math.log(-2*m_loss+2)
            total_loss = 10 * (math.log(1 + 5 / m_loss))
        elif kl_loss_mode == "EMLoss":
            test = torch.square(p - q)
            em_loss = torch.sqrt(torch.sum(torch.square(p - q)))
            total_loss = math.log(1 + 5 / (em_loss))
        elif kl_loss_mode == "CSLoss":
            test = torch.cosine_similarity(p, q, dim=1)
            cs_loss = torch.sum(torch.cosine_similarity(p, q, dim=1))
            total_loss = math.log(1 + 5 / (cs_loss))
        else:
            total_loss = 0
            print('损失种类错误')
        return total_loss
    def get_token(self,h: torch.tensor, x: torch.tensor, token: int):
        """ Get specific token embedding (e.g. [CLS]) """
        emb_size = h.shape[-1]

        token_h = h.view(-1, emb_size)
        flat = x.contiguous().view(-1)

        # get contextualized embedding of given token
        token_h = token_h[flat == token, :]

        return token_h  # 拿出cls的token




class BertForSpanMarkerBiNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 4, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.onedropout = config.onedropout
        self.reduce_dim = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.blinear = nn.Bilinear(config.hidden_size, config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mention_pos=None,
            full_attention_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            full_attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        m1_start_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]]
        m1_end_states = hidden_states[torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]]

        m1 = torch.cat([e1_hidden_states, m1_start_states], dim=2)
        m2 = torch.cat([e2_hidden_states, m1_end_states], dim=2)

        feature_vector = torch.cat([m1, m2], dim=2)
        if not self.onedropout:
            feature_vector = self.dropout(feature_vector)
        ner_prediction_scores = self.ner_classifier(feature_vector)

        # m1 = self.dropout(self.reduce_dim(m1))
        # m2 = self.dropout(self.reduce_dim(m2))

        m1 = F.gelu(self.reduce_dim(m1))
        m2 = F.gelu(self.reduce_dim(m2))

        ner_prediction_scores_bilinear = self.blinear(m1, m2)

        ner_prediction_scores = ner_prediction_scores + ner_prediction_scores_bilinear

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class BertLMPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        return hidden_states


class BertOnlyMLMHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHeadTransform(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertForMaskedLMTransform(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHeadTransform(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        return outputs


class BertForLeftLMNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.cls = BertOnlyMLMHeadTransform(config)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            example_L=None,
            # mention_pos=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        # e2_hidden_states = hidden_states[:, seq_len+ent_len: ]

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        feature_vector = self.cls(e1_hidden_states)
        ner_prediction_scores = self.ner_classifier(feature_vector)

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            if example_L is None:
                loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores))
                ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores),
                                                reduce=False)
                bsz, num_predict, num_label = ner_prediction_scores.shape
                example_L = example_L.unsqueeze(1).expand((-1, num_predict))
                ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
                ner_loss = ner_loss / example_L.to(ner_loss).view(-1)

                # ner_loss = torch.mean(ner_loss, dim=0)
                ner_loss = torch.sum(ner_loss, dim=-1) / bsz
            outputs = (ner_loss,) + outputs

        return outputs


class BertForRightLMNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.cls = BertOnlyMLMHeadTransform(config)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            # mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        # e1_hidden_states = hidden_states[:, seq_len:seq_len+ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        # feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        feature_vector = self.cls(e2_hidden_states)
        ner_prediction_scores = self.ner_classifier(feature_vector)

        outputs = (ner_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(ner_prediction_scores))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            outputs = (ner_loss,) + outputs

        return outputs


class BertForEvent(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_ner_labels = 8
        self.num_trigger_labels = 34
        self.num_argument_labels = 23

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.trigger_classifier = nn.Linear(config.hidden_size * 2, self.num_trigger_labels)
        # self.augment_classifier = nn.Linear(config.hidden_size*2, self.num_argument_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            # labels=None,
            # mention_pos=None,
            ner_labels=None,
            trigger_labels=None,
            # argument_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))
        trigger_prediction_scores = self.trigger_classifier(self.dropout(feature_vector))
        # augment_prediction_scores = self.augment_classifier(self.dropout(feature_vector))

        outputs = (ner_prediction_scores, trigger_prediction_scores,) + outputs[
                                                                        2:]  # Add hidden states and attention if they are here

        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            ner_loss = loss_fct(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))
            trigger_loss = loss_fct(trigger_prediction_scores.view(-1, self.num_trigger_labels),
                                    trigger_labels.view(-1))
            # argument_loss = loss_fct(augment_prediction_scores.view(-1, self.num_argument_labels), argument_labels.view(-1))
            argument_loss = 0
            loss = ner_loss + trigger_loss  # + argument_loss

            outputs = (loss, ner_loss, trigger_loss, argument_loss) + outputs

        return outputs


class BertForEventArg(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_ner_labels = 8
        self.num_trigger_labels = 34
        self.num_argument_labels = 23

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.trigger_classifier = nn.Linear(config.hidden_size * 2, self.num_trigger_labels)
        self.argument_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_argument_labels)
        self.argument_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_argument_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            # labels=None,
            # mention_pos=None,
            sub_positions=None,
            ner_labels=None,
            trigger_labels=None,
            argument_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.argument_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.argument_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        argument_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        trigger_prediction_scores = self.trigger_classifier(self.dropout(feature_vector))

        # trigger_prediction_scores = self.trigger_classifier(self.dropout(m1_states))

        outputs = (ner_prediction_scores, trigger_prediction_scores, argument_prediction_scores) + outputs[
                                                                                                   2:]  # Add hidden states and attention if they are here

        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            ner_loss = loss_fct(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))
            trigger_loss = loss_fct(trigger_prediction_scores.view(-1, self.num_trigger_labels),
                                    trigger_labels.view(-1))
            argument_loss = loss_fct(argument_prediction_scores.view(-1, self.num_argument_labels),
                                     argument_labels.view(-1))

            loss = ner_loss + trigger_loss + argument_loss

            outputs = (loss, ner_loss, trigger_loss, argument_loss) + outputs

        return outputs


class BertForMarkerSEQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.max_seq_length = config.max_seq_length
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            start_positions=None,
            end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - self.max_seq_length) // 2

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)
        logits = self.classifier(self.dropout(feature_vector))

        qa_hidden_states = hidden_states[:, :seq_len]
        qa_logits = self.qa_outputs(qa_hidden_states)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            span_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2

            loss = span_loss + qa_loss
            outputs = (loss, span_loss, qa_loss) + outputs

        return outputs


class BertForTACRED(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.ner_classifier = nn.Linear(config.hidden_size*2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        # ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.re_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores), reduce=False)
            re_prediction_scores = re_prediction_scores.view(-1, self.num_labels)
            labels = labels.view(-1, labels.shape[-1])
            re_label_num = torch.sum(labels >= 0, dim=-1)  # .to(re_prediction_scores)
            re_tot = torch.sum(re_label_num > 0).to(re_prediction_scores)
            re_losses = [loss_fct_re(re_prediction_scores, label) for label in torch.unbind(labels, dim=-1)]
            re_loss = sum(re_losses)
            re_loss = torch.sum(re_loss) / re_tot

            ner_loss = 0
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForTACREDNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mentions=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sub_positions=None,
            labels=None,
            ner_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        e1_hidden_states = hidden_states[:, seq_len:seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len:]

        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(self.dropout(feature_vector))

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)

        m1_scores = self.re_classifier_m1(self.dropout(m1_states))  # bsz, num_label
        m2_scores = self.re_classifier_m2(self.dropout(feature_vector))  # bsz, ent_len, num_label
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[
                                                                  2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores), reduce=False)
            re_prediction_scores = re_prediction_scores.view(-1, self.num_labels)
            labels = labels.view(-1, labels.shape[-1])
            re_label_num = torch.sum(labels >= 0, dim=-1)  # .to(re_prediction_scores)
            re_tot = torch.sum(re_label_num > 0).to(re_prediction_scores)
            re_losses = [loss_fct_re(re_prediction_scores, label) for label in torch.unbind(labels, dim=-1)]
            re_loss = sum(re_losses)
            re_loss = torch.sum(re_loss) / re_tot

            loss_fct_ner = CrossEntropyLoss(ignore_index=-1, reduce=False)
            ner_prediction_scores = ner_prediction_scores.view(-1, self.num_ner_labels)
            ner_labels = ner_labels.view(-1, ner_labels.shape[-1])
            ner_label_num = torch.sum(ner_labels >= 0, dim=-1)  # .to(ner_prediction_scores)
            ner_tot = torch.sum(ner_label_num > 0).to(ner_prediction_scores)
            ner_losses = [loss_fct_ner(ner_prediction_scores, label) for label in torch.unbind(ner_labels, dim=-1)]
            ner_loss = sum(ner_losses)
            ner_loss = torch.sum(ner_loss) / ner_tot

            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
