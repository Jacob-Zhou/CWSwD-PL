# -*- coding: utf-8 -*-
import tensorflow as tf

from models.SegmentModel import SegmentModel
from .ModelConfig import ModelConfig
from tensorflow.contrib import layers
from tensorflow.contrib import crf
from .supercell import HyperLSTMCell
from models import model_utils


class DictHyperConfig(ModelConfig):
    """Configuration for `DictConcatConfig`."""

    def __init__(self, vocab_size=8004, embedding_size=100, hyper_embedding_size=16, hidden_size=128,
                 dict_hidden_size=160, num_hidden_layers=1, bi_direction=True, multitag=False, rnn_cell="lstm",
                 l2_reg_lamda=0.0001, embedding_dropout_prob=0.2, hidden_dropout_prob=0.2, num_classes=4, **kw):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          bi_direction: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          rnn_cell: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          embedding_dropout_prob: The dropout ratio for the attention
            probabilities.
          embedding_size: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          num_classes: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          init_embedding: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        super(DictHyperConfig).__init__(**kw)
        self.multitag = multitag
        self.l2_reg_lamda = l2_reg_lamda
        self.dict_hidden_size = dict_hidden_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.rnn_cell = rnn_cell
        self.bi_direction = bi_direction
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.embedding_size = embedding_size
        self.hyper_embedding_size = hyper_embedding_size
        self.num_classes = num_classes


class DictHyperModel(SegmentModel):
    '''
    Model 1
    Concating outputs of two parallel Bi-LSTMs which take feature vectors and uni_embedding vectors
    as inputs respectively.
    '''

    def __init__(self, config, features, dropout_keep_prob, init_embedding=None, bi_embedding=None):

        super(DictHyperModel).__init__()
        input_ids = features["input_ids"]
        input_dicts = features["input_dicts"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.label_ids = label_ids
        self.dict = input_dicts
        self.seq_length = seq_length

        x, batch_size, feat_size = model_utils.input_embedding(
            input_ids, config, init_embedding=init_embedding, bi_embedding=bi_embedding)
        x = tf.reshape(x, [batch_size, -1, feat_size * config.embedding_size])
        x = tf.nn.dropout(x, dropout_keep_prob)

        def hyperlstm_cell(dim, input_main_dim, input_hyper_dim):
            cell = HyperLSTMCell(num_units=dim,
                                 input_main_dim=input_main_dim, input_hyper_dim=input_hyper_dim,
                                 forget_bias=1.0, use_recurrent_dropout=False,
                                 dropout_keep_prob=1.0, use_layer_norm=False, hyper_num_units=config.dict_hidden_size,
                                 hyper_embedding_size=config.hyper_embedding_size, hyper_use_recurrent_dropout=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
            return cell

        with tf.variable_scope('hyper'):
            self.dict = tf.cast(self.dict, dtype=tf.float32)
            input_main_dim = model_utils.get_shape_list(x, expected_rank=3)[2]
            input_hyper_dim = model_utils.get_shape_list(self.dict, expected_rank=3)[2]
            x = tf.concat([x, self.dict], axis=2)
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=hyperlstm_cell(config.hidden_size, input_main_dim, input_hyper_dim),
                cell_bw=hyperlstm_cell(config.hidden_size, input_main_dim, input_hyper_dim),
                inputs=x,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            output = tf.concat([forward_output, backword_output], axis=2)

        with tf.variable_scope('output'):
            scores = layers.fully_connected(
                inputs=output,
                num_outputs=config.num_classes,
                activation_fn=None
            )
            transition_param = tf.get_variable("transitions", [config.num_classes, config.num_classes])
            self.prediction, _ = crf.crf_decode(scores, transition_param, self.seq_length)

        with tf.variable_scope('loss'):
            # crf
            if config.multitag:
                self.label_ids = tf.cast(self.label_ids, dtype=tf.bool)
                self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
                    scores, self.label_ids, self.seq_length, transition_param)
            else:
                self.log_likelihood, _ = crf.crf_log_likelihood(
                    scores, self.label_ids, self.seq_length, transition_param)
            self.loss = tf.reduce_mean(-self.log_likelihood)
