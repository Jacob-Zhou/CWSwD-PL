# -*- coding: utf-8 -*-
import tensorflow as tf

from models.SegmentModel import SegmentModel
from tensorflow.contrib import layers
from tensorflow.contrib import crf
from .supercell import HyperLSTMCell
from models import model_utils


class DictHyperModel(SegmentModel):
    '''
    Model 1
    Concating outputs of two parallel Bi-LSTMs which take feature vectors and uni_embedding vectors
    as inputs respectively.
    '''

    def __init__(self, config, features, dropout_keep_prob, init_embeddings=None):

        super(DictHyperModel).__init__()
        input_ids = features["input_ids"]
        input_dicts = features["input_dicts"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.label_ids = label_ids
        self.dict = input_dicts
        self.seq_length = seq_length

        x, batch_size, feat_size = model_utils.input_embedding(
            input_ids, config, init_embeddings=init_embeddings)
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
