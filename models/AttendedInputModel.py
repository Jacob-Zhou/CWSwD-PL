# -*- coding: utf-8 -*-
import tensorflow as tf

from models.SegmentModel import SegmentModel
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf
from models import model_utils
import numpy as np


class AttendedInputModel(SegmentModel):
    '''
    Model 1
    Concating outputs of two parallel Bi-LSTMs which take feature vectors and uni_embedding vectors
    as inputs respectively.
    '''

    def __init__(self, config, features, dropout_keep_prob, init_embeddings=None):

        super(AttendedInputModel).__init__()
        input_ids = features["input_ids"]
        input_dicts = features["input_dicts"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.label_ids = label_ids
        self.dict = input_dicts
        self.seq_length = seq_length

        dict_shape = model_utils.get_shape_list(input_dicts, expected_rank=3)
        self.dict_dim = dict_shape[2]

        x, self.batch_size, feat_size = model_utils.input_embedding(
            input_ids, config, init_embeddings=init_embeddings)

        # with tf.variable_scope('dict'):
        #     self.dict = tf.cast(self.dict, dtype=tf.float32)
        #     (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
        #         cell_bw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
        #         inputs=self.dict,
        #         sequence_length=self.seq_length,
        #         dtype=tf.float32
        #     )
        #     dict_output = tf.concat([forward_output, backword_output], axis=2)

        dict_output = tf.cast(self.dict, dtype=tf.float32)

        with tf.variable_scope('input_attention'):
            input_attention = layers.fully_connected(
                inputs=dict_output,
                num_outputs=feat_size,
                activation_fn=tf.sigmoid
            )

            input_bias = layers.fully_connected(
                inputs=dict_output,
                num_outputs=feat_size,
                activation_fn=tf.sigmoid
            )
            # [B, L, F] * [B, L, F, E] -> [B, L, F, E]
            input_attention = tf.expand_dims(input_attention, -1)
            attend_input = tf.multiply(x, input_attention) + tf.expand_dims(input_bias, axis=-1)
            attend_input = tf.reshape(attend_input, [self.batch_size, -1, feat_size * config.embedding_size])
            attend_input = tf.nn.dropout(attend_input, dropout_keep_prob)

        with tf.variable_scope('character'):
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                cell_bw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                inputs=attend_input,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            output = tf.concat([forward_output, backword_output], axis=2)

        with tf.variable_scope('output'):
            output = tf.concat([dict_output, output], axis=2)
            scores = layers.fully_connected(
                inputs=output,
                num_outputs=config.num_classes,
                activation_fn=None
            )
            transition_param = tf.get_variable("transitions", [config.num_classes, config.num_classes])
            self.prediction, _ = crf.crf_decode(scores, transition_param, self.seq_length)

        # with tf.variable_scope('loss'):
        #     # crf
        #     if config.multitag:
        #         self.label_ids = tf.cast(self.label_ids, dtype=tf.bool)
        #         self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
        #             scores, self.label_ids, self.seq_length, transition_param)
        #     else:
        #         self.log_likelihood, _ = crf.crf_log_likelihood(
        #             scores, self.label_ids, self.seq_length, transition_param)

        #     self.loss = tf.reduce_mean(-self.log_likelihood)

        with tf.variable_scope('noise_correct'):
            pure_noise_matrix = tf.Variable(config.noise_matrix, dtype=tf.float32, name='noise_matrix', trainable=False)
            tf.logging.info(f"\n{config.noise_matrix}")

            if config.fix_noise:
                noise_matrix = pure_noise_matrix
            else:
                eye_matrix = tf.Variable(np.eye(4), dtype=tf.float32, name='eye_matrix', trainable=False)
                rate = tf.Variable(np.ones([4, 1]), dtype=tf.float32, name='rate')
                norm_rate = tf.sigmoid(rate)
                noise_matrix = tf.broadcast_to(norm_rate, [4, 4]) * pure_noise_matrix + \
                            tf.broadcast_to((1 - norm_rate), [4, 4]) * eye_matrix

        with tf.variable_scope('loss'):
            # crf
            if config.multitag:
                prob = tf.nn.softmax(scores, axis=-1)
                candidate_label_num = tf.reduce_sum(self.label_ids, axis=2)
                full_label_data = tf.equal(tf.reduce_max(candidate_label_num, axis=-1), 1)
                self.label_ids = tf.cast(self.label_ids, dtype=tf.bool)
                full_label_seq_len = tf.where(full_label_data, self.seq_length, tf.zeros_like(self.seq_length))
                self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
                    scores, self.label_ids, full_label_seq_len, transition_param)
                gt = tf.cast(self.label_ids, dtype=tf.float32)

                nll_loss = - self.log_likelihood

                part_label_mask = tf.cast(
                    tf.logical_and(candidate_label_num > 1, candidate_label_num < config.num_classes),
                    dtype=tf.float32)

                j_l0_norm = 1.0 / (1e-12 + tf.reduce_sum(part_label_mask, axis=-1))
                if config.log_dot_loss:
                    ## log dot loss
                    dot_loss = -j_l0_norm * tf.reduce_sum(
                        part_label_mask * tf.log(tf.clip_by_value(
                            tf.einsum("bld, bld->bl", gt, tf.einsum("ji, blj->bli", noise_matrix, prob)),
                            clip_value_min=1e-16, clip_value_max=1)), axis=-1)
                else:
                    ## dot loss
                    dot_loss = j_l0_norm * tf.reduce_sum(
                        part_label_mask * (1 - tf.clip_by_value(
                            tf.einsum("bld, bld->bl", gt, tf.einsum("ji, blj->bli", noise_matrix, prob)),
                            clip_value_min=0, clip_value_max=1)), axis=-1)

            else:
                raise ValueError("PartLabelModel request multi-tag")

            self.loss = tf.reduce_mean(nll_loss + 0.01 * dot_loss)

