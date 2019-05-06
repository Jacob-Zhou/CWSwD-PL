# -*- coding: utf-8 -*-
import tensorflow as tf
from .ModelConfig import ModelConfig
from .SegmentModel import SegmentModel
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf
from models import model_utils


class AAAIModel(SegmentModel):
    '''
    Baseline models
    BiLSTM+CRF and Stacked BiLSTM+CRF
    '''

    def __init__(self, config, features, dropout_keep_prob, init_embedding=None, bi_embedding=None):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. rue for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int64 Tensor of shape [batch_size, seq_length, feat_size].
          label_ids: (optional) int64 Tensor of shape [batch_size, seq_length].
          seq_length: (optional) int64 Tensor of shape [batch_size].
          init_embedding: (optional)

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """

        super(AAAIModel).__init__()
        input_ids = features["input_ids"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.input_ids = input_ids
        self.label_ids = label_ids
        self.seq_length = seq_length

        x, batch_size, feat_size = model_utils.input_embedding(
            input_ids, config, init_embedding=init_embedding, bi_embedding=bi_embedding)
        x = tf.reshape(x, [batch_size, -1, feat_size * config.embedding_size])
        x = tf.nn.dropout(x, dropout_keep_prob)

        with tf.variable_scope('rnn'):
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                cell_bw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
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
            # self.prediction = tf.cast(tf.argmax(scores, axis=2), tf.int32)

        with tf.variable_scope('loss'):
            # crf
            candidate_label_num = tf.reduce_sum(self.label_ids, axis=2)
            exact_label_mask = tf.cast(tf.equal(candidate_label_num, 1), dtype=tf.float32)
            part_label_mask = tf.cast(tf.logical_and(candidate_label_num > 1, candidate_label_num < config.num_classes),
                                      dtype=tf.float32)

            # first_term_factor = 1.0 / (1e-12 + tf.reduce_sum(exact_label_mask, axis=-1))

            j_l0_norm = 1.0 / (1e-12 + tf.reduce_sum(
                part_label_mask * tf.cast(config.num_classes - candidate_label_num, dtype=tf.float32), axis=-1))

            gt = tf.cast(self.label_ids, dtype=tf.float32)
            masked_gt = tf.expand_dims(tf.cast(part_label_mask, dtype=tf.int64), -1) * \
                        tf.ones_like(self.label_ids) + self.label_ids
            masked_gt = tf.cast(masked_gt, dtype=tf.bool)

            self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
                scores, masked_gt, self.seq_length, transition_param)
            first_term = -self.log_likelihood

            tf.reduce_logsumexp(scores, axis=-1)
            second_term = -j_l0_norm * tf.reduce_sum(
                part_label_mask * tf.einsum("bld, bld->bl",
                                            1.0 - gt, tf.log(1.0 - tf.nn.sigmoid(scores))), axis=-1)
            self.loss = tf.reduce_mean(first_term + second_term)
