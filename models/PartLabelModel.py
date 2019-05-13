# -*- coding: utf-8 -*-
import tensorflow as tf
from .SegmentModel import SegmentModel
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf
from models import model_utils
import numpy as np


class PartLabelModel(SegmentModel):
    """
    PartLabelModel 模型
    """

    def __init__(self, config, features, dropout_keep_prob, init_embeddings=None):
        """PartLabelModel 的构造器.

        Args:
          config: `Config` 实例，用来描述模型.
          features: dict 用来向模型传递输入，和训练时所用的标签
          dropout_keep_prob: float32 Tensor of shape [] 用来传递dropout保留率.
          init_embedding: (optional) float32 np.ndarray of shape [vocab_size, embedding_size].
          bi_embedding: (optional) float32 np.ndarray of shape [bigram_size, embedding_size].

        Raises:
          ValueError: 在config中没有设置multi-tag
        """

        super(PartLabelModel).__init__()
        input_ids = features["input_ids"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.input_ids = input_ids
        self.label_ids = label_ids
        self.seq_length = seq_length

        x, batch_size, feat_size = model_utils.input_embedding(
            input_ids, config, init_embeddings=init_embeddings)
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

        with tf.variable_scope('noise_correct'):
            pure_noise_matrix = tf.Variable(config.noise_matrix,
                                            dtype=tf.float32, name='noise_matrix', trainable=False)
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
                # 计算每一位置上的候选标签数量
                candidate_label_num = tf.reduce_sum(self.label_ids, axis=2)

                ## 对全标签数据使用CRF计算负对数损失
                gt_bool = tf.cast(self.label_ids, dtype=tf.bool)
                # 将弱标签数据的长度设置为0，以屏蔽对弱标签数据的求值
                full_label_data = tf.equal(tf.reduce_max(candidate_label_num, axis=-1), 1)
                full_label_seq_len = tf.where(full_label_data, self.seq_length, tf.zeros_like(self.seq_length))

                self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
                    scores, gt_bool, full_label_seq_len, transition_param)
                nll_loss = - self.log_likelihood

                ## 对弱标签数据使用点积损失
                gt_float = tf.cast(self.label_ids, dtype=tf.float32)
                prob = tf.nn.softmax(scores, axis=-1)
                # 获取弱标签数据的掩码
                partial_label_mask = tf.cast(
                    tf.logical_and(candidate_label_num > 1, candidate_label_num < config.num_classes),
                    dtype=tf.float32)

                partial_label_factor = 1.0 / (1e-12 + tf.reduce_sum(partial_label_mask, axis=-1))
                if config.log_dot_loss:
                    # 使用对数约束
                    dot_loss = -partial_label_factor * tf.reduce_sum(
                        partial_label_mask * tf.log(tf.clip_by_value(
                            tf.einsum("bld, bld->bl", gt_float, tf.einsum("ji, blj->bli", noise_matrix, prob)),
                            clip_value_min=1e-16, clip_value_max=1)), axis=-1)
                else:
                    # 使用线性约束
                    dot_loss = partial_label_factor * tf.reduce_sum(
                        partial_label_mask * (1 - tf.clip_by_value(
                            tf.einsum("bld, bld->bl", gt_float, tf.einsum("ji, blj->bli", noise_matrix, prob)),
                            clip_value_min=0, clip_value_max=1)), axis=-1)

            else:
                raise ValueError("PartLabelModel request multi-tag")

            self.loss = tf.reduce_mean(nll_loss + dot_loss)

        # ugly but useful
        # with tf.variable_scope('loss'):
        #     # crf
        #     if config.multitag:
        #         prob = tf.nn.softmax(scores, axis=-1)
        #         candidate_label_num = tf.reduce_sum(self.label_ids, axis=2)
        #         full_label_data = tf.equal(tf.reduce_max(candidate_label_num, axis=-1), 1)
        #         self.label_ids = tf.cast(self.label_ids, dtype=tf.bool)
        #         # with tf.control_dependencies([tf.print(self.label_ids)]):
        #         full_label_seq_len = tf.where(full_label_data, self.seq_length, tf.zeros_like(self.seq_length))
        #         self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
        #             scores, self.label_ids, full_label_seq_len, transition_param)
        #         gt = tf.cast(self.label_ids, dtype=tf.float32)
        #
        #         nll_loss = - self.log_likelihood
        #
        #         # part_label_data = tf.reduce_max(candidate_label_num, axis=-1) > 1
        #         # inside_mask = tf.cast(tf.equal(candidate_label_num, 1),
        #         #                       dtype=tf.float32)
        #         # i_l0_norm = tf.cast(part_label_data, dtype=tf.float32) / (1e-12 + tf.reduce_sum(inside_mask, axis=-1))
        #         # inside_matrix = tf.get_variable("inside_rate", [config.num_classes, config.num_classes])
        #         # inside_matrix = tf.nn.softmax(inside_matrix, axis=0)
        #
        #         # inside_loss = -i_l0_norm * tf.reduce_sum(
        #         #     inside_mask * tf.einsum("bld, bld->bl", gt, tf.log(tf.clip_by_value(
        #         #         tf.einsum("ji, blj->bli", inside_matrix, prob), clip_value_min=1e-16, clip_value_max=1))
        #         #                             ), axis=-1)
        #
        #         part_label_mask = tf.cast(
        #             tf.logical_and(candidate_label_num > 1, candidate_label_num < config.num_classes),
        #             dtype=tf.float32)
        #
        #         j_l0_norm = 1.0 / (1e-12 + tf.reduce_sum(part_label_mask, axis=-1))
        #         if config.log_dot_loss:
        #             ## log dot loss
        #             dot_loss = -j_l0_norm * tf.reduce_sum(
        #                 part_label_mask * tf.log(tf.clip_by_value(
        #                     tf.einsum("bld, bld->bl", gt, tf.einsum("ji, blj->bli", noise_matrix, prob)),
        #                     clip_value_min=1e-16, clip_value_max=1)), axis=-1)
        #         else:
        #             ## dot loss
        #             dot_loss = j_l0_norm * tf.reduce_sum(
        #                 part_label_mask * (1 - tf.clip_by_value(
        #                     tf.einsum("bld, bld->bl", gt, tf.einsum("ji, blj->bli", noise_matrix, prob)),
        #                     clip_value_min=0, clip_value_max=1)), axis=-1)
        #
        #     else:
        #         raise ValueError("PartLabelModel request multi-tag")
        #
        #     # with tf.control_dependencies([tf.print(dot_loss)]):
        #     # with tf.control_dependencies([tf.verify_tensor_all_finite(dot_loss, ""), tf.assert_greater_equal(dot_loss, 0.0)]):
        #     self.loss = tf.reduce_mean(nll_loss + dot_loss)
        #     # self.loss = tf.reduce_mean(nll_loss + dot_loss + config.inside_factor * inside_loss)
