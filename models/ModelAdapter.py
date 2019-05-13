import tensorflow as tf

import utils
from models import model_utils


class ModelAdapter:
    def __init__(self, segment_model, dim_info, config, init_checkpoint, tokenizer, learning_rate, init_embedding=None):
        uni_embedding = None
        bi_embedding = None
        if init_embedding is not None:
            uni_embedding = utils.get_embedding(init_embedding, tokenizer.vocab, config.embedding_size)
            if "bigram_vocab" in tokenizer.__dict__:
                bi_embedding = utils.get_embedding(init_embedding, tokenizer.bigram_vocab, config.embedding_size)

        self.input_ids = tf.placeholder(dtype=tf.int64,
                                        shape=[None, None, dim_info.feature_dims['input_ids']], name='input_ids')
        self.input_dicts = tf.placeholder(dtype=tf.int64,
                                          shape=[None, None, dim_info.feature_dims['input_dicts']], name='input_dicts')
        if dim_info.label_dim == 1:
            self.label_ids = tf.placeholder(dtype=tf.int64, shape=[None, None], name='label_ids')
        else:
            self.label_ids = tf.placeholder(dtype=tf.int64, shape=[None, None, dim_info.label_dim], name='label_ids')
        self.seq_length = tf.placeholder(dtype=tf.int64, shape=[None], name='seq_length')

        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")

        features = {
            "input_ids": self.input_ids,
            "input_dicts": self.input_dicts,
            "label_ids": self.label_ids,
            "seq_length": self.seq_length
        }

        self.model = segment_model(config, features, self.dropout_keep_prob,
                                   init_embeddings={"uni_embedding": uni_embedding, "bi_embedding": bi_embedding})

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = model_utils.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            utils.variable_summaries(var)
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        (loss, label_ids, prediction, seq_length) = self.model.get_all_results()

        l2_reg_lamda = config.l2_reg_lamda
        clip = 5

        with tf.variable_scope('train_op'):
            self.lr_update = tf.assign(self.learning_rate, self.new_learning_rate)
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            if l2_reg_lamda > 0:
                l2_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tvars if (v.get_shape().ndims > 1 and "rate" not in v.name)])
                tf.logging.info("**** L2 Loss Variables ****")
                for var in tvars:
                    if var.get_shape().ndims > 1 and "rate" not in var.name:
                        tf.logging.info("  name = %s, shape = %s", var.name, var.shape)
                total_loss = loss + l2_reg_lamda * l2_loss
            else:
                total_loss = loss

            if config.clip_grad:
                grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), clip)
                train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            else:
                train_op = optimizer.minimize(total_loss, global_step=global_step)

        self.loss = loss
        self.total_loss = total_loss
        self.seq_length = seq_length
        self.prediction = prediction
        self.train_op = train_op

    def assign_lr(self, session, learning_rate):
        session.run(self.lr_update, feed_dict={self.new_learning_rate: learning_rate})