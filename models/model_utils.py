import collections
import re

import six
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_norm
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_multitag_sequence_score


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def lstm_cell(dim, dropout_keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(dim, name='basic_lstm_cell')
    cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    return cell


def multi_lstm_cell(hidden_size, num_hidden_layers, dropout_keep_prob):
    return tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell(hidden_size, dropout_keep_prob)for _ in range(num_hidden_layers)])


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def create_bigram(uni, window_size):
    # uui: [batch, len, window, embed]
    e = tf.eye(window_size)
    mask = tf.slice(e, [0, 0], [window_size, window_size - 1]) + tf.slice(e, [0, 1], [window_size, window_size - 1])
    return tf.einsum("blwe,wv->blve", uni, mask)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def crf_multitag_log_likelihood(inputs,
                                tag_bitmap,
                                sequence_lengths,
                                transition_params=None):
    """使用CRF计算多标签的负对数损失.

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] 用于输入CRF层的未归一化的发射概率.
      tag_bitmap: A [batch_size, max_seq_len, num_tags] 布尔值张量用来表示正确标签，正确的标签为1.
      sequence_lengths: A [batch_size] 每一个序列的真实长度.
      transition_params: A [num_tags, num_tags] 转移矩阵, 如果已经创建则进行输入.
    Returns:
      log_likelihood: A [batch_size] `Tensor` 包含每一个输入序列的负对数损失.
      transition_params: A [num_tags, num_tags] 转移矩阵. 如果有输入时，则返回输入值否则返回一个新建的.
    """

    # 当没有给出转移矩阵时，获取
    if transition_params is None:
        # 获取输入维度
        num_tags = inputs.get_shape()[2].value
        transition_params = tf.get_variable("transitions", [num_tags, num_tags])

    sequence_scores = crf_multitag_sequence_score(inputs, tag_bitmap, sequence_lengths,
                                                  transition_params)
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # 计算负对数损失
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def input_embedding(input_ids, config, init_embedding=None, bi_embedding=None):
    initializer = tf.random_uniform_initializer(-0.5, 0.5)
    # initializer = tf.truncated_normal_initializer(stddev=0.001)

    if config.traditional_template:
        # [B L D]
        input_shape = get_shape_list(input_ids, expected_rank=3)
        batch_size = input_shape[0]
        max_length = input_shape[1]
        window_size = input_shape[2]

        input_ids, input_bi_ids, type_ids, bi_type_ids, ident = tf.split(
            input_ids,
            num_or_size_splits=config.input_dim_info,
            axis=-1)

        if init_embedding is None:
            uni_embedding = tf.get_variable(shape=[config.vocab_size, config.embedding_size],
                                            dtype=tf.float32,
                                            name='uni_embedding',
                                            initializer=initializer)
        else:
            uni_embedding = tf.Variable(init_embedding, dtype=tf.float32, name='uni_embedding')

        if bi_embedding is None:
            bi_embedding = tf.get_variable(shape=[config.bigram_size, config.embedding_size],
                                           dtype=tf.float32,
                                           name='bi_embedding',
                                           initializer=initializer)
        else:
            bi_embedding = tf.Variable(bi_embedding, dtype=tf.float32, name='bi_embedding')

        type_embedding = tf.get_variable(shape=[config.type_size, config.embedding_size],
                                         dtype=tf.float32,
                                         name='type_embedding',
                                         initializer=initializer)

        bi_type_embedding = tf.get_variable(shape=[config.bigram_type_size, config.embedding_size],
                                            dtype=tf.float32,
                                            name='bi_type_embedding',
                                            initializer=initializer)

        ## mistake fix
        identical_embedding = tf.get_variable(shape=[1, config.embedding_size],
                                              dtype=tf.float32,
                                              name='identical_embedding',
                                              initializer=initializer)

        # identical_embedding = tf.concat([identical_embedding, tf.zeros_like(identical_embedding)], axis=0)
        ## mistake fix

        ## todo really fix
        # identical_embedding = tf.get_variable(shape=[2, config.embedding_size],
        #                                       dtype=tf.float32,
        #                                       name='identical_embedding',
        #                                       initializer=initializer)

        with tf.variable_scope('embedding'):
            uni_embedded = tf.nn.embedding_lookup(uni_embedding, input_ids)
            bi_embedded = tf.nn.embedding_lookup(bi_embedding, input_bi_ids)
            type_embedded = tf.nn.embedding_lookup(type_embedding, type_ids)
            bi_type_embedded = tf.nn.embedding_lookup(bi_type_embedding, bi_type_ids)
            identical_embedded = tf.nn.embedding_lookup(identical_embedding, ident)

            x = tf.concat([uni_embedded, bi_embedded, type_embedded, bi_type_embedded, identical_embedded], axis=-2)
            feat_size = window_size

    else:
        input_shape = get_shape_list(input_ids, expected_rank=3)
        batch_size = input_shape[0]
        max_length = input_shape[1]
        window_size = input_shape[2]

        if init_embedding is None:
            embedding = tf.get_variable(shape=[config.vocab_size, config.embedding_size],
                                        dtype=tf.float32,
                                        name='uni_embedding',
                                        initializer=initializer)
        else:
            embedding = tf.Variable(init_embedding, dtype=tf.float32, name='uni_embedding')

        with tf.variable_scope('uni_embedding'):
            x = tf.nn.embedding_lookup(embedding, input_ids)
            feat_size = window_size

    return x, batch_size, feat_size
