import codecs

import numpy as np
import gensim
import os

import tensorflow as tf
import six
import random


class DimInfo:
    def __init__(self, input_dim, dict_dim, label_dim):
        self.label_dim = label_dim
        self.dict_dim = dict_dim
        self.input_dim = input_dim


def padding_features(features, dim_info, pad_id=0):
    input_ids, input_dicts, label_ids, seq_length = zip(*features)
    max_length = max(seq_length)

    batch_size = len(features)
    padded_input_ids = np.ones((batch_size, max_length, dim_info.input_dim), dtype=np.int32) * pad_id
    padded_input_dicts = np.ones((batch_size, max_length, dim_info.dict_dim), dtype=np.int32) * pad_id
    if dim_info.label_dim == 1:
        padded_label_ids = np.ones((batch_size, max_length), dtype=np.int32) * pad_id
    else:
        padded_label_ids = np.ones((batch_size, max_length, dim_info.label_dim), dtype=np.int32) * pad_id

    for i in range(batch_size):
        slen = seq_length[i]
        padded_input_ids[i, :slen] = input_ids[i]
        padded_input_dicts[i, :slen] = input_dicts[i]
        padded_label_ids[i, :slen] = label_ids[i]

    return padded_input_ids, padded_input_dicts, padded_label_ids, seq_length


def data_iterator(features, dim_info, batch_size=128, shuffle=True, pad_id=0, pl_features=None):
    if shuffle:
        # print(f"build-in: {random.random()}")
        random.shuffle(features)
        if pl_features is not None:
            random.shuffle(pl_features)
            features = features[:10000] + pl_features[:10000]
            # features = features + pl_features[:len(features)]
            random.shuffle(features)

    data_len = len(features)
    batch_len = (data_len // batch_size) + 1

    for i in range(batch_len):
        if i * batch_size < data_len:
            batch_features = features[i * batch_size:(i + 1) * batch_size]
            yield padding_features(batch_features, dim_info=dim_info, pad_id=pad_id)


def data_batch_base_mix_iterator(full_features, part_features, dim_info, batch_size=128,
                                 part_batch_factor=1, pad_id=0):
    random.shuffle(full_features)
    random.shuffle(part_features)
    full_features = full_features[:10000]
    part_features = part_features[:10000 * part_batch_factor]

    data_len = len(full_features)
    batch_len = (data_len // batch_size) + 1

    for i in range(batch_len):
        if i * batch_size < data_len:
            batch_full_features = full_features[i * batch_size:(i + 1) * batch_size]
            batch_part_features = part_features[i * (batch_size * part_batch_factor):
                                                (i + 1) * (batch_size * part_batch_factor)]
            if random.random() > 0.5:
                yield padding_features(batch_part_features, dim_info=dim_info, pad_id=pad_id)
                yield padding_features(batch_full_features, dim_info=dim_info, pad_id=pad_id)
            else:
                yield padding_features(batch_full_features, dim_info=dim_info, pad_id=pad_id)
                yield padding_features(batch_part_features, dim_info=dim_info, pad_id=pad_id)


def evaluate_word_PRF(y_pred, y):
    import itertools
    y_pred = list(itertools.chain.from_iterable(y_pred))
    y = list(itertools.chain.from_iterable(y))
    assert len(y_pred) == len(y)
    cor_num = 0
    yp_word_num = y_pred.count(2) + y_pred.count(3)
    yt_word_num = y.count(2) + y.count(3)
    start = 0
    for i in range(len(y)):
        if y[i] == 2 or y[i] == 3:
            flag = True
            for j in range(start, i + 1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i + 1

    assert yp_word_num != 0 and yt_word_num != 0
    P = cor_num / float(yp_word_num)
    R = cor_num / float(yt_word_num)
    F = 2 * P * R / (P + R)
    return P, R, F


def convert_word_segmentation(x, y, output_dir, output_file='result.txt'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = os.path.join(output_dir, output_file)
    f = codecs.open(output_file, 'w', encoding='utf-8')
    for i in range(len(x)):
        sentence = []
        for j in range(len(x[i])):
            if y[i][j] == 2 or y[i][j] == 3:
                sentence.append(x[i][j])
                sentence.append("  ")
            else:
                sentence.append(x[i][j])
        f.write(''.join(sentence).strip() + '\n')
    f.close()


# get pre-trained embeddings
def get_embedding(embedding_file, vocab, size=100):
    # print(f"numpy: {np.random.randn()}")
    init_embedding = np.zeros(shape=[len(vocab), size])
    pre_trained = gensim.models.KeyedVectors.load(embedding_file)
    pre_trained_vocab = set([w for w in pre_trained.wv.vocab.keys()])

    if "<UNK>" in vocab:
        unk = "<UNK>"
    elif "<BUNK>" in vocab:
        unk = "<BUNK>"
    else:
        unk = "U"

    def get_new_embedding():
        return np.random.uniform(-0.5, 0.5, size)
        # return np.random.normal(0, 0.001, size)

    init_embedding[vocab[unk]] = get_new_embedding()

    c = 0
    for word in vocab.keys():
        if word in pre_trained_vocab:
            init_embedding[vocab[word]] = pre_trained[word]
        else:
            if len(word) == 1 or len(word) > 2:
                c += 1
                init_embedding[vocab[word]] = get_new_embedding()
            else:
                e = 0
                for char in word:
                    e += pre_trained[char] if char in pre_trained_vocab else get_new_embedding()
                e /= len(word)
                init_embedding[vocab[word]] = e

    if "P" in vocab:
        init_embedding[vocab["P"]] = np.zeros(shape=size)
    else:
        init_embedding[vocab["<PAD>"]] = np.zeros(shape=size)
    tf.logging.info('oov character rate %f' % (float(c) / len(vocab)))
    return init_embedding


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var.name.split(":")[0] + '/summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


