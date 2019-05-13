import copy
import json

import numpy as np
import gensim

import tensorflow as tf
import six


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


class Config(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BaselineConfig` from a Python dictionary of parameters."""
        config = cls()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))