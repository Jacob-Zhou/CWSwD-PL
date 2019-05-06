import codecs
import collections
from typing import *
import os

import opencc
import six
import tensorflow as tf

import preprocess
from preprocess import strQ2B, get_type
import tokenization
import numpy as np
import re

import utils
from utils import convert_to_unicode


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class TypedInputExample(InputExample):
    def __init__(self, guid, text, types, labels=None):
        if six.PY3:
            super().__init__(guid, text, labels)
        else:
            super(InputExample, self).__init__(guid, text, labels)
        self.types = types


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_dicts, label_ids, seq_length):
        self.input_ids = input_ids
        self.input_dicts = input_dicts
        self.seq_length = seq_length
        self.label_ids = label_ids


def convert_single_example(ex_index, example: InputExample,
                           tokenizer, label_map, dict_builder=None, typed_features=False):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # label_map = {"B": 0, "M": 1, "E": 2, "S": 3}

    # tokens_raw = tokenizer.tokenize(example.text)
    tokens_raw = example.text
    labels_raw = example.labels

    # Account for [CLS] and [SEP] with "- 2"

    # The convention in BERT is:
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The uni_embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # uni_embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    label_ids = []
    types = []
    # if not isinstance(tokenizer, tokenization.TrTeWinBiTokenizer):
    assert len(tokens_raw) == len(labels_raw)

    for token, label in zip(tokens_raw, labels_raw):
        tokens.append(token)
        label_ids.append(label_map[label])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # else:
    #     assert isinstance(example, TypedInputExample), "please use PlInputExample"
    #     types_raw = example.types
    #
    #     seq_length = len(tokens_raw)
    #     assert seq_length == len(types_raw)
    #     assert seq_length == len(labels_raw)
    #
    #     for token, t_type, label in zip(tokens_raw, types_raw, labels_raw):
    #         tokens.append(token)
    #         types.append(t_type)
    #         label_ids.append(label_map[label])
    #
    #     input_ids = tokenizer.convert_tokens_to_ids((tokens, types))

    if dict_builder is None:
        input_dicts = np.zeros_like(tokens_raw, dtype=np.int64)
    else:
        input_dicts = dict_builder.extract(tokens)
    seq_length = len(tokens)
    assert seq_length == len(input_ids)
    assert seq_length == len(input_dicts)
    assert seq_length == len(label_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.

    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid:        %s" % example.guid)
        tf.logging.info("tokens:      %s" % " ".join(
            [utils.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids:   %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_types: %s" % " ".join([str(x) for x in types]))
        tf.logging.info("input_dicts: %s" % " ".join([str(x) for x in input_dicts]))
        tf.logging.info("labels:      %s" % " ".join([str(x) for x in example.labels]))
        tf.logging.info("labels_ids:  %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_dicts=input_dicts,
        label_ids=label_ids,
        seq_length=seq_length)
    return feature


def convert_examples_to_features(
        examples, tokenizer, label_map, dict_builder=None, pl_features=False):

    features = []
    examples = list(examples)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, tokenizer, label_map, dict_builder, pl_features)

        features.append((feature.input_ids, feature.input_dicts, feature.label_ids, feature.seq_length))

    return features


def file_based_convert_examples_to_features(
        examples, tokenizer, label_map, output_file, dict_builder=None, pl_features=False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        # 12836239
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d" % ex_index)

        feature = convert_single_example(ex_index, example, tokenizer, label_map, dict_builder, pl_features)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        plain_input_ids = np.array(feature.input_ids).reshape([-1])
        features["input_ids"] = create_int_feature(plain_input_ids)
        plain_input_dicts = np.array(feature.input_dicts).reshape([-1])
        features["input_dicts"] = create_int_feature(plain_input_dicts)
        features["seq_length"] = create_int_feature([feature.seq_length])
        plain_label_ids = np.array(feature.label_ids).reshape([-1])
        features["label_ids"] = create_int_feature(plain_label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


class DataProcessor(object):
    def __init__(self):
        self.label_dim = 1

    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, domain):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the map of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return f.readlines()


class CWSProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        if six.PY3:
            super().__init__()
        else:
            super(DataProcessor, self).__init__()
        self.language = "zh"
        self.label_dim = 1

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "dev")), "dev")

    def get_test_examples(self, data_dir, domain):
        """See base class."""
        if domain is None:
            return self._create_examples(
                self._read_file(os.path.join(data_dir, "test")), "test")
        else:
            return self._create_examples(
                self._read_file(os.path.join(data_dir, f"{domain}_test")), f"{domain}_test")

    def get_labels(self):
        """See base class."""
        return {"B": 0, "M": 1, "E": 2, "S": 3}

    def get_break_ids(self):
        return [2, 3]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        tf.logging.info(f"creating {set_type} examples")
        text = []
        types = []
        labels = []
        for line in lines:
            # Only the test set has a header
            o_line = line
            line = utils.convert_to_unicode(strQ2B(line.strip()))
            char_info = line.split()
            if len(char_info) != 3:
                if len(char_info) != 0:
                    raise ValueError(o_line)
                if len(text) != 0:
                    yield TypedInputExample(guid="", text=text, types=types, labels=labels)
                text = []
                labels = []
                types = []
            else:
                text.append(char_info[0].strip())
                types.append(get_type(char_info[0].strip()))
                # types.append(char_info[1].strip())
                labels.append(char_info[2].strip())

    # @staticmethod
    # def examples_constructor(lines):

    @staticmethod
    def _labels_words(text):
        def word2label(w):
            if len(w) == 1:
                return ["S"]
            if len(w) == 2:
                return ["B", "E"]
            label = ["B"]
            for i in range(1, len(w) - 1):
                label.append("M")
            label.append("E")
            return label

        words = text.split()
        labels = []
        for word in words:
            labels += word2label(word)
        return "".join(labels)

    def evaluate_word_PRF(self, y_pred, y):
        import itertools
        y_pred = list(itertools.chain.from_iterable(y_pred))
        y = list(itertools.chain.from_iterable(y))
        assert len(y_pred) == len(y)
        cor_num = 0
        break_ids = self.get_break_ids()
        yp_word_num = 0
        yt_word_num = 0
        for i in break_ids:
            yp_word_num += y_pred.count(i)
            yt_word_num += y.count(i)
        # yp_word_num = y_pred.count(2) + y_pred.count(3)
        # yt_word_num = y.count(2) + y.count(3)
        start = 0
        for i in range(len(y)):
            if y[i] in break_ids:
                flag = True
                for j in range(start, i + 1):
                    if y[j] != y_pred[j]:
                        flag = False
                        break
                if flag:
                    cor_num += 1
                start = i + 1

        P = cor_num / float(yp_word_num)
        R = cor_num / float(yt_word_num)
        F = 2 * P * R / (P + R)
        return P, R, F

    def convert_word_segmentation(self, x, y, output_dir, output_file='result.txt'):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, output_file)
        f = codecs.open(output_file, 'w', encoding='utf-8')
        break_ids = self.get_break_ids()
        for i in range(len(x)):
            sentence = []
            for j in range(len(x[i])):
                if y[i][j] in break_ids:
                    sentence.append(x[i][j])
                    sentence.append("  ")
                else:
                    sentence.append(x[i][j])
            f.write(''.join(sentence).strip() + '\n')
        f.close()


class MultiTagProcessor(CWSProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        if six.PY3:
            super().__init__()
        else:
            super(CWSProcessor, self).__init__()
        self.language = "zh"
        self.label_dim = 4

    def get_pl_examples(self, data_dir, domain):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, f"{domain}_domain")), f"{domain}_domain")

    def get_labels(self):
        """See base class."""
        class LabelMaker:
            def __getitem__(self, item):
                if not isinstance(item, str):
                    raise ValueError("Value can only be (B?M?E?S?|A)")
                if item == "A":
                    return [1, 1, 1, 1]
                label_set = {"B": 0, "M": 1, "E": 2, "S": 3}
                label_id = [0, 0, 0, 0]
                for l in item:
                    if l not in label_set:
                        raise ValueError()
                    label_id[label_set[l]] = 1
                return label_id

        return LabelMaker()
        # return {"B": [1, 0, 0, 0],
        #         "M": [0, 1, 0, 0],
        #         "E": [0, 0, 1, 0],
        #         "S": [0, 0, 0, 1],
        #         "BS": [1, 0, 0, 1],
        #         "ES": [0, 0, 1, 1],
        #         "BMS": [1, 1, 0, 1],
        #         "MES": [0, 1, 1, 1],
        #         "A": [1, 1, 1, 1]}

    def get_break_ids(self):
        return [2, 3]


class PlainLineProcessor(MultiTagProcessor):
    def __init__(self):
        if six.PY3:
            super().__init__()
        else:
            super(MultiTagProcessor, self).__init__()
        self.language = "zh"
        self.label_dim = 4

    def get_train_examples(self, data_dir):
        """See base class."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """See base class."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, domain):
        """See base class."""
        raise NotImplementedError()

    def get_examples(self, data_dir, file_name, with_label=True):
        return self._create_examples(codecs.open(os.path.join(data_dir, file_name),
                                     encoding="UTF-8"))

    def get_single_example(self, sentence):
        return self._create_examples([sentence])

    def get_multi_example(self, sentences):
        return self._create_examples(sentences)

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        # re_ENUM = re.compile(r"([-.a-zA-Z0-9]+)")
        re_ENUM = re.compile(r'(([-–+])?\d+(([.·])\d+)?%?|([0-9_.·]*[A-Za-z]+[0-9_.·]*)+)')
        converter = opencc.OpenCC('t2s')

        def _labels_words(p_text_segment):
            inside_tokens = []
            inside_labels = []
            inside_types = []
            for segment in p_text_segment:
                hyper_tokens = segment.split()
                segment_tokens = []
                for hyper_token in hyper_tokens:
                    hyper_token = hyper_token.strip()
                    if len(hyper_token) > 0:
                        is_chinese = False
                        for c in hyper_token:
                            if preprocess.is_cjk_char(ord(c)):
                                is_chinese = True
                                break
                        if is_chinese:
                            segment_tokens.extend(list(hyper_token))
                        else:
                            segment_tokens.append(hyper_token)

                inside_tokens.extend(segment_tokens)
                if len(segment_tokens) == 1:
                    inside_labels.extend(["A"])
                elif len(segment_tokens) > 1:
                    inside_labels.extend(["BS"] + ["A"] * (len(segment_tokens) - 2) + ["ES"])
                inside_types.extend(list(map(preprocess.get_type, segment_tokens)))

            return inside_tokens, inside_labels, inside_types

        for (i, line) in enumerate(lines):
            # Only the test set has a header
            line = convert_to_unicode(line.strip())
            text = str.lower(preprocess.strQ2B(line))
            text = converter.convert(text)
            text = re_ENUM.sub(" \\1 ", text)
            text_segment = text.split("☃")
            tokens, labels, types = _labels_words(text_segment)
            o_text = re.sub(r"\s|☃", "", line)
            offset = 0
            o_tokens = []
            for token in tokens:
                o_tokens.append(o_text[offset: offset + len(token)])
                offset += len(token)
            yield TypedInputExample(guid=o_tokens, text=tokens, types=types, labels=labels)


class BiLabelProcessor(CWSProcessor):
    label_dim = 1

    def get_labels(self):
        """See base class."""
        return {"N": 0, "E": 1}

    def get_break_ids(self):
        return [1]

    @staticmethod
    def _labels_words(text):
        def word2label(w):
            if len(w) == 1:
                return ["E"]
            label = []
            for i in range(len(w) - 1):
                label.append("N")
            label.append("E")
            return label

        words = text.split()
        labels = []
        for word in words:
            labels += word2label(word)
        return "".join(labels)

    def evaluate_word_PRF(self, y_pred, y):
        import itertools
        y_pred = list(itertools.chain.from_iterable(y_pred))
        y = list(itertools.chain.from_iterable(y))
        assert len(y_pred) == len(y)
        cor_num = 0
        break_ids = self.get_break_ids()
        yp_word_num = 0
        yt_word_num = 0
        for i in break_ids:
            yp_word_num += y_pred.count(i)
            yt_word_num += y.count(i)
        # yp_word_num = y_pred.count(2) + y_pred.count(3)
        # yt_word_num = y.count(2) + y.count(3)
        start = 0
        len_y = len(y)
        for i in range(len_y - 1):
            if y_pred[i] == 1 or y_pred[i] == 3:
                if y_pred[i + 1] == 1:
                    y_pred[i + 1] = 3
                else:
                    y_pred[i + 1] = 2

            if y[i] == 1 or y[i] == 3:
                if y[i + 1] == 1:
                    y[i + 1] = 3
                else:
                    y[i + 1] = 2

        for i in range(len_y):
            if y[i] == 1 or y[i] == 3:
                flag = True
                for j in range(start, i + 1):
                    if y[j] != y_pred[j]:
                        flag = False
                        break
                if flag:
                    cor_num += 1
                start = i + 1

        P = cor_num / float(yp_word_num)
        R = cor_num / float(yt_word_num)
        F = 2 * P * R / (P + R)
        return P, R, F


