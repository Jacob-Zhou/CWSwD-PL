import codecs
import os
import re

import opencc
import six
import tensorflow as tf

import process.process_utils
import utils
from process.process_utils import strQ2B
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


class DataProcessor(object):
    def __init__(self):
        self.label_dim = 1

    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir=None, data=None, example_type=None, **kwargs):
        """Gets a collection of `InputExample`s for example_type."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the map of labels for this data set."""
        raise NotImplementedError()

    @staticmethod
    def _create_examples(lines, set_type):
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return f.readlines()


class CWSDataProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        if six.PY3:
            super().__init__()
        else:
            super(DataProcessor, self).__init__()
        self.language = "zh"
        self.label_dim = 1

    def get_examples(self, data_dir=None, data=None, example_type=None, **kwargs):
        if data_dir is not None:
            if example_type == "train":
                return self._create_examples(
                    self._read_file(os.path.join(data_dir, "train")), "train")
            elif example_type == "dev":
                return self._create_examples(
                    self._read_file(os.path.join(data_dir, "dev")), "dev")
            elif example_type == "test":
                if "domain" in kwargs:
                    domain = kwargs["domain"]
                    return self._create_examples(
                        self._read_file(os.path.join(data_dir, f"{domain}_test")), f"{domain}_test")
                else:
                    return self._create_examples(
                        self._read_file(os.path.join(data_dir, "test")), "test")
        elif data is not None:
            if example_type is None or example_type == "single":
                return list(self._create_examples([data], ""))[0]
            elif example_type == "multi":
                return self._create_examples(data, "")
        else:
            raise ValueError("one of data_dir or data should be NOT NONE")

        if 'key_error' in kwargs:
            raise ValueError(kwargs['key_error'])
        else:
            raise ValueError("CWSData only support train, dev and test")

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
                    yield InputExample(guid="", text=text, labels=labels)
                text = []
                labels = []
            else:
                text.append(char_info[0].strip())
                labels.append(char_info[2].strip())

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

    def evaluate(self, y_pred, y):
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

    def segment(self, x, y, output_dir, output_file='result.txt'):
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


class MultiTagDataProcessor(CWSDataProcessor):

    def __init__(self):
        if six.PY3:
            super().__init__()
        else:
            super(CWSDataProcessor, self).__init__()
        self.language = "zh"
        self.label_dim = 4

    def get_pl_examples(self, data_dir, domain):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, f"{domain}_domain")), f"{domain}_domain")

    def get_examples(self, data_dir=None, data=None, example_type=None, **kwargs):
        if data_dir is not None and example_type == "pl":
            domain = kwargs["domain"]
            return self._create_examples(
                self._read_file(os.path.join(data_dir, f"{domain}_domain")), f"{domain}_domain")
        else:
            kwargs['key_error'] = "CWSData only support train, dev, test and pl"
            return super().get_examples(data_dir=data_dir, data=data, example_type=example_type, **kwargs)

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

    def get_break_ids(self):
        return [2, 3]


class PlainLineDataProcessor(MultiTagDataProcessor):
    def __init__(self):
        if six.PY3:
            super().__init__()
        else:
            super(MultiTagDataProcessor, self).__init__()
        self.language = "zh"
        self.label_dim = 4

    def get_examples(self, data_dir=None, data=None, example_type=None, **kwargs):
        if data_dir is not None and "file_name" in kwargs:
            file_name = kwargs["file_name"]
            return self._create_examples(codecs.open(os.path.join(data_dir, file_name), encoding="UTF-8"))
        else:
            kwargs['key_error'] = f"don't support {example_type}"
            return super().get_examples(data_dir=data_dir, data=data, example_type=example_type, **kwargs)

    @staticmethod
    def _create_examples(lines, set_type=None):
        """Creates examples for the training and dev sets."""
        # re_ENUM = re.compile(r"([-.a-zA-Z0-9]+)")
        re_ENUM = re.compile(r'(([-–+])?\d+(([.·])\d+)?%?|([0-9_.·]*[A-Za-z]+[0-9_.·]*)+)')
        converter = opencc.OpenCC('t2s')

        def _labels_words(p_text_segment):
            inside_tokens = []
            inside_labels = []
            for segment in p_text_segment:
                hyper_tokens = segment.split()
                segment_tokens = []
                for hyper_token in hyper_tokens:
                    hyper_token = hyper_token.strip()
                    if len(hyper_token) > 0:
                        is_chinese = False
                        for c in hyper_token:
                            if process.process_utils.is_cjk_char(ord(c)):
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

            return inside_tokens, inside_labels

        for (i, line) in enumerate(lines):
            # Only the test set has a header
            line = convert_to_unicode(line.strip())
            text = str.lower(process.process_utils.strQ2B(line))
            text = converter.convert(text)
            text = re_ENUM.sub(" \\1 ", text)
            text_segment = text.split("☃")
            tokens, labels = _labels_words(text_segment)
            o_text = re.sub(r"\s|☃", "", line)
            offset = 0
            o_tokens = []
            for token in tokens:
                o_tokens.append(o_text[offset: offset + len(token)])
                offset += len(token)
            yield InputExample(guid=o_tokens, text=tokens, labels=labels)
