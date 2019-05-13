import collections

import six
import tensorflow as tf

import utils
from process.feature_extractor import FeatureExtractor


class DictionaryFeatureExtractor(FeatureExtractor):
    def __init__(self, dictionary_file):
        self.dictionary = self.load_dict(dictionary_file)

    def extract(self, tokens):
        raise NotImplementedError()

    def restore(self, ids):
        return ids

    @staticmethod
    def load_dict(dictionary_files):
        """Loads a vocabulary file into a dictionary."""
        dictionary = collections.OrderedDict()
        dictionary_files = dictionary_files.split(",")
        for dictionary_file in dictionary_files:
            if not str.isspace(dictionary_file):
                with tf.gfile.GFile(dictionary_file, "r") as reader:
                    while True:
                        token = utils.convert_to_unicode(reader.readline())
                        if not token:
                            break
                        token = token.strip().split(" ")
                        if len(token) == 2:
                            dictionary[token[0]] = token[1]
                        else:
                            dictionary[token[0]] = 1
        return dictionary


class DefaultDictionaryFeatureExtractor(DictionaryFeatureExtractor):
    def __init__(self, dictionary_file, min_word_len, max_word_len):
        if not max_word_len > min_word_len:
            raise ValueError("min word length should smaller than max word length")
        self.max_word_len = max_word_len
        self.min_word_len = min_word_len
        self.dim = 2 * (max_word_len - min_word_len + 1)
        if six.PY3:
            super().__init__(dictionary_file)
        else:
            super(DictionaryFeatureExtractor, self).__init__(dictionary_file)

    def extract(self, tokens):
        result = []
        for i in range(len(tokens)):
            # fw
            word_tag = []
            for l in range(self.max_word_len - 1, self.min_word_len - 2, -1):
                if (i - l) < 0:
                    word_tag.append(0)
                    continue
                word = ''.join(tokens[i - l:i + 1])
                if word in self.dictionary:
                    word_tag.append(self.dictionary[word])
                else:
                    word_tag.append(0)
            # bw
            for l in range(self.min_word_len - 1, self.max_word_len):
                if (i + l) >= len(tokens):
                    word_tag.append(0)
                    continue
                word = ''.join(tokens[i:i + l + 1])
                if word in self.dictionary:
                    word_tag.append(self.dictionary[word])
                else:
                    word_tag.append(0)
            result.append(word_tag)
        return result

