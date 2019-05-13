from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import tensorflow as tf
import utils
from process.process_utils import is_cjk_char, is_chinese_char, is_punctuation, rENUM, rNUM, rENG
from process.feature_extractor import FeatureExtractor


class ContextFeatureExtractor(FeatureExtractor):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, unk_token="<UNK>"):
        self.vocab, self.inv_vocab = self.load_vocab(vocab_file)
        self.unk_token = unk_token
        self.dim = 1
        self.dim_detail = [1]
        self.size_info = {"vocab_size": len(self.vocab)}

    def extract(self, tokens):
        return self.extract_by_vocab(self.vocab, tokens, self.unk_token)

    def restore(self, ids):
        return self.extract_by_vocab(self.inv_vocab, ids)

    @staticmethod
    def extract_by_vocab(vocab, items, unk_token="<UNK>"):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            if item in vocab:
                output.append(vocab[item])
            else:
                output.append(vocab[unk_token])
        return output

    @staticmethod
    def load_vocab(vocab_files, preserve_token=None):
        """Loads a vocabulary file into a dictionary."""
        if preserve_token is None:
            preserve_token = []
        vocab = collections.OrderedDict()
        index = 0
        if preserve_token is not None:
            for token in preserve_token:
                vocab[token] = index
                index += 1
        vocab_files = vocab_files.split(",")
        for vocab_file in vocab_files:
            with tf.gfile.GFile(vocab_file, "r") as reader:
                while True:
                    token = utils.convert_to_unicode(reader.readline())
                    if not token:
                        break
                    token = token.strip()
                    if token not in vocab:
                        vocab[token] = index
                        index += 1
        inv_vocab = {v: k for k, v in vocab.items()}
        return vocab, inv_vocab


class WindowContextFeatureExtractor(ContextFeatureExtractor):
    def __init__(self, vocab_file, window_size):
        if six.PY3:
            super().__init__(vocab_file, "<UNK>")
        else:
            super(WindowContextFeatureExtractor, self).__init__(vocab_file, "<UNK>")
        self.window_size = window_size
        self.dim = window_size
        self.dim_detail = [window_size]

    def extract(self, tokens):
        """
        :param tokens:
        :return: windowed ids
        """
        fw = self.window_size // 2
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        unwind_ids = [self.vocab["<SOS>"]] * fw + self.extract_by_vocab(self.vocab, tokens, self.unk_token) + \
                     [self.vocab["<EOS>"]] * bw
        windowed_ids = [unwind_ids[i: i + self.window_size] for i in range(len(tokens))]
        return windowed_ids

    def restore(self, ids):
        c_inx = self.window_size // 2
        c_ids = [win[c_inx] for win in ids]
        return self.extract_by_vocab(self.inv_vocab, c_ids)


class WindowBigramContextFeatureExtractor(WindowContextFeatureExtractor):
    def __init__(self, vocab_file, bigram_file, window_size):
        if six.PY3:
            super().__init__(",".join([vocab_file, bigram_file]), window_size=window_size)
        else:
            super(WindowBigramContextFeatureExtractor, self).__init__(",".join([vocab_file, bigram_file]),
                                                                      window_size=window_size)
        self.dim = window_size * 2 - 1
        self.dim_detail = [window_size, window_size - 1]

    def extract(self, tokens):
        fw = self.window_size // 2
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        padded_tokens = ["<SOS>"] * fw + tokens + ["<EOS>"] * bw
        uni_ids = self.extract_by_vocab(self.vocab, padded_tokens, self.unk_token)
        bi_unk = "<UNK>" if "<BUNK>" not in self.vocab else "<BUNK>"
        bi_ids = self.extract_by_vocab(self.vocab,
                                       ["".join(padded_tokens[i: i + 2]) for i in range(len(padded_tokens) - 1)],
                                       bi_unk)
        wb_ids = [uni_ids[i: i + self.window_size] + bi_ids[i: i + self.window_size - 1] for i in range(len(tokens))]
        return wb_ids


class TrTeBiContextFeatureExtractor(ContextFeatureExtractor):
    def __init__(self, vocab_file, bigram_file, type_file, bigram_type_file):
        if six.PY3:
            super().__init__(",".join([vocab_file]))
        else:
            super(TrTeBiContextFeatureExtractor, self).__init__(",".join([vocab_file]))
        self.bigram_vocab, self.inv_bigram_vocab = self.load_vocab(bigram_file)

        self.type_vocab, self.inv_type_vocab = self.load_vocab(type_file)

        self.bigram_type_vocab, self.inv_bigram_type_vocab = self.load_vocab(bigram_type_file)

        self.dim = 5
        self.dim_detail = [1, 1, 1, 1, 1]
        self.size_info = {"vocab_size": len(self.vocab),
                          "bigram_size": len(self.bigram_vocab),
                          "type_size": len(self.type_vocab),
                          "bigram_type_size": len(self.bigram_type_vocab)}
        self.unk_token = "<UNK>"

    def extract(self, inputs):
        fw, bw = 1, 1
        # tokens, types = inputs
        tokens = inputs
        types = list(map(get_type, inputs))
        padded_tokens = ["<SOS>"] * fw + tokens + ["<EOS>"] * bw
        uni_ids = self.extract_by_vocab(self.vocab, padded_tokens, self.unk_token)
        bi_ids = self.extract_by_vocab(self.bigram_vocab,
                                       ["".join(padded_tokens[i: i + 2]) for i in range(len(padded_tokens) - 1)],
                                  "<BUNK>")

        padded_types = ["<SOS>"] * fw + types + ["<EOS>"] * bw
        uni_ty_ids = self.extract_by_vocab(self.type_vocab, padded_types, self.unk_token)
        bi_ty_ids = self.extract_by_vocab(self.bigram_type_vocab,
                                          ["".join(padded_types[i: i + 2]) for i in range(len(padded_types) - 1)],
                                          self.unk_token)

        ident = [1 if padded_tokens[i] == padded_tokens[i + 1] else 0 for i in range(len(padded_tokens) - 1)]
        wb_ids = [[uni_ids[i + 1]] + [bi_ids[i + 1]] +
                  [uni_ty_ids[i + 1]] + [bi_ty_ids[i + 1]] +
                  [ident[i + 1]] for i in range(len(tokens))
                  ]
        return wb_ids

    def restore(self, ids):
        c_inx = 0
        c_ids = [win[c_inx] for win in ids]
        return self.extract_by_vocab(self.inv_vocab, c_ids)


class TrTeWinBiContextFeatureExtractor(WindowContextFeatureExtractor):
    def __init__(self, vocab_file, bigram_file, type_file, bigram_type_file, window_size):
        if six.PY3:
            super().__init__(",".join([vocab_file]), window_size=window_size)
        else:
            super(TrTeWinBiContextFeatureExtractor, self).__init__(",".join([vocab_file]),
                                                                   window_size=window_size)
        self.bigram_vocab, self.inv_bigram_vocab = self.load_vocab(bigram_file)

        self.type_vocab, self.inv_type_vocab = self.load_vocab(type_file)

        self.bigram_type_vocab, self.inv_bigram_type_vocab = self.load_vocab(bigram_type_file)

        self.dim = window_size * 5 - 3
        self.dim_detail = [window_size, window_size - 1, window_size, window_size - 1, window_size - 1]
        self.size_info = {"vocab_size": len(self.vocab),
                          "bigram_size": len(self.bigram_vocab),
                          "type_size": len(self.type_vocab),
                          "bigram_type_size": len(self.bigram_type_vocab)}
        self.unk_token = "<UNK>"

    def extract(self, inputs):
        fw = self.window_size // 2
        # tokens, types = inputs
        tokens = inputs
        types = list(map(get_type, inputs))
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        padded_tokens = ["<SOS>"] * fw + tokens + ["<EOS>"] * bw
        uni_ids = self.extract_by_vocab(self.vocab, padded_tokens, self.unk_token)
        bi_ids = self.extract_by_vocab(self.bigram_vocab,
                                       ["".join(padded_tokens[i: i + 2]) for i in range(len(padded_tokens) - 1)],
                                  "<BUNK>")

        padded_types = ["<SOS>"] * fw + types + ["<EOS>"] * bw
        uni_ty_ids = self.extract_by_vocab(self.type_vocab, padded_types, self.unk_token)
        bi_ty_ids = self.extract_by_vocab(self.bigram_type_vocab,
                                          ["".join(padded_types[i: i + 2]) for i in range(len(padded_types) - 1)],
                                          self.unk_token)

        ident = [1 if padded_tokens[i] == padded_tokens[i + 1] else 0 for i in range(len(padded_tokens) - 1)]
        wb_ids = [uni_ids[i: i + self.window_size] + bi_ids[i: i + self.window_size - 1] +
                  uni_ty_ids[i: i + self.window_size] + bi_ty_ids[i: i + self.window_size - 1] +
                  ident[i: i + self.window_size - 1] for i in range(len(tokens))
                  ]
        return wb_ids


class WindowNgramContextFeatureExtractor(WindowContextFeatureExtractor):
    def __init__(self, vocab_file, ngram_file, window_size):
        if six.PY3:
            super().__init__(vocab_file, window_size=window_size)
        else:
            super(WindowNgramContextFeatureExtractor, self).__init__(vocab_file,
                                                                     window_size=window_size)
        assert window_size <= 15, "window size is too big, must small than 15"
        self.vocab, _ = self.load_vocab(",".join([vocab_file, ngram_file]),
                                        preserve_token=["P", "S", "E"] + ["U" + str(n) for n in range(15)])
        self.dim = (window_size * (window_size + 1)) // 2
        self.dim_detail = [i for i in reversed(range(1, window_size + 1))]

    def extract(self, tokens):
        fw = self.window_size // 2
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        padded_tokens = ["S"] * fw + tokens + ["E"] * bw
        unzipped_ids = [self.extract_by_vocab(self.vocab,
                                              ["".join(padded_tokens[i: i + 1 + n]) for i in range(len(padded_tokens) - n)], f"U{n - 1 if n > 0 else ''}"
                                              ) for n in range(self.window_size)]
        wb_ids = []
        for i in range(len(tokens)):
            window_ids = []
            for n in range(self.window_size):
                window_ids += unzipped_ids[n][i: i + self.window_size - n]
            wb_ids.append(window_ids)
        return wb_ids


def get_type(word):
    word_len = len(word)
    if word_len == 1:
        o_ch = ord(word)
        if is_cjk_char(o_ch):
            if is_chinese_char(o_ch):
                return "CH"
            else:
                return "JK"
        elif is_punctuation(o_ch):
            return "PU"

    if rENUM.match(word):
        if rNUM.match(word):
            return "NU"
        elif rENG.match(word):
            return "EN"
        else:
            return "OL"
    else:
        return "UN"