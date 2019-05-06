# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
import re
import tensorflow as tf
import utils
from preprocess import get_type


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
    return vocab


def convert_by_vocab(vocab, items, unk_token="<UNK>", bi_items=False, bi_unk_token="<BUNK>"):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    if bi_unk_token not in vocab:
        bi_unk_token = unk_token
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(vocab[unk_token if not bi_items else bi_unk_token])
    return output


def convert_tokens_to_ids(vocab, tokens, unk_token="<UNK>"):
    return convert_by_vocab(vocab, tokens, unk_token=unk_token)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class Tokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, unk_token="<UNK>", do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.unk_token = unk_token
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.dim = 1
        self.dim_info = [1]
        self.size_info = {"vocab_size": len(self.vocab)}

    def tokenize(self, text):
        split_tokens = []

        for token in self.basic_tokenizer.tokenize(text):
            if token in self.vocab:
                split_tokens.append(token)
            else:
                split_tokens.append(self.unk_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens, self.unk_token)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class WindowTokenizer(Tokenizer):
    def __init__(self, vocab_file, window_size, do_lower_case=True):
        if six.PY3:
            super().__init__(vocab_file, "<UNK>", do_lower_case)
        else:
            super(WindowTokenizer, self).__init__(vocab_file, "<UNK>", do_lower_case)
        self.window_size = window_size
        self.dim = window_size
        self.dim_info = [window_size]

    def convert_tokens_to_ids(self, tokens):
        """
        :param tokens:
        :return: windowed ids
        """
        fw = self.window_size // 2
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        unwind_ids = [self.vocab["<SOS>"]] * fw + convert_by_vocab(self.vocab, tokens, self.unk_token) + \
                     [self.vocab["<EOS>"]] * bw
        windowed_ids = [unwind_ids[i: i + self.window_size] for i in range(len(tokens))]
        return windowed_ids

    def convert_ids_to_tokens(self, ids):
        c_inx = self.window_size // 2
        c_ids = [win[c_inx] for win in ids]
        return convert_by_vocab(self.inv_vocab, c_ids)


class WindowBigramTokenizer(WindowTokenizer):
    def __init__(self, vocab_file, bigram_file, window_size, do_lower_case=True):
        if six.PY3:
            super().__init__(",".join([vocab_file, bigram_file]), window_size=window_size, do_lower_case=do_lower_case)
        else:
            super(WindowBigramTokenizer, self).__init__(",".join([vocab_file, bigram_file]),
                                                        window_size=window_size,
                                                        do_lower_case=do_lower_case)
        self.dim = window_size * 2 - 1
        self.dim_info = [window_size, window_size - 1]

    def convert_tokens_to_ids(self, tokens):
        fw = self.window_size // 2
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        padded_tokens = ["<SOS>"] * fw + tokens + ["<EOS>"] * bw
        uni_ids = convert_by_vocab(self.vocab, padded_tokens, self.unk_token)
        bi_ids = convert_by_vocab(self.vocab,
                                  ["".join(padded_tokens[i: i + 2]) for i in range(len(padded_tokens) - 1)],
                                  bi_items=True)
        wb_ids = [uni_ids[i: i + self.window_size] + bi_ids[i: i + self.window_size - 1] for i in range(len(tokens))]
        return wb_ids

    def convert_ids_to_tokens(self, ids):
        c_inx = self.window_size // 2
        c_ids = [win[c_inx] for win in ids]
        return convert_by_vocab(self.inv_vocab, c_ids)


class TrTeBiTokenizer(Tokenizer):
    def __init__(self, vocab_file, bigram_file, type_file, bigram_type_file, do_lower_case=True):
        if six.PY3:
            super().__init__(",".join([vocab_file]),
                             do_lower_case=do_lower_case)
        else:
            super(TrTeBiTokenizer, self).__init__(",".join([vocab_file]),
                                                  do_lower_case=do_lower_case)
        self.bigram_vocab = load_vocab(bigram_file)
        self.inv_bigram_vocab = {v: k for k, v in self.bigram_vocab.items()}

        self.type_vocab = load_vocab(type_file)
        self.inv_type_vocab = {v: k for k, v in self.type_vocab.items()}

        self.bigram_type_vocab = load_vocab(bigram_type_file)
        self.inv_bigram_type_vocab = {v: k for k, v in self.bigram_type_vocab.items()}

        self.dim = 5
        self.dim_info = [1, 1, 1, 1, 1]
        self.size_info = {"vocab_size": len(self.vocab),
                          "bigram_size": len(self.bigram_vocab),
                          "type_size": len(self.type_vocab),
                          "bigram_type_size": len(self.bigram_type_vocab)}
        self.unk_token = "<UNK>"

    def convert_tokens_to_ids(self, inputs):
        fw, bw = 1, 1
        # tokens, types = inputs
        tokens = inputs
        types = list(map(get_type, inputs))
        padded_tokens = ["<SOS>"] * fw + tokens + ["<EOS>"] * bw
        uni_ids = convert_by_vocab(self.vocab, padded_tokens, self.unk_token)
        bi_ids = convert_by_vocab(self.bigram_vocab,
                                  ["".join(padded_tokens[i: i + 2]) for i in range(len(padded_tokens) - 1)],
                                  "<BUNK>")

        padded_types = ["<SOS>"] * fw + types + ["<EOS>"] * bw
        uni_ty_ids = convert_by_vocab(self.type_vocab, padded_types, self.unk_token)
        bi_ty_ids = convert_by_vocab(self.bigram_type_vocab,
                                     ["".join(padded_types[i: i + 2]) for i in range(len(padded_types) - 1)],
                                     self.unk_token)

        ident = [1 if padded_tokens[i] == padded_tokens[i + 1] else 0 for i in range(len(padded_tokens) - 1)]
        wb_ids = [[uni_ids[i + 1]] + [bi_ids[i + 1]] +
                  [uni_ty_ids[i + 1]] + [bi_ty_ids[i + 1]] +
                  [ident[i + 1]] for i in range(len(tokens))
                  ]
        return wb_ids

    def convert_ids_to_tokens(self, ids):
        c_inx = 0
        c_ids = [win[c_inx] for win in ids]
        return convert_by_vocab(self.inv_vocab, c_ids)


class TrTeWinBiTokenizer(WindowTokenizer):
    def __init__(self, vocab_file, bigram_file, type_file, bigram_type_file, window_size, do_lower_case=True):
        if six.PY3:
            super().__init__(",".join([vocab_file]), window_size=window_size,
                             do_lower_case=do_lower_case)
        else:
            super(TrTeWinBiTokenizer, self).__init__(",".join([vocab_file]),
                                                     window_size=window_size,
                                                     do_lower_case=do_lower_case)
        self.bigram_vocab = load_vocab(bigram_file)
        self.inv_bigram_vocab = {v: k for k, v in self.bigram_vocab.items()}

        self.type_vocab = load_vocab(type_file)
        self.inv_type_vocab = {v: k for k, v in self.type_vocab.items()}

        self.bigram_type_vocab = load_vocab(bigram_type_file)
        self.inv_bigram_type_vocab = {v: k for k, v in self.bigram_type_vocab.items()}

        self.dim = window_size * 5 - 3
        self.dim_info = [window_size, window_size - 1, window_size, window_size - 1, window_size - 1]
        self.size_info = {"vocab_size": len(self.vocab),
                          "bigram_size": len(self.bigram_vocab),
                          "type_size": len(self.type_vocab),
                          "bigram_type_size": len(self.bigram_type_vocab)}
        self.unk_token = "<UNK>"

    def convert_tokens_to_ids(self, inputs):
        fw = self.window_size // 2
        # tokens, types = inputs
        tokens = inputs
        types = list(map(get_type, inputs))
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        padded_tokens = ["<SOS>"] * fw + tokens + ["<EOS>"] * bw
        uni_ids = convert_by_vocab(self.vocab, padded_tokens, self.unk_token)
        bi_ids = convert_by_vocab(self.bigram_vocab,
                                  ["".join(padded_tokens[i: i + 2]) for i in range(len(padded_tokens) - 1)],
                                  "<BUNK>")

        padded_types = ["<SOS>"] * fw + types + ["<EOS>"] * bw
        uni_ty_ids = convert_by_vocab(self.type_vocab, padded_types, self.unk_token)
        bi_ty_ids = convert_by_vocab(self.bigram_type_vocab,
                                     ["".join(padded_types[i: i + 2]) for i in range(len(padded_types) - 1)],
                                     self.unk_token)

        ident = [1 if padded_tokens[i] == padded_tokens[i + 1] else 0 for i in range(len(padded_tokens) - 1)]
        wb_ids = [uni_ids[i: i + self.window_size] + bi_ids[i: i + self.window_size - 1] +
                  uni_ty_ids[i: i + self.window_size] + bi_ty_ids[i: i + self.window_size - 1] +
                  ident[i: i + self.window_size - 1] for i in range(len(tokens))
                  ]
        return wb_ids

    def convert_ids_to_tokens(self, ids):
        c_inx = self.window_size // 2
        c_ids = [win[c_inx] for win in ids]
        return convert_by_vocab(self.inv_vocab, c_ids)


class WindowNgramTokenizer(WindowTokenizer):
    def __init__(self, vocab_file, ngram_file, window_size, do_lower_case=True):
        if six.PY3:
            super().__init__(vocab_file, window_size=window_size, do_lower_case=do_lower_case)
        else:
            super(WindowNgramTokenizer, self).__init__(vocab_file,
                                                       window_size=window_size,
                                                       do_lower_case=do_lower_case)
        assert window_size <= 15, "window size is too big, must small than 15"
        self.vocab = load_vocab(",".join([vocab_file, ngram_file]),
                                preserve_token=["P", "S", "E"] + ["U" + str(n) for n in range(15)])
        self.dim = (window_size * (window_size + 1)) // 2
        self.dim_info = [i for i in reversed(range(1, window_size + 1))]

    def convert_by_vocab(self, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            if item in self.vocab:
                output.append(self.vocab[item])
            else:
                output.append(self.vocab["U" + str(len(item) - 1)])
        return output

    def convert_tokens_to_ids(self, tokens):
        fw = self.window_size // 2
        bw = fw if fw * 2 + 1 == self.window_size else fw - 1
        padded_tokens = ["S"] * fw + tokens + ["E"] * bw
        unzipped_ids = [self.convert_by_vocab(
            ["".join(padded_tokens[i: i + 1 + n]) for i in range(len(padded_tokens) - n)]
        ) for n in range(self.window_size)]
        wb_ids = []
        for i in range(len(tokens)):
            window_ids = []
            for n in range(self.window_size):
                window_ids += unzipped_ids[n][i: i + self.window_size - n]
            wb_ids.append(window_ids)
        return wb_ids

    def convert_ids_to_tokens(self, ids):
        c_inx = self.window_size // 2
        c_ids = [win[c_inx] for win in ids]
        return convert_by_vocab(self.inv_vocab, c_ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = utils.convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            output.append(" ")
            output.append(char)
            output.append(" ")
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((0x4E00 <= cp <= 0x9FFF) or  #
                (0x3400 <= cp <= 0x4DBF) or  #
                (0x20000 <= cp <= 0x2A6DF) or  #
                (0x2A700 <= cp <= 0x2B73F) or  #
                (0x2B740 <= cp <= 0x2B81F) or  #
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or  #
                (0x2F800 <= cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
