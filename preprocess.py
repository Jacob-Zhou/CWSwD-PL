# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import re
import os
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input', None,
                    "The input data. data need to be segment by '  '")
flags.DEFINE_string('filter_file', None,
                    " ")
flags.DEFINE_string('prepercess_output_dir', 'data',
                    "The output data dir")
flags.DEFINE_string('output', None,
                    "The input data file name, should without suffix")
flags.DEFINE_bool("log", False,
                  "show log on the console or not")


rENUM = re.compile(r'(([-–+])?\d+(([.·])\d+)?%?|([0-9_.·]*[A-Za-z]+[0-9_.·]*)+)')
rNUM = re.compile(r'([-–+])?\d+(([.·])\d+)?%?')
rENG = re.compile(r'([0-9_.·]*[A-Za-z]+[0-9_.·]*)+')
rWEEK = re.compile(r'\[\s*([^][]*)\s*\]')
rSUB = re.compile(r'↕{2,}')
rCLR = re.compile(r' *[][\']+ *')
rCLR_2 = re.compile(r'cite book.*$')
rCLR_3 = re.compile(r'↕noteTag \|')
rCLR_4 = re.compile(r'↕lang \| [^| ]* \|')
rCLR_5 = re.compile(
    r'↕(name|date|time|bot|[0-9]+|catIdx|loc|abbr|sp|k|page|pp|p|deadurl|(access|fix)[ -]*(date|attempted)|isbn|italic|a|hg) =[^↕]*↕')
rCLR_6 = re.compile(r'[0-9 -]*↕action = edit↕编辑$')
rCLR_7 = re.compile(r'(^[↕ ]+)|([↕ ]+$)')
rCHSP = re.compile(r'([\u4e00-\u9fff]) ([\u4e00-\u9fff])')


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def is_cjk_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((0x3040 <= cp <= 0x30FF) or  # 日文
            (0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0xAC00 <= cp <= 0xD7AF) or  # 韩文
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True
    return False


def is_chinese_char(cp):
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


def is_punctuation(cp):
    if (
            (0x0020 <= cp <= 0x002F) or
            (0x003A <= cp <= 0x0040) or
            (0x005B <= cp <= 0x0060) or
            (0x007B <= cp <= 0x007E) or
            (0x00A0 <= cp <= 0x00BF) or
            (0x2000 <= cp <= 0x206F) or  #
            (0x2E00 <= cp <= 0x2E7F) or  #
            (0x3000 <= cp <= 0x303F) or
            (0xFE30 <= cp <= 0xFE4F) or
            (0xFE50 <= cp <= 0xFE6F)
    ):  #
        return True
    return False


def is_english_word(word):
    for c in word:
        if "a" <= c <= "z":
            return True
    return False


def filter_wiki(input, filter_file, output_dir, output):
    output_filename = os.path.join(output_dir, output)
    with codecs.open(filter_file, 'r', 'utf-8') as ffl:
        filter_text = "".join(ffl.readlines())
    with codecs.open(input, 'r', 'utf-8') as fin:
        with codecs.open(output_filename, 'w', 'utf-8') as fout:
            for line in fin:
                sents = strQ2B(line.strip())
                sents = re.sub('。', '。\n', sents).split("\n")
                for sent in sents:
                    week_labels = rWEEK.findall(sent)
                    match_count = 0
                    for week_label in week_labels:
                        week_label = week_label.strip()
                        segment = "".join(week_label.split())
                        if len(segment) > 1 and is_chinese_char(ord(segment[0])) and segment in filter_text:
                            match_count += 1
                        if match_count > 2:
                            sent = rCLR.sub('↕', sent)  # 替换弱标记
                            if rCLR_2.match(sent) or rCLR_6.match(sent):
                                break
                            iters = 0
                            while True:
                                old, iters = sent, iters + 1
                                sent = rCLR_3.sub('↕', sent)
                                sent = rCLR_4.sub('', sent)
                                sent = rCLR_5.sub('↕', sent)
                                sent = rCHSP.sub('\\1↕\\2', sent)
                                if old == sent or iters > 4:
                                    break
                            sent = rSUB.sub('↕', sent)  # 替换弱标记
                            sent = rCLR_7.sub('', sent)  # 替换弱标记
                            if 20 < len(sent) < 200 and sent.count('↕') > 1:
                                fout.write(sent)
                                fout.write("\n")
                            break


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


def to_training(input, output_dir, output):
    output_filename = os.path.join(output_dir, output)
    with codecs.open(input, 'r', 'utf-8') as fin:
        with codecs.open(output_filename, 'w', 'utf-8') as fout:
            for line in fin:
                sent = str.lower(strQ2B(line))
                vocabs = sent.split('↕')
                for vocab in vocabs:
                    vocab = vocab.strip()
                    # 将标点和英文分割出来
                    segments = vocab.split()
                    if len(segments) == 1:
                        if len(segments[0]) == 1:
                            char = segments[0][0]
                            c_type = get_type(char)
                            fout.write(segments[0][0] + "  " + c_type + "  " + "A" + "\n")
                        elif len(segments[0]) > 1:
                            if is_cjk_char(ord(segments[0][0])):
                                segment_len = len(segments[0])
                                for c_i, segment_char in enumerate(segments[0]):
                                    c_type = get_type(segment_char)
                                    fout.write(
                                        segment_char + "  " + c_type + "  " + (
                                            "BS" if c_i == 0 else (
                                                "ES" if c_i == segment_len - 1 else "A"
                                            )
                                        ) + "\n"
                                    )
                            else:
                                c_type = get_type(segments[0])
                                fout.write(segments[0] + "  " + c_type + "  " + "A" + "\n")

                    elif len(segments) > 1:
                        segments_len = len(segments)
                        for s_i, segment in enumerate(segments):
                            if len(segment) <= 0:
                                continue
                            if is_cjk_char(ord(segment[0])):
                                segment_len = len(segment)
                                for c_i, segment_char in enumerate(segment):
                                    c_type = get_type(segment_char)
                                    fout.write(segment_char + "  " + c_type + "  " + (
                                        "BS" if s_i == 0 and c_i == 0 else (
                                            "ES" if s_i == segments_len - 1 and c_i == segment_len - 1 else "A"
                                        )
                                    ) + "\n")
                            else:
                                c_type = get_type(segment)
                                fout.write(segment + "  " + c_type + "  " + (
                                    "BS" if s_i == 0 else ("ES" if s_i == segments_len - 1 else "A")
                                ) + "\n")

                fout.write("\n\n")


def get_vocab(input, preprocess_output_dir, output):
    output_filename = os.path.join(preprocess_output_dir, output)
    sents = []
    vocabs_dict = {}
    with codecs.open(input, 'r', 'utf-8') as fin:
        with codecs.open(output_filename, 'w', 'utf-8') as fout:
            for line in fin:
                sent = strQ2B(line)
                vocabs = sent.split('↕')
                for vocab in vocabs:
                    vocab = vocab.strip()
                    segments = vocab.split(' ')
                    for segment in segments:
                        if len(segment) > 1 and is_cjk_char(ord(segment[0])):
                            for char in segment:
                                if char in vocabs_dict:
                                    vocabs_dict[char] += 1
                                else:
                                    vocabs_dict[char] = 1
                        else:
                            segment = str.lower(segment)
                            if segment in vocabs_dict:
                                vocabs_dict[segment] += 1
                            else:
                                vocabs_dict[segment] = 1

            sorted_vocabs = sorted(vocabs_dict.items(), key=lambda item: item[1], reverse=True)
            for v, c in sorted_vocabs:
                if (str.isdigit(v) and c <= 15) or c < 1:
                    continue
                fout.write(str(v) + " " + str(c) + "\n")

                # fout.write(sent)

            # for line in fin:
            #     sent = strQ2B(line).split()
            #     new_sent = []
            #     for word in sent:
            #         word = rNUM.sub('0', word)
            #         word = rENG.sub('X', word)
            #         new_sent.append(word)
            #     sents.append(new_sent)
            # for sent in sents:
            #     fout.write('  '.join(sent))
            #     fout.write('\n')


def main(_):
    if FLAGS.log:
        tf.logging.set_verbosity(tf.logging.INFO)
    output_dir = FLAGS.prepercess_output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    to_training(FLAGS.input, output_dir, FLAGS.output)
    # get_vocab(FLAGS.input, output_dir, FLAGS.output)
    # filter_wiki(FLAGS.input, FLAGS.filter_file, output_dir, FLAGS.output)


if __name__ == '__main__':
    flags.mark_flag_as_required("input")
    flags.mark_flag_as_required("output")
    tf.app.run()
