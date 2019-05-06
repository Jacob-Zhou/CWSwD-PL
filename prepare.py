import json

import tensorflow as tf

import augmenter
import dictionary_builder
import models
import process
import tokenization
from models import ModelConfig
import numpy as np
import os

from utils import DimInfo

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tf_record files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", "./",
    "The scores directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "model", None,
    "baseline or dict_concat")

## Tokenizer parameters
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("bigram_file", None,
                    "The bigram file that the BERT model was trained on.")

flags.DEFINE_string("type_file", None,
                    "The type file that the BERT model was trained on.")

flags.DEFINE_string("bigram_type_file", None,
                    "The bigram type file that the BERT model was trained on.")

flags.DEFINE_string("ngram_file", None,
                    "The ngram file that the BERT model was trained on.")

flags.DEFINE_integer("window_size", 1,
                     "The max word length.")

flags.DEFINE_bool("traditional_template", False,
                  "Use Part Label Data to Train.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

## Dictionary Builder parameters
flags.DEFINE_string("dict_file", None,
                    "The dict file that the BERT model was trained on.")

flags.DEFINE_integer("min_word_len", 2,
                     "The min word length.")

flags.DEFINE_integer("max_word_len", 5,
                     "The max word length.")

flags.DEFINE_float(
    "dict_augment_rate", 0,
    "the probability of filp dict input")

## Ground Truth Label Processor parameters
flags.DEFINE_bool("multitag", False,
                  "how the label represent")

flags.DEFINE_string("processor", "CWSProcessor", "BiLabelProcessor or CWSProcessor")

## Noise Matrix for Nova Model
flags.DEFINE_string("noise_matrix", None, "Noise Matrix for Nova Model")
flags.DEFINE_float("inside_factor", 1, "Noise Matrix for Nova Model")


def prepare_form_config(flags):
    tf.gfile.MakeDirs(flags.output_dir)
    processor = getattr(process, flags.processor)()
    if flags.bigram_file is None:
        tokenizer = tokenization.WindowTokenizer(
            vocab_file=flags.vocab_file,
            do_lower_case=flags.do_lower_case, window_size=flags.window_size)
    else:
        tokenizer = tokenization.WindowBigramTokenizer(
            vocab_file=flags.vocab_file, bigram_file=flags.bigram_file,
            do_lower_case=flags.do_lower_case, window_size=flags.window_size)
    if flags.multitag:
        processor = process.MultiTagProcessor()
    dict_builder = None
    if flags.dict_file is not None:
        dict_builder = dictionary_builder.DefaultDictionaryBuilder(flags.dict_file,
                                                                   min_word_len=flags.min_word_len,
                                                                   max_word_len=flags.max_word_len)
    data_augmenter = augmenter.DefaultAugmenter(flags.dict_augment_rate)
    cls = None
    if flags.model == "baseline":
        cls = models.BaselineModel
    elif flags.model == "nova":
        cls = models.PartLabelModel
    elif flags.model == "aaai":
        cls = models.AAAIModel
    elif flags.model == "pytorch":
        cls = models.PytorchModel
    elif flags.model == "dict_concat":
        cls = models.DictConcatModel
    elif flags.model == "dict_hyper":
        cls = models.DictHyperModel
    elif flags.model == "attend_dict":
        cls = models.AttendedDictModel
    elif flags.model == "attend_input":
        cls = models.AttendedInputModel
    elif flags.model == "dual_dict":
        cls = models.DictConcatModel
        assert flags.bigram_file is not None, "dual_dict must need bigram file"
        tokenizer = tokenization.WindowNgramTokenizer(
            vocab_file=flags.vocab_file, ngram_file=flags.bigram_file,
            do_lower_case=flags.do_lower_case, window_size=flags.window_size)
        if dict_builder is None:
            dict_builder = dictionary_builder.DefaultDictionaryBuilder(flags.bigram_file,
                                                                       min_word_len=flags.min_word_len,
                                                                       max_word_len=flags.max_word_len)
        data_augmenter = augmenter.DualAugmenter(flags.window_size)
    if flags.traditional_template:
        if flags.vocab_file is None or flags.bigram_file is None or flags.type_file is None \
                or flags.bigram_type_file is None:
            raise ValueError("pl must set ...")

        if flags.window_size == 1:
            tokenizer = tokenization.TrTeBiTokenizer(
                vocab_file=flags.vocab_file, bigram_file=flags.bigram_file,
                type_file=flags.type_file, bigram_type_file=flags.bigram_type_file,
                do_lower_case=flags.do_lower_case)
        else:
            tokenizer = tokenization.TrTeWinBiTokenizer(
                vocab_file=flags.vocab_file, bigram_file=flags.bigram_file,
                type_file=flags.type_file, bigram_type_file=flags.bigram_type_file,
                do_lower_case=flags.do_lower_case, window_size=flags.window_size)
    if cls is None:
        raise ValueError("please use the right model nickname")
    config = ModelConfig.from_json_file(flags.config_file)
    config.__dict__["multitag"] = flags.multitag
    config.__dict__["traditional_template"] = flags.traditional_template
    if flags.noise_matrix:
        config.__dict__["noise_matrix"] = np.load(flags.noise_matrix)
    else:
        config.__dict__["noise_matrix"] = np.eye(config.num_classes)
    config.__dict__["inside_factor"] = flags.inside_factor
    dim_info = DimInfo(tokenizer.dim, dict_builder.dim if dict_builder else 1, 4 if flags.multitag else 1)
    for key, value in tokenizer.size_info.items():
        config.__dict__[key] = value
    config.__dict__["input_dim_info"] = tokenizer.dim_info
    return cls, config, dim_info, dict_builder, processor, tokenizer, data_augmenter


def get_info(info_file):
    info = json.load(open(info_file, "r"))
    assert isinstance(info, dict), "wrong type of json file"
    return info
