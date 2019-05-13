import json

import tensorflow as tf

import models
from process import data_processor, dictionary_feature_extractor, augmenter, context_feature_extractor
import numpy as np
from collections import OrderedDict

from process.feature_builder import DimInfo
from utils import Config

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

flags.DEFINE_string("data_processor", "CWSDataProcessor", "BiLabelProcessor or CWSDataProcessor")

## Noise Matrix for Nova Model
flags.DEFINE_string("noise_matrix", None, "Noise Matrix for Nova Model")
flags.DEFINE_float("inside_factor", 1, "Noise Matrix for Nova Model")


def prepare_form_config(flags):
    tf.gfile.MakeDirs(flags.output_dir)
    processor = getattr(data_processor, flags.data_processor)()
    if flags.bigram_file is None:
        cxt_feature_extractor = context_feature_extractor.WindowContextFeatureExtractor(
            vocab_file=flags.vocab_file, window_size=flags.window_size)
    else:
        cxt_feature_extractor = context_feature_extractor.WindowBigramContextFeatureExtractor(
            vocab_file=flags.vocab_file, bigram_file=flags.bigram_file, window_size=flags.window_size)
    if flags.multitag:
        processor = data_processor.MultiTagDataProcessor()
    dict_feature_extractor = None
    if flags.dict_file is not None:
        dict_feature_extractor = dictionary_feature_extractor.DefaultDictionaryFeatureExtractor(flags.dict_file,
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
        cxt_feature_extractor = context_feature_extractor.WindowNgramContextFeatureExtractor(
            vocab_file=flags.vocab_file, ngram_file=flags.bigram_file, window_size=flags.window_size)
        if dict_feature_extractor is None:
            dict_feature_extractor = dictionary_feature_extractor.DefaultDictionaryFeatureExtractor(flags.bigram_file,
                                                                                                    min_word_len=flags.min_word_len,
                                                                                                    max_word_len=flags.max_word_len)
        data_augmenter = augmenter.DualAugmenter(flags.window_size)
    if flags.traditional_template:
        if flags.vocab_file is None or flags.bigram_file is None or flags.type_file is None \
                or flags.bigram_type_file is None:
            raise ValueError("pl must set ...")

        if flags.window_size == 1:
            cxt_feature_extractor = context_feature_extractor.TrTeBiContextFeatureExtractor(
                vocab_file=flags.vocab_file, bigram_file=flags.bigram_file,
                type_file=flags.type_file, bigram_type_file=flags.bigram_type_file)
        else:
            cxt_feature_extractor = context_feature_extractor.TrTeWinBiContextFeatureExtractor(
                vocab_file=flags.vocab_file, bigram_file=flags.bigram_file,
                type_file=flags.type_file, bigram_type_file=flags.bigram_type_file, window_size=flags.window_size)
    if cls is None:
        raise ValueError("please use the right model nickname")
    config = Config.from_json_file(flags.config_file)
    config.__dict__["multitag"] = flags.multitag
    config.__dict__["traditional_template"] = flags.traditional_template
    if flags.noise_matrix:
        config.__dict__["noise_matrix"] = np.load(flags.noise_matrix)
    else:
        config.__dict__["noise_matrix"] = np.eye(config.num_classes)
    config.__dict__["inside_factor"] = flags.inside_factor

    extractors = OrderedDict({"input_ids": cxt_feature_extractor,
                              "input_dicts": dict_feature_extractor})
    dim_info = DimInfo({feature_name: feature.dim for feature_name, feature in extractors.items()},
                       4 if flags.multitag else 1)
    for key, value in cxt_feature_extractor.size_info.items():
        config.__dict__[key] = value
    config.__dict__["input_dim_info"] = cxt_feature_extractor.dim_detail
    return cls, config, dim_info, processor, extractors, data_augmenter


def get_info(info_file):
    info = json.load(open(info_file, "r"))
    assert isinstance(info, dict), "wrong type of json file"
    return info
