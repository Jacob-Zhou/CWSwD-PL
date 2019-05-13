import collections

import numpy as np
import tensorflow as tf

import utils


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_features, label_ids, seq_length):
        self.input_features = input_features
        self.seq_length = seq_length
        self.label_ids = label_ids


class FeatureBuilder(object):
    def __init__(self, extractors, label_map):
        self.extractors = extractors
        self.label_map = label_map

    def build_single_example(self, ex_index, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        tokens_raw = example.text
        labels_raw = example.labels

        tokens = []
        label_ids = []
        assert len(tokens_raw) == len(labels_raw)

        for token, label in zip(tokens_raw, labels_raw):
            tokens.append(token)
            label_ids.append(self.label_map[label])

        input_features = {}
        seq_length = len(tokens)
        assert seq_length == len(label_ids)
        for feature_name, feature_extractor in self.extractors.items():
            feature = feature_extractor.extract(tokens)
            input_features[feature_name] = feature
            assert seq_length == len(feature)

        if ex_index < 1:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid:        %s" % example.guid)
            tf.logging.info("tokens:      %s" % " ".join(
                [utils.printable_text(x) for x in tokens]))
            for feature_name, feature in input_features.items():
                tf.logging.info("%s:   %s" % (feature_name, " ".join([str(x) for x in feature])))
            tf.logging.info("labels:      %s" % " ".join([str(x) for x in example.labels]))
            tf.logging.info("labels_ids:  %s" % " ".join([str(x) for x in label_ids]))

        feature = InputFeatures(
            input_features=input_features,
            label_ids=label_ids,
            seq_length=seq_length)
        return feature

    def build_features_from_examples(self, examples, output_file=None):

        if output_file:
            writer = tf.python_io.TFRecordWriter(output_file)

        features = []
        examples = list(examples)
        count = 0
        for (ex_index, example) in enumerate(examples):
            count += 1
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d" % ex_index)

            feature = self.build_single_example(ex_index, example)

            if output_file:
                feature_dict = collections.OrderedDict()
                for feature_name, input_feature in feature.input_features.items():
                    plain_feature = np.array(input_feature).reshape([-1])
                    feature_dict[feature_name] = self._create_float_feature(plain_feature)
                feature_dict["seq_length"] = self._create_int_feature([feature.seq_length])
                plain_label_ids = np.array(feature.label_ids).reshape([-1])
                feature_dict["label_ids"] = self._create_int_feature(plain_label_ids)

                tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(tf_example.SerializeToString())
            else:
                features.append((feature.input_features, feature.label_ids, feature.seq_length))

        tf.logging.info("Writing example %d" % count)
        if output_file:
            return None
        else:
            return features

    @staticmethod
    def _create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    @staticmethod
    def _create_float_feature(values):
        f = tf.train.Feature(int64_list=tf.train.FloatList(value=list(values)))
        return f


class DimInfo:
    def __init__(self, feature_dims, label_dim):
        self.label_dim = label_dim
        self.feature_dims = feature_dims