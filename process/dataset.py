import random

import tensorflow as tf
import numpy as np


class Dataset(object):
    def __iter__(self):
        raise NotImplementedError()


class TFRecordDataset(Dataset):
    def __init__(self, sess, features_files, batch_size, dim_info):
        self.dim_info = dim_info
        self.batch_size = batch_size
        self.features_files = features_files
        self.sess = sess

        def _decode_record(record):
            """Decodes a record to a TensorFlow example."""
            name_to_features = {feat_name: tf.VarLenFeature(tf.float64) for feat_name in dim_info}
            name_to_features.update({"label_ids": tf.VarLenFeature(tf.int64),
                                     "seq_length": tf.FixedLenFeature([], tf.int64)})
            raw_example = tf.parse_single_example(record, name_to_features)
            example = {}

            for feat_name, feat_dim in dim_info.feature_dims.items():
                example[feat_name] = tf.sparse.to_dense(raw_example[feat_name])
                example[feat_name] = tf.reshape(example[feat_name], shape=[-1, feat_dim])

            example["label_ids"] = tf.sparse.to_dense(raw_example["label_ids"])
            if dim_info.label_dim != 1:
                example["label_ids"] = tf.reshape(example["label_ids"], shape=[-1, dim_info.label_dim])
            else:
                example["label_ids"] = tf.reshape(example["label_ids"], shape=[-1])
            example["seq_length"] = raw_example["seq_length"]

            return example

        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(features_files)

        d = d.map(map_func=lambda record: _decode_record(record), num_parallel_calls=6)
        padded_shapes = {name: [None, dim] for name, dim in dim_info.feature_dims.items()}
        padded_shapes.update({
            "label_ids": [None, dim_info.label_dim] if dim_info.label_dim != 1 else [None],
            "seq_length": []
        })

        d = d.padded_batch(batch_size=batch_size,
                           padded_shapes=padded_shapes,
                           drop_remainder=False)
        d = d.prefetch(buffer_size=batch_size + 1)
        self.iterator = d.make_initializable_iterator()

    def __iter__(self):
        self.sess.run(self.iterator.initializer)
        next_element = self.iterator.get_next()
        while True:
            try:
                example = self.sess.run(next_element)
                yield example
            except tf.errors.OutOfRangeError:
                pass


class PaddingDataset(Dataset):
    def __init__(self, features_list, batch_size, dim_info, shuffle=True, pad_id=0):
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dim_info = dim_info
        if isinstance(features_list, list) and isinstance(features_list[0], list):
            self.features_list = features_list
        else:
            self.features_list = [features_list]

    def _padding_features(self, batch_features):
        return self.padding_features(batch_features, self.dim_info, self.pad_id)

    @staticmethod
    def padding_features(batch_features, dim_info, pad_id=0):
        # input_features: Dict()
        input_features, label_ids, seq_length = zip(*batch_features)
        max_length = max(seq_length)

        batch_size = len(batch_features)
        padded_input = {}
        for name, dim in dim_info.feature_dims.items():
            padded_input[name] = np.ones(
                (batch_size, max_length, dim), dtype=np.int32) * pad_id

        if dim_info.label_dim == 1:
            padded_label_ids = np.ones((batch_size, max_length), dtype=np.int32) * pad_id
        else:
            padded_label_ids = np.ones((batch_size, max_length, dim_info.label_dim), dtype=np.int32) * pad_id

        for i in range(batch_size):
            slen = seq_length[i]
            for name, feature in input_features[i].items():
                padded_input[name][i, :slen] = feature
            padded_label_ids[i, :slen] = label_ids[i]

        return padded_input, padded_label_ids, seq_length

    def __iter__(self):
        train_features = []
        for feature in self.features_list:
            train_features += feature
        if self.shuffle:
            random.shuffle(train_features)

        data_len = len(train_features)
        batch_size = self.batch_size
        batch_len = (data_len // batch_size) + 1

        for i in range(batch_len):
            if i * batch_size < data_len:
                batch_features = train_features[i * batch_size:(i + 1) * batch_size]
                yield self._padding_features(batch_features)


class CorpusWeightingDataset(PaddingDataset):
    def __init__(self, features_list, weight, batch_size, dim_info, pad_id=0):
        super().__init__(features_list, batch_size, dim_info, True, pad_id)
        self.weight = weight

    def __iter__(self):
        train_features = []
        for feature, w in zip(self.features_list, self.weight):
            random.shuffle(feature)
            train_features += feature[:w]
        random.shuffle(train_features)

        data_len = len(train_features)
        batch_size = self.batch_size
        batch_len = (data_len // batch_size) + 1

        for i in range(batch_len):
            if i * batch_size < data_len:
                batch_features = train_features[i * batch_size:(i + 1) * batch_size]
                yield self._padding_features(batch_features)


class BatchMixDataset(PaddingDataset):
    def __init__(self, features_list, weight, batch_size, dim_info, pad_id=0):
        super().__init__(features_list, batch_size, dim_info, True, pad_id)
        self.weight = weight

    def __iter__(self):
        batch_len = float("Inf")
        for feature, w in zip(self.features_list, self.weight):
            random.shuffle(feature)
            max_batch = int(len(feature) // w)
            batch_len = max_batch if max_batch < batch_len else batch_len

        for i in range(batch_len):
            for feature, w in zip(self.features_list, self.weight):
                batch_features = feature[i * self.batch_size * w:(i + 1) * self.batch_size * w]
                yield self._padding_features(batch_features)
