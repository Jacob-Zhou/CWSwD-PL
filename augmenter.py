import tensorflow as tf

class BasicAugmenter:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def augment(self, input_ids, input_dicts):
        raise NotImplementedError()


class DefaultAugmenter(BasicAugmenter):
    def __init__(self, dict_augment_rate=0.01, **kwargs):
        self.dict_augment_rate = dict_augment_rate
        super().__init__(**kwargs)

    def augment(self, input_ids, input_dicts):
        if self.dict_augment_rate > 0:
            flip_mask = tf.random.uniform(tf.shape(input_dicts)) < self.dict_augment_rate
            # flip if flip mask is true
            input_dicts = tf.cast(input_dicts, dtype=tf.bool)
            input_dicts = tf.logical_xor(input_dicts, flip_mask)
            input_dicts = tf.cast(input_dicts, dtype=tf.int64)
        return input_ids, input_dicts


class DualAugmenter(BasicAugmenter):
    def __init__(self, windows, input_augment_rate=0.25, dict_augment_rate=0.25, **kwargs):
        # dict_augment_rate 0.80 0.50
        self.windows = windows
        self.input_augment_rate = input_augment_rate
        self.dict_augment_rate = dict_augment_rate
        super().__init__(**kwargs)

    def augment(self, input_ids, input_dicts):
        # [len, dim]
        shape = tf.shape(input_dicts)
        slen = shape[0]
        if self.input_augment_rate > 0:
            # TODO need to fix
            drop_mask_0 = tf.random.uniform((slen, 5)) < 0.05
            drop_mask_1 = tf.random.uniform((slen, 7)) < 0.25 # 0.50 # 0.25
            drop_mask_2 = tf.random.uniform((slen, 3)) < 0.10 # 0.50 # 0.10
            unk_token = tf.ones_like(input_ids) * tf.constant([3, 3, 3, 3, 3,
                                                               4, 4, 4, 4,
                                                               5, 5, 5,
                                                               6, 6,
                                                               7], dtype=tf.int64)
            drop_mask = tf.cast(tf.concat([drop_mask_0, drop_mask_1, drop_mask_2], axis=1), dtype=tf.int64)
            input_ids = input_ids * (1 - drop_mask) + unk_token * drop_mask
        if self.dict_augment_rate > 0:
            keep_mask = tf.random.uniform(tf.shape(input_dicts)) >= self.dict_augment_rate
            keep_mask = tf.cast(keep_mask, dtype=tf.int64)
            input_dicts = input_dicts * keep_mask
        return input_ids, input_dicts
