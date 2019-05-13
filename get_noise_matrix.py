from datetime import datetime

import os

import models.ModelAdapter
from checkmate import get_all_checkpoint
from models import SegmentModel
import tensorflow as tf
from process import data_processor
import numpy as np

from prepare import prepare_form_config

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 512, "Total batch size for predict.")
flags.DEFINE_string("prefix", "", "")
flags.DEFINE_bool("plain_pl", False, "")
flags.DEFINE_string("pretrain_dir", None, "")
flags.DEFINE_string("pl_domain", None,
                    "the domain use for part label training")


def file_based_input_builder(input_file, batch_size, dim_info):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.VarLenFeature(tf.int64),
        "input_dicts": tf.VarLenFeature(tf.int64),
        "label_ids": tf.VarLenFeature(tf.int64),
        "seq_length": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        input_ids = tf.sparse.to_dense(example["input_ids"])
        input_ids = tf.reshape(input_ids, shape=[-1, dim_info.input_dim])

        input_dicts = tf.sparse.to_dense(example["input_dicts"])
        input_dicts = tf.reshape(input_dicts, shape=[-1, dim_info.dict_dim])
        example["input_ids"], example["input_dicts"] = input_ids, input_dicts
        example["label_ids"] = tf.sparse.to_dense(example["label_ids"])
        if dim_info.label_dim != 1:
            example["label_ids"] = tf.reshape(example["label_ids"], shape=[-1, dim_info.label_dim])
        else:
            example["label_ids"] = tf.reshape(example["label_ids"], shape=[-1])
        example["seq_length"] = example["seq_length"]

        return example

    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)

    d = d.map(map_func=lambda record: _decode_record(record, name_to_features), num_parallel_calls=6)
    if dim_info.label_dim != 1:
        d = d.padded_batch(batch_size=batch_size,
                           padded_shapes={"input_ids": [None, dim_info.input_dim],
                                          "input_dicts": [None, dim_info.dict_dim],
                                          "label_ids": [None, dim_info.label_dim],
                                          "seq_length": []},
                           drop_remainder=False)
    else:
        d = d.padded_batch(batch_size=batch_size,
                           padded_shapes={"input_ids": [None, dim_info.input_dim],
                                          "input_dicts": [None, dim_info.dict_dim],
                                          "label_ids": [None],
                                          "seq_length": []},
                           drop_remainder=False)
    d = d.prefetch(buffer_size=batch_size + 1)
    return d.make_one_shot_iterator()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_class, config, dim_info, dict_builder, processor, tokenizer, data_augmenter = prepare_form_config(FLAGS)

    part_label_file = os.path.join(FLAGS.data_dir, f"{FLAGS.pl_domain}_part_label.tf_record")

    if FLAGS.plain_pl:
        processor = data_processor.PlainLineDataProcessor()
        part_label_examples = processor.get_examples(data_dir=FLAGS.data_dir, file_name=f"{FLAGS.pl_domain}_domain")
    else:
        part_label_examples = processor.get_pl_examples(FLAGS.data_dir, FLAGS.pl_domain)
    data_processor.file_based_convert_examples_to_features(
        examples=part_label_examples, tokenizer=tokenizer, dict_builder=dict_builder,
        label_map=processor.get_labels(), output_file=part_label_file)

    train_features = None
    # train_examples = processor.get_examples(data_dir=FLAGS.data_dir, example_type="train")
    # train_features = process.convert_examples_to_features(
    #     examples=train_examples, tokenizer=tokenizer, dict_builder=dict_builder,
    #     label_map=processor.get_labels())
    # del train_examples

    test_features = None
    # test_examples = processor.get_examples(data_dir=FLAGS.data_dir, example_type="test", domain=FLAGS.pl_domain)
    # test_features = process.convert_examples_to_features(
    #     examples=test_examples, tokenizer=tokenizer, dict_builder=dict_builder,
    #     label_map=processor.get_labels())

    model_path = os.path.join(FLAGS.pretrain_dir, "model.ckpt")

    with tf.Graph().as_default(), tf.Session() as sess:
        part_label_dataset = file_based_input_builder(part_label_file, FLAGS.batch_size, dim_info)

        # train & eval
        model = models.ModelAdapter.ModelAdapter(model_class,
                                                 dim_info=dim_info,
                                                 config=config, init_checkpoint=None, tokenizer=tokenizer,
                                                 init_embedding=None, learning_rate=0.01)

        saver = tf.train.Saver()
        epoch, checkpoint = get_all_checkpoint(FLAGS.pretrain_dir)[-1]

        saver.restore(sess, checkpoint)
        get_noise(sess, model, dim_info, train_features, test_features, part_label_dataset, FLAGS.output_dir)


def get_noise(sess, model, dim_info, train_features, test_features, pl_dataset, output_dir, show_info=True):
    next_element = pl_dataset.get_next()
    start_time = datetime.now()
    step = 0
    batch_index = 0
    count_ik = np.zeros([4, 4])
    count_i = np.zeros([4])

    # for step, (input_ids, input_dicts, label_ids, seq_length) in enumerate(
    #         utils.data_iterator(test_features, dim_detail=dim_detail, batch_size=64, shuffle=False)):
    #     loss, length, prediction = sess.run(
    #         [model.total_loss, model.seq_length, model.prediction],
    #         feed_dict={model.input_ids: input_ids,
    #                    model.input_dicts: input_dicts,
    #                    model.label_ids: np.zeros_like(label_ids),
    #                    model.seq_length: seq_length,
    #                    model.dropout_keep_prob: 1}
    #     )
    #
    #     step += 1
    #     # label_ids: [B, MaxLen, D]
    #     # prediction_one_hot: [B, MaxLen, D]
    #     # prediction: [B, MaxLen]
    #     prediction_one_hot = tf.keras.utils.to_categorical(prediction)
    #     true_batch_size = label_ids.shape[0]
    #     for i in range(true_batch_size):
    #         prediction_one_hot[i, length[i]:] = 0
    #     count_i += np.add.reduce(label_ids, axis=(0, 1))
    #     count_ik += np.add.reduce(
    #         np.multiply(np.transpose(np.expand_dims(prediction_one_hot, axis=-1), (0, 1, 3, 2)),
    #                     np.expand_dims(label_ids, axis=-1)), axis=(0, 1))
    #
    #     if step % 100 == 0 and show_info:
    #         now_time = datetime.now()
    #         tf.logging.info(
    #             f"Step: {step} ({(now_time - start_time).total_seconds():.2f} sec)")
    #         if step % 10000 == 0:
    #             tf.logging.info(
    #                 f"noise_count_ij:\n{count_ik}\n\ncount_i:\n{count_i}\n")
    #         start_time = now_time
    #     batch_index += 1
    #
    # real_count_ik = count_ik + count_ik.transpose()
    # real_count_ik[range(4), range(4)] //= 2
    # p_ik = real_count_ik / np.expand_dims(count_i, axis=-1)
    # tf.logging.info(
    #     f"p:\n{p_ik}\n")

    noise_count_kj = np.zeros([4, 4])
    noise_count_k = np.zeros([4])
    right_count = 0
    example_count = 0
    while True:
        try:
            example = sess.run(next_element)
            input_ids = example["input_ids"]
            input_dicts = example["input_dicts"]
            label_ids = example["label_ids"]
            seq_length = example["seq_length"]

            loss, length, prediction = sess.run(
                [model.total_loss, model.seq_length, model.prediction],
                feed_dict={model.input_ids: input_ids,
                           model.input_dicts: input_dicts,
                           model.label_ids: label_ids,
                           model.seq_length: seq_length,
                           model.dropout_keep_prob: 1}
            )

            step += 1
            # label_ids: [B, MaxLen, D]
            # prediction_one_hot: [B, MaxLen, D]
            # prediction: [B, MaxLen]
            prediction_one_hot = tf.keras.utils.to_categorical(prediction)
            true_batch_size = label_ids.shape[0]
            example_count += true_batch_size
            for i in range(true_batch_size):
                prediction_one_hot[i, length[i]:] = 0
            # count_i += np.add.reduce(prediction_one_hot, axis=(0, 1))
            right_pred = np.logical_or.reduce(np.logical_and(prediction_one_hot, label_ids), axis=-1)
            for i in range(true_batch_size):
                if np.sum(right_pred[i, :length[i]]) == length[i]:
                    right_count += 1
            label_num = np.add.reduce(label_ids, axis=-1)
            week_label = np.expand_dims(np.logical_and(label_num > 1, label_num < 4), axis=-1)
            noise_count_k += np.add.reduce(week_label * prediction_one_hot, axis=(0, 1))
            label_count = np.expand_dims(right_pred, axis=-1) * prediction_one_hot + \
                          np.expand_dims(1 - right_pred, axis=-1) * label_ids
            noise_count_kj += np.add.reduce(
                np.multiply(np.transpose(np.expand_dims(week_label * prediction_one_hot, axis=-1), (0, 1, 3, 2)),
                            np.expand_dims(label_count, axis=-1)), axis=(0, 1))

            if step % 1000 == 0 and show_info:
                now_time = datetime.now()
                tf.logging.info(
                    f"Step: {step} ({(now_time - start_time).total_seconds():.2f} sec)")
                if step % 10000 == 0:
                    tf.logging.info(
                        f"noise_count_ij:\n{noise_count_kj}\n\ncount_i:\n{noise_count_k}\n")
                start_time = now_time
            batch_index += 1
        except tf.errors.OutOfRangeError:
            break

    np.set_printoptions(precision=16)
    real_count_kj = noise_count_kj + noise_count_kj.transpose()
    real_count_kj[range(4), range(4)] //= 2
    p_kj = real_count_kj / np.expand_dims(noise_count_k, axis=-1)
    p_kj /= np.expand_dims(np.add.reduce(p_kj, axis=-1), axis=-1)
    tf.logging.info(
        f"p:\n{p_kj}\n{right_count} / {example_count}\n")
    # p_ij = np.einsum("kj,ik->ij", p_kj, p_ik)
    # tf.logging.info(
    #     f"p:\n{p_ij}\n")
    # np.save(os.path.join(output_dir, f"{FLAGS.pl_domain}_mistake.npy"), p_ik)
    # np.save(os.path.join(output_dir, f"{FLAGS.pl_domain}_inacc_noise.npy"), p_kj)
    # np.save(os.path.join(output_dir, f"{FLAGS.pl_domain}_noise.npy"), p_ij)
    np.save(os.path.join(output_dir, f"{FLAGS.pl_domain}_{FLAGS.prefix}_noise.npy"), p_kj)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model")
    tf.app.run()
