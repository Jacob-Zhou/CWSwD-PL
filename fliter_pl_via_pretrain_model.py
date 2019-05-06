import codecs
from datetime import datetime

import utils
import os

from checkmate import get_all_checkpoint
from models import SegmentModel
import tensorflow as tf
import process

from prepare import prepare_form_config

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 512, "Total batch size for predict.")
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

    part_label_file = os.path.join(FLAGS.data_dir, "part_label.tf_record")

    # if FLAGS.plain_pl:
    #     processor = process.PlainLineProcessor()
    #     part_label_examples = processor.get_examples(FLAGS.data_dir, f"{FLAGS.pl_domain}_domain")
    # else:
    #     part_label_examples = list(processor.get_pl_examples(FLAGS.data_dir, FLAGS.pl_domain))
    #
    # process.file_based_convert_examples_to_features(
    #     examples=part_label_examples, tokenizer=tokenizer, dict_builder=dict_builder,
    #     label_map=processor.get_labels(), output_file=part_label_file)

    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(31415926)

        input_dataset = file_based_input_builder(part_label_file, FLAGS.batch_size, dim_info)

        # train & eval
        model = SegmentModel.LowLevelModel(model_class,
                                           dim_info=dim_info,
                                           config=config, init_checkpoint=None, tokenizer=tokenizer,
                                           init_embedding=None, learning_rate=0.01)

        saver = tf.train.Saver()
        epoch, checkpoint = get_all_checkpoint(FLAGS.pretrain_dir)[-1]

        saver.restore(sess, checkpoint)
        # saver.restore(sess, model_path)

        full_label_running(sess, model, input_dataset, FLAGS.output_dir, tokenizer=tokenizer)


def keep(gt, pred):
    for g, p in zip(gt, pred):
        if g[p] == 0:
            return True
        # if (g == [1, 0, 0, 0] or g == [1, 0, 0, 1]) and (p == 1 or p == 2):
        #     return True
        # elif (g == [0, 0, 1, 0] or g == [0, 0, 1, 1]) and (p == 0 or p == 1):
        #     return True
        # elif g == [0, 0, 0, 1] and p != 3:
        #     return True
    return False


def label2str(label):
    if label == [1, 0, 0, 0] or label == 0:
        return "B"
    elif label == [0, 1, 0, 0] or label == 1:
        return "M"
    elif label == [0, 0, 1, 0] or label == 2:
        return "E"
    elif label == [0, 0, 0, 1] or label == 3:
        return "S"
    elif label == [1, 0, 0, 1]:
        return "BS"
    elif label == [0, 0, 1, 1]:
        return "ES"
    elif label == [1, 1, 1, 1]:
        return "A"
    else:
        raise ValueError(str(label))
        # return "A"


def full_label_running(sess, model, dataset, output_dir, show_info=True, tokenizer=None):
    next_element = dataset.get_next()
    start_time = datetime.now()
    step = 0
    output_file = os.path.join(output_dir, f"filtered_{FLAGS.pl_domain}_part_label")
    # output_file = os.path.join(output_dir, "filtered_com_part_label")
    keep_count = 0
    batch_index = 0
    f = codecs.open(output_file, 'w', encoding='utf-8')

    while True:
        ground_truth = []
        predictions = []
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
            true_batch_size = len(input_ids)
            ground_truth.extend([label_ids[i, :length[i]].tolist() for i in range(true_batch_size)])
            predictions.extend([prediction[i, :length[i]].tolist() for i in range(true_batch_size)])
            texts_ids = [input_ids[i, :length[i].tolist()] for i in range(true_batch_size)]
            tokens = list(map(lambda x: tokenizer.convert_ids_to_tokens(x), texts_ids))
            texts = [list(map(lambda t: utils.printable_text(t), token)) for token in tokens]
            for index, (gt, pred, txt, inputs) in enumerate(zip(ground_truth, predictions, texts, input_ids)):
                if keep(gt, pred):
                    keep_count += 1
                    f.write(f"{batch_index * FLAGS.batch_size + index}\n")
                    for g, p, t, i in zip(gt, pred, txt, inputs):
                        f.write(f"{t}  {tokenizer.inv_type_vocab[i[6]]}  {label2str(g)}  {label2str(p)}"
                                f"{'  ◁' if g[p] == 0 else ''}\n")
                    f.write("\n")

            if step % 1000 == 0 and show_info:
                now_time = datetime.now()
                tf.logging.info(
                    f"Step: {step} ({(now_time - start_time).total_seconds():.2f} sec)")
                start_time = now_time
            batch_index += 1
        except tf.errors.OutOfRangeError:
            tf.logging.info(
                f"Finish Keep: {keep_count}")
            break


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model")
    tf.app.run()
