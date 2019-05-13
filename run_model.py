import random
from datetime import datetime

import models.ModelAdapter
import utils
from models import SegmentModel
import tensorflow as tf
from process import feature_builder, dataset
import numpy as np

from prepare import prepare_form_config

flags = tf.flags

FLAGS = flags.FLAGS

## Dataset Input parameters
flags.DEFINE_string("pl_domain", None,
                    "the domain use for part label training")

flags.DEFINE_string("test_domain", None,
                    "the domain for test use")

## Running parameters
flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 128, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 128, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 128, "Total batch size for predict.")

flags.DEFINE_float(
    "gpu_memory", 0.95,
    "how many memory can model use")

## Training Behave parameters
flags.DEFINE_bool("debug_mode", False, "Whether to run training.")

flags.DEFINE_bool(
    "early_stop", True,
    "Whether to use early stop.")

flags.DEFINE_integer(
    "early_stop_epochs", 30,
    "Whether to use early stop.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 200,
                     "Total number of training epochs to perform.")

flags.DEFINE_bool(
    "mix_pl_data", False,
    "Whether to use mix full label data with part label data")

flags.DEFINE_bool(
    "corpus_weighting", True,
    "Whether to use mix full label data with part label data")

flags.DEFINE_integer(
    "whole_pl_training_epoch", 5,
    "train part label data each N epoch")

## Model parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("init_embedding", None, "Initial Embedding.")


def main(_):
    if FLAGS.do_train:
        tf.logging.set_verbosity(tf.logging.INFO)
        np.random.seed(31415926)
        random.seed(31415926)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `train`, `eval` or `predict' must be select.")
    model_class, config, dim_info, processor, extractors, data_augmenter = prepare_form_config(FLAGS)
    test_dataset_map = {}
    cxt_feature_extractor = extractors["input_ids"]
    feat_builder = feature_builder.FeatureBuilder(extractors=extractors,
                                                  label_map=processor.get_labels())
    train_features = []
    part_label_dataset = None
    train_dataset = None
    dev_dataset = None
    if FLAGS.do_train:
        train_examples = processor.get_examples(data_dir=FLAGS.data_dir, example_type="train")
        train_features = feat_builder.build_features_from_examples(examples=train_examples)
        train_dataset = dataset.PaddingDataset(train_features, batch_size=FLAGS.train_batch_size,
                                               dim_info=dim_info)
        del train_examples
    if FLAGS.do_eval:
        dev_examples = processor.get_examples(data_dir=FLAGS.data_dir, example_type="dev")
        dev_features = feat_builder.build_features_from_examples(examples=dev_examples)
        dev_dataset = dataset.PaddingDataset(dev_features, batch_size=FLAGS.eval_batch_size,
                                             dim_info=dim_info)
        del dev_examples

    if FLAGS.pl_domain is not None and FLAGS.do_train:
        if not FLAGS.multitag:
            raise ValueError("part label train must use multi tag!")
        part_label_examples = processor.get_examples(data_dir=FLAGS.data_dir, example_type="pl", domain=FLAGS.pl_domain)
        part_label_features = feat_builder.build_features_from_examples(examples=part_label_examples)
        if FLAGS.mix_pl_data:
            if FLAGS.corpus_weighting:
                part_label_dataset = dataset.CorpusWeightingDataset([train_features, part_label_features],
                                                                    [10000, 10000], batch_size=FLAGS.train_batch_size,
                                                                    dim_info=dim_info)
            else:
                part_label_dataset = dataset.BatchMixDataset([train_features, part_label_features],
                                                             [1, 5], batch_size=FLAGS.train_batch_size,
                                                             dim_info=dim_info)
        else:
            part_label_dataset = dataset.PaddingDataset(part_label_features, batch_size=FLAGS.train_batch_size,
                                                        dim_info=dim_info)

        del part_label_examples
    if FLAGS.do_predict:
        if FLAGS.test_domain is not None:
            domains = FLAGS.test_domain.split(",")
            test_dataset_map = {domain: dataset.PaddingDataset(
                feat_builder.build_features_from_examples(
                    examples=processor.get_examples(data_dir=FLAGS.data_dir, example_type="test", domain=domain)),
                batch_size=FLAGS.predict_batch_size,
                dim_info=dim_info
            ) for domain in domains}
        else:
            test_dataset_map = {"test": dataset.PaddingDataset(
                feat_builder.build_features_from_examples(
                    examples=processor.get_examples(data_dir=FLAGS.data_dir, example_type="test")),
                batch_size=FLAGS.predict_batch_size,
                dim_info=dim_info
            )}

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        tf.set_random_seed(31415926)

        # train & eval
        model = models.ModelAdapter.ModelAdapter(model_class,
                                                 dim_info=dim_info,
                                                 config=config, init_checkpoint=FLAGS.init_checkpoint,
                                                 tokenizer=cxt_feature_extractor,
                                                 init_embedding=FLAGS.init_embedding, learning_rate=FLAGS.learning_rate)

        sess.run(tf.global_variables_initializer())

        # if FLAGS.pl_domain is not None:
        #     model_path = os.path.join(FLAGS.output_dir, f"{FLAGS.pl_domain}_model.ckpt")
        # else:
        #     model_path = os.path.join(FLAGS.output_dir, "model.ckpt")
        # saver = tf.train.Saver()

        if FLAGS.do_train:

            # saver = BestCheckpointSaver(
            #     save_dir=FLAGS.output_dir,
            #     num_to_keep=3,
            #     maximize=True
            # )
            best_valid_f1 = 0.
            best_epoch = 0
            best_heap = []
            very_start_time = datetime.now()
            for epoch in range(FLAGS.num_train_epochs):
                start_time = datetime.now()
                if FLAGS.pl_domain is not None:
                    tf.logging.info(f"Epoch: {epoch} Domain: {FLAGS.pl_domain}")
                if epoch < 10:
                    model.assign_lr(sess, FLAGS.learning_rate)
                if 10 <= epoch < 15:
                    model.assign_lr(sess, FLAGS.learning_rate * config.lr_decay)
                if 15 <= epoch < 20:
                    model.assign_lr(sess, FLAGS.learning_rate * config.lr_decay ** 2)
                if 20 <= epoch < 25:
                    model.assign_lr(sess, FLAGS.learning_rate * config.lr_decay ** 3)
                if 25 <= epoch:
                    model.assign_lr(sess, FLAGS.learning_rate * config.lr_decay ** 4)

                if part_label_dataset is None:
                    _, _, _, total_loss, total_step = dataset_running(
                        sess, model, train_dataset, dim_info, config, is_training=True, show_info=True)
                else:
                    if FLAGS.mix_pl_data:
                        total_loss, total_step = dataset_running(
                            sess, model, part_label_dataset, dim_info, config, is_training=True, show_info=True)
                    else:
                        total_pl_loss = 0
                        total_pl_step = 0
                        if epoch % FLAGS.whole_pl_training_epoch == 0:
                            total_pl_loss, total_pl_step = dataset_running(
                            sess, model, part_label_dataset, dim_info, config, is_training=True, show_info=True)

                        _, _, _, total_loss, total_step = dataset_running(
                            sess, model, train_dataset, dim_info, config, is_training=True, show_info=True)

                        total_loss += total_pl_loss
                        total_step += total_pl_step

                avg_loss = total_loss / total_step
                now_time = datetime.now()
                tf.logging.info(
                    f"Epoch: {epoch} Average Loss: {avg_loss} ({(now_time - start_time).total_seconds():.2f} sec)")

                if FLAGS.do_eval:
                    dev_ground_true, dev_prediction, dev_texts, dev_loss, dev_step = dataset_running(
                        sess, model, dev_dataset, dim_info, config,
                        is_training=False)

                    p, r, f = processor.evaluate(dev_prediction, dev_ground_true)

                    # if saver.handle(f, sess, epoch, FLAGS.pl_domain if FLAGS.pl_domain else None):
                    #     heapq.heappush(best_heap, (f, epoch))
                    #     if len(best_heap) > 3:
                    #         heapq.heappop(best_heap)
                    #     best_epoch = epoch
                    # else:
                    #     if epoch - best_epoch >= FLAGS.early_stop_epochs and FLAGS.early_stop:
                    #             tf.logging.info(f"Early Stop Best F1: {best_valid_f1}")
                    #             break

                    tf.logging.info("Epoch: %d Dev Dataset Precision: %.5f Recall: %.5f F1: %.5f" % (epoch, p, r, f))
                    for rank, (top_f, top_epoch) in enumerate(sorted(best_heap, reverse=True)):
                        tf.logging.info("Top %d: Epoch: %d F1: %.5f" % (rank + 1, top_epoch, top_f))

                if FLAGS.debug_mode:
                    for domain, test_dataset in test_dataset_map.items():
                        predict_ground_truth, predict_prediction, predict_texts, predict_loss, predict_step = dataset_running(
                            sess, model, test_dataset, dim_info, config, is_training=False)
                        p, r, f = processor.evaluate(predict_prediction, predict_ground_truth)
                        tf.logging.info('%s Domain: %s Test: P:%f R:%f F1:%f' % (FLAGS.data_dir, domain, p, r, f))
                        tokens = list(map(lambda x: cxt_feature_extractor.restore(x), predict_texts))
                        texts = [list(map(lambda t: utils.printable_text(t), token)) for token in tokens]
                        processor.segment(texts, predict_prediction, FLAGS.output_dir,
                                     f"{domain}_predict")
                        processor.segment(texts, predict_ground_truth,
                                     FLAGS.output_dir, f"{domain}_predict_golden")

            now_time = datetime.now()
            tf.logging.info(f"Train Spent: {now_time - very_start_time} sec")

        # if FLAGS.do_predict:
        #     checkpoints = get_all_checkpoint(FLAGS.output_dir)
        #     saver = tf.train.Saver()
        #     for epoch, checkpoint in sorted(checkpoints):
        #         saver.restore(sess, checkpoint)
        #         print(f"Check Point at Epoch {epoch}:")
        #         # test
        #         for domain, test_dataset in test_dataset_map.items():
        #             predict_ground_truth, predict_prediction, predict_texts, predict_loss, predict_step = dataset_running(
        #                 sess, model, test_dataset, dim_info, config, is_training=False)
        #             p, r, f = processor.evaluate(predict_prediction, predict_ground_truth)
        #             print('%s Domain: %s Test: P:%f R:%f F1:%f' % (FLAGS.data_dir, domain, p, r, f))
        #             tokens = list(map(lambda x: cxt_feature_extractor.restore(x), predict_texts))
        #             texts = [list(map(lambda t: utils.printable_text(t), token)) for token in tokens]
        #             processor.segment(texts, predict_prediction, FLAGS.output_dir, f"{domain}_predict")
        #             processor.segment(texts, predict_ground_truth,
        #                          FLAGS.output_dir, f"{domain}_predict_golden")


def dataset_running(sess, model, running_dataset, dim_info, config, is_training=False, show_info=False):
    total_loss = 0.
    step = 0
    ground_truth = []
    predictions = []
    texts = []
    start_time = datetime.now()
    for step, (input_features, label_ids, seq_length) in enumerate(running_dataset):
        input_ids = input_features["input_ids"]
        input_dicts = input_features["input_dicts"]
        if is_training:
            loss, length, prediction, _ = sess.run(
                [model.total_loss, model.seq_length, model.prediction, model.train_op],
                feed_dict={model.input_features: input_features,
                           model.input_dicts: input_dicts,
                           model.label_ids: label_ids,
                           model.seq_length: seq_length,
                           model.dropout_keep_prob: 1 - config.hidden_dropout_prob}
            )
        else:
            loss, length, prediction = sess.run(
                [model.total_loss, model.seq_length, model.prediction],
                feed_dict={model.input_ids: input_ids,
                           model.input_dicts: input_dicts,
                           model.label_ids: np.zeros_like(label_ids),
                           model.seq_length: seq_length,
                           model.dropout_keep_prob: 1}
            )

        total_loss += loss
        step += 1

        if step % 100 == 0 and show_info:
            now_time = datetime.now()
            tf.logging.info(
                f"Step: {step} Loss: {total_loss / step} ({(now_time - start_time).total_seconds():.2f} sec)")
            start_time = now_time

        if not is_training:
            true_batch_size = len(input_ids)
            if dim_info.label_dim != 1:
                ground_truth.extend([np.argmax(label_ids[i, :length[i]], axis=-1).tolist() for i in range(true_batch_size)])
            else:
                ground_truth.extend([label_ids[i, :length[i]].tolist() for i in range(true_batch_size)])
            predictions.extend([prediction[i, :length[i]].tolist() for i in range(true_batch_size)])
            texts.extend([input_ids[i, :length[i].tolist()] for i in range(true_batch_size)])
    if is_training:
        return total_loss, step
    else:
        return ground_truth, predictions, texts, total_loss, step


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("model")
    tf.app.run()
