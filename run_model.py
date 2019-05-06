import random
from datetime import datetime

import utils
import os
from models import SegmentModel
import tensorflow as tf
import process
import numpy as np
from checkmate import BestCheckpointSaver, get_all_checkpoint
import heapq

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
    model_class, config, dim_info, dict_builder, processor, tokenizer, data_augmenter = prepare_form_config(FLAGS)
    train_features = []
    dev_features = []
    part_label_features = []
    test_features_map = {}
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        train_features = process.convert_examples_to_features(
            examples=train_examples, tokenizer=tokenizer, dict_builder=dict_builder,
            label_map=processor.get_labels())
        del train_examples
    if FLAGS.do_eval:
        dev_examples = processor.get_dev_examples(FLAGS.data_dir)
        dev_features = process.convert_examples_to_features(
            examples=dev_examples, tokenizer=tokenizer, dict_builder=dict_builder,
            label_map=processor.get_labels())
        del dev_examples
    if FLAGS.pl_domain is not None and FLAGS.do_train:
        if not FLAGS.multitag:
            raise ValueError("part label train must use multi tag!")
        part_label_examples = processor.get_pl_examples(FLAGS.data_dir, FLAGS.pl_domain)
        part_label_features = process.convert_examples_to_features(
            examples=part_label_examples, tokenizer=tokenizer, dict_builder=dict_builder,
            label_map=processor.get_labels())
        del part_label_examples
    if FLAGS.do_predict:
        if FLAGS.test_domain is not None:
            domains = FLAGS.test_domain.split(",")
            test_features_map = {domain: process.convert_examples_to_features(
                examples=processor.get_test_examples(FLAGS.data_dir, domain),
                tokenizer=tokenizer, dict_builder=dict_builder,
                label_map=processor.get_labels()) for domain in domains}
        else:
            test_features_map = {"test": process.convert_examples_to_features(
                examples=processor.get_test_examples(FLAGS.data_dir, None),
                tokenizer=tokenizer, dict_builder=dict_builder,
                label_map=processor.get_labels())}

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        tf.set_random_seed(31415926)

        # train & eval
        model = SegmentModel.LowLevelModel(model_class,
                                           dim_info=dim_info,
                                           config=config, init_checkpoint=FLAGS.init_checkpoint, tokenizer=tokenizer,
                                           init_embedding=FLAGS.init_embedding, learning_rate=FLAGS.learning_rate)

        sess.run(tf.global_variables_initializer())

        # if FLAGS.pl_domain is not None:
        #     model_path = os.path.join(FLAGS.output_dir, f"{FLAGS.pl_domain}_model.ckpt")
        # else:
        #     model_path = os.path.join(FLAGS.output_dir, "model.ckpt")
        # saver = tf.train.Saver()

        if FLAGS.do_train:

            saver = BestCheckpointSaver(
                save_dir=FLAGS.output_dir,
                num_to_keep=3,
                maximize=True
            )
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

                if len(part_label_features) == 0:
                    _, _, _, total_loss, total_step = full_label_running(
                        sess, model, train_features, dim_info, config,
                        batch_size=FLAGS.train_batch_size, is_training=True, show_info=True)
                else:
                    if FLAGS.mix_pl_data:
                        total_loss, total_step = mix_training(
                            sess, model, train_features, part_label_features, dim_info, config,
                            batch_size=FLAGS.train_batch_size, batch_base=not FLAGS.corpus_weighting, show_info=True)
                    else:
                        total_pl_loss = 0
                        total_pl_step = 0
                        if epoch % FLAGS.whole_pl_training_epoch == 0:
                            total_pl_loss, total_pl_step = part_label_training(
                                sess, model, part_label_features, dim_info, config,
                                batch_size=FLAGS.train_batch_size, show_info=True)

                        _, _, _, total_loss, total_step = full_label_running(
                            sess, model, train_features, dim_info, config,
                            batch_size=FLAGS.train_batch_size, is_training=True, show_info=True)

                        total_loss += total_pl_loss
                        total_step += total_pl_step

                avg_loss = total_loss / total_step
                now_time = datetime.now()
                tf.logging.info(
                    f"Epoch: {epoch} Average Loss: {avg_loss} ({(now_time - start_time).total_seconds():.2f} sec)")

                if FLAGS.do_eval:
                    dev_ground_true, dev_prediction, dev_texts, dev_loss, dev_step = full_label_running(
                        sess, model, dev_features, dim_info, config, batch_size=FLAGS.eval_batch_size,
                        is_training=False)

                    p, r, f = processor.evaluate_word_PRF(dev_prediction, dev_ground_true)

                    # if f > best_valid_f1:
                    #     best_valid_f1 = f
                    #     best_epoch = epoch
                    #     saver.save(sess, model_path)
                    #     if FLAGS.debug_mode:
                    #         tokens = list(map(lambda x: tokenizer.convert_ids_to_tokens(x), dev_texts))
                    #         texts = [list(map(lambda t: utils.printable_text(t), token)) for token in tokens]
                    #         processor.convert_word_segmentation(texts, dev_prediction, FLAGS.output_dir, "dev")
                    #         processor.convert_word_segmentation(texts, dev_ground_true, FLAGS.output_dir, "dev_golden")
                    # else:
                    #     if epoch - best_epoch >= FLAGS.early_stop_epochs and FLAGS.early_stop:
                    #         tf.logging.info(f"Early Stop Best F1: {best_valid_f1}")
                    #         break
                    if saver.handle(f, sess, epoch, FLAGS.pl_domain if FLAGS.pl_domain else None):
                        heapq.heappush(best_heap, (f, epoch))
                        if len(best_heap) > 3:
                            heapq.heappop(best_heap)
                        best_epoch = epoch
                    else:
                        if epoch - best_epoch >= FLAGS.early_stop_epochs and FLAGS.early_stop:
                                tf.logging.info(f"Early Stop Best F1: {best_valid_f1}")
                                break

                    tf.logging.info("Epoch: %d Dev Dataset Precision: %.5f Recall: %.5f F1: %.5f" % (epoch, p, r, f))
                    for rank, (top_f, top_epoch) in enumerate(sorted(best_heap, reverse=True)):
                        tf.logging.info("Top %d: Epoch: %d F1: %.5f" % (rank + 1, top_epoch, top_f))

                if FLAGS.debug_mode:
                    for domain, test_features in test_features_map.items():
                        predict_ground_true, predict_prediction, predict_texts, predict_loss, predict_step = full_label_running(
                            sess, model, test_features, dim_info, config,
                            batch_size=FLAGS.predict_batch_size, is_training=False)
                        p, r, f = processor.evaluate_word_PRF(predict_prediction, predict_ground_true)
                        tf.logging.info('%s Domain: %s Test: P:%f R:%f F1:%f' % (FLAGS.data_dir, domain, p, r, f))
                        tokens = list(map(lambda x: tokenizer.convert_ids_to_tokens(x), predict_texts))
                        texts = [list(map(lambda t: utils.printable_text(t), token)) for token in tokens]
                        processor.convert_word_segmentation(texts, predict_prediction, FLAGS.output_dir,
                                                            f"{domain}_predict")
                        processor.convert_word_segmentation(texts, predict_ground_true,
                                                            FLAGS.output_dir, f"{domain}_predict_golden")

            now_time = datetime.now()
            tf.logging.info(f"Train Spent: {now_time - very_start_time} sec")

        if FLAGS.do_predict:
            checkpoints = get_all_checkpoint(FLAGS.output_dir)
            saver = tf.train.Saver()
            for epoch, checkpoint in sorted(checkpoints):
                saver.restore(sess, checkpoint)
                print(f"Check Point at Epoch {epoch}:")
                # test
                for domain, test_features in test_features_map.items():
                    predict_ground_true, predict_prediction, predict_texts, predict_loss, predict_step = full_label_running(
                        sess, model, test_features, dim_info, config,
                        batch_size=FLAGS.predict_batch_size, is_training=False)
                    p, r, f = processor.evaluate_word_PRF(predict_prediction, predict_ground_true)
                    print('%s Domain: %s Test: P:%f R:%f F1:%f' % (FLAGS.data_dir, domain, p, r, f))
                    tokens = list(map(lambda x: tokenizer.convert_ids_to_tokens(x), predict_texts))
                    texts = [list(map(lambda t: utils.printable_text(t), token)) for token in tokens]
                    processor.convert_word_segmentation(texts, predict_prediction, FLAGS.output_dir, f"{domain}_predict")
                    processor.convert_word_segmentation(texts, predict_ground_true,
                                                        FLAGS.output_dir, f"{domain}_predict_golden")


def mix_training(sess, model, features, pl_features, dim_info, config, batch_size, batch_base=False, show_info=False):
    total_loss = 0.
    step = 0
    start_time = datetime.now()
    if batch_base:
        data_iterator = utils.data_batch_base_mix_iterator(features, pl_features,
                                                           dim_info=dim_info, batch_size=batch_size)
    else:
        data_iterator = utils.data_iterator(features, pl_features=pl_features,
                                            dim_info=dim_info, batch_size=batch_size, shuffle=True)

    for step, (input_ids, input_dicts, label_ids, seq_length) in enumerate(data_iterator):
        loss, _ = sess.run(
            [model.total_loss, model.train_op],
            feed_dict={model.input_ids: input_ids,
                       model.input_dicts: input_dicts,
                       model.label_ids: label_ids,
                       model.seq_length: seq_length,
                       model.dropout_keep_prob: 1 - config.hidden_dropout_prob}
        )

        total_loss += loss
        step += 1
        if step % 100 == 0 and show_info:
            now_time = datetime.now()
            tf.logging.info(
                f"Step: {step} Loss: {total_loss / step} ({(now_time - start_time).total_seconds():.2f} sec)")
            start_time = now_time
    return total_loss, step


def part_label_training(sess, model, features, dim_info, config, batch_size, show_info=False):
    total_loss = 0.
    step = 0
    start_time = datetime.now()
    for step, (input_ids, input_dicts, label_ids, seq_length) in enumerate(
            utils.data_iterator(features, dim_info=dim_info, batch_size=batch_size, shuffle=True)):
        loss, _ = sess.run(
            [model.loss, model.pl_train_op],
            feed_dict={model.input_ids: input_ids,
                       model.input_dicts: input_dicts,
                       model.label_ids: label_ids,
                       model.seq_length: seq_length,
                       model.dropout_keep_prob: 1 - config.hidden_dropout_prob}
        )

        total_loss += loss
        step += 1
        if step % 100 == 0 and show_info:
            now_time = datetime.now()
            tf.logging.info(
                f"Step: {step} Loss: {total_loss / step} ({(now_time - start_time).total_seconds():.2f} sec)")
            start_time = now_time
    return total_loss, step


def full_label_running(sess, model, features, dim_info, config, batch_size, is_training=False, show_info=False):
    total_loss = 0.
    step = 0
    ground_true = []
    predictions = []
    texts = []
    start_time = datetime.now()
    for step, (input_ids, input_dicts, label_ids, seq_length) in enumerate(
            utils.data_iterator(features, dim_info=dim_info, batch_size=batch_size, shuffle=is_training)):
        if is_training:
            loss, length, prediction, _ = sess.run(
                [model.total_loss, model.seq_length, model.prediction, model.train_op],
                feed_dict={model.input_ids: input_ids,
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
        true_batch_size = len(input_ids)
        if dim_info.label_dim != 1:
            ground_true.extend([np.argmax(label_ids[i, :length[i]], axis=-1).tolist() for i in range(true_batch_size)])
        else:
            ground_true.extend([label_ids[i, :length[i]].tolist() for i in range(true_batch_size)])
        predictions.extend([prediction[i, :length[i]].tolist() for i in range(true_batch_size)])
        texts.extend([input_ids[i, :length[i].tolist()] for i in range(true_batch_size)])
        if step % 100 == 0 and show_info:
            now_time = datetime.now()
            tf.logging.info(
                f"Step: {step} Loss: {total_loss / step} ({(now_time - start_time).total_seconds():.2f} sec)")
            start_time = now_time
    return ground_true, predictions, texts, total_loss, step


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("model")
    tf.app.run()
