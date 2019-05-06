import json
import os

from matplotlib import gridspec

import process
from models import SegmentModel
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from prepare import prepare_form_config
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("pretrain_dir", None, "")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_class, config, dim_info, dict_builder, processor, tokenizer, data_augmenter = prepare_form_config(FLAGS)
    # ckpt_state = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
    # model_path = os.path.join(FLAGS.pretrain_dir,
    #                           os.path.basename(ckpt_state.model_checkpoint_path))
    model_path = "output\\nova_dot\\ctb_l2_clip\\zx\\zx_model.ckpt-115"
    # model_path = "output\\baseline\\pd\\model.ckpt-60"
    processor = process.PlainLineProcessor()
    examples = processor.get_single_example("私营企业主成为社会主义事业的建设者")
    example = list(examples)[0]
    feature = process.convert_single_example(0, example, tokenizer=tokenizer, dict_builder=dict_builder,
                                             label_map=processor.get_labels())
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        # train & eval
        model = SegmentModel.LowLevelModel(model_class,
                                           dim_info=dim_info,
                                           config=config, init_checkpoint=None, tokenizer=tokenizer,
                                           init_embedding=None, learning_rate=0.01)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        feature_json = {
            "input_ids": feature.input_ids,
            "input_dicts": feature.input_dicts,
            "label_ids": feature.label_ids,
            "seq_length": feature.seq_length}
        input_ids = feature.input_ids
        input_dicts = feature.input_dicts
        label_ids = feature.label_ids
        seq_length = feature.seq_length
        feature_json = json.dumps(feature_json)
        print(feature_json)

        def id2label(lid):
            if lid == 0:
                return "B"
            elif lid == 1:
                return "M"
            elif lid == 2:
                return "E"
            else:
                return "S"

        gvar_map = {var.name: var for var in tf.global_variables()}
        transitions = gvar_map['output/transitions:0']
        #
        pure_noise_matrix = gvar_map['noise_correct/noise_matrix:0']
        eye_matrix = gvar_map['noise_correct/eye_matrix:0']
        rate = gvar_map["noise_correct/rate:0"]
        norm_rate = tf.sigmoid(rate)
        noise_matrix = tf.broadcast_to(norm_rate, [4, 4]) * pure_noise_matrix + \
                       tf.broadcast_to((1 - norm_rate), [4, 4]) * eye_matrix

        # fig = plt.figure()
        # sns.heatmap(sess.run(transitions), square=True,
        #             xticklabels=["B", "M", "E", "S"], yticklabels=["B", "M", "E", "S"], annot=True, fmt=".2", vmin=-4, vmax=4, cmap="YlGnBu")
        # plt.show()
        # plt.close(fig)

        f, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 6))
        ax[0].set_xlabel("transitions")
        sns.heatmap(sess.run(pure_noise_matrix), annot=True, vmax=1, vmin=0, cmap="YlGnBu",
                    xticklabels=["B", "M", "E", "S"], yticklabels=["B", "M", "E", "S"], ax=ax[1], fmt=".2%")
        ax[1].set_xlabel("pure_noise_matrix")
        sns.heatmap(sess.run(noise_matrix), annot=True, vmax=1, vmin=0, cmap="YlGnBu",
                    xticklabels=["B", "M", "E", "S"], yticklabels=["B", "M", "E", "S"], ax=ax[2], fmt=".2%")
        ax[2].set_xlabel("noise_matrix")
        print(sess.run(rate))
        plt.show()
        plt.close(f)

        # scores = model.model.scores
        # p = tf.nn.softmax(scores, axis=-1)
        # nm = tf.constant(np.load("data/pl/PD_ZYD/com_noise.npy"), dtype=tf.float32)
        # corr_p = tf.einsum("ji, blj->bli", nm, p)
        # gt = tf.expand_dims(
        #     tf.constant(np.array([[1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]] + [[1, 1, 1, 1]] * 11 + [[0, 0, 1, 1]]), dtype=tf.float32), axis=0)
        # ce_loss = -tf.einsum("bld, bld->bl", gt, tf.log(tf.clip_by_value(corr_p, clip_value_min=1e-16, clip_value_max=1)))
        # dot_loss = 1 - tf.clip_by_value(tf.einsum("bld, bld->bl", gt, corr_p), clip_value_min=0, clip_value_max=1)
        # log_dot_loss = -tf.log(tf.clip_by_value(tf.einsum("bld, bld->bl", gt, corr_p), clip_value_min=1e-16, clip_value_max=1))
        # prediction, scores, p, corr_p, dot_loss, log_dot_loss, ce_loss = sess.run(
        #     [model.prediction, scores, p, corr_p, dot_loss, log_dot_loss, ce_loss],
        #     feed_dict={model.input_ids: np.expand_dims(input_ids, axis=0),
        #                model.input_dicts: np.expand_dims(input_dicts, axis=0),
        #                model.label_ids: np.expand_dims(label_ids, axis=0),
        #                model.seq_length: np.expand_dims(seq_length, axis=0),
        #                model.dropout_keep_prob: 1}
        # )
        # print(" ".join(map(str, prediction[0])))
        # print(" ".join(map(id2label, prediction[0])))
        # label_ids = np.array([[1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]] + [[1, 1, 1, 1]] * 11 + [[0, 0, 1, 1]])
        # # f, ax = plt.subplots(ncols=1, nrows=7, figsize=(20, 28))
        # fig = plt.figure(figsize=(20, 28))
        # gs1 = gridspec.GridSpec(4, 1)
        # ax0 = fig.add_subplot(gs1[0])
        # ax1 = fig.add_subplot(gs1[1])
        # ax2 = fig.add_subplot(gs1[2])
        # ax3 = fig.add_subplot(gs1[3])
        # sns.set(font_scale=1.5)
        # sns.heatmap(np.transpose(scores[0]), square=True, xticklabels=False, yticklabels=["B", "M", "E", "S"], ax=ax0, cbar=False, annot=True)
        # sns.heatmap(np.transpose(p[0]), square=True, xticklabels=False, yticklabels=["B", "M", "E", "S"], ax=ax1, cbar=False, annot=True, fmt=".2%")
        # sns.heatmap(np.transpose(corr_p[0]), square=True, xticklabels=False, yticklabels=["B", "M", "E", "S"], ax=ax2, cbar=False, annot=True, fmt=".2%")
        # sns.heatmap(np.transpose(label_ids), square=True, xticklabels=False, yticklabels=["B", "M", "E", "S"], ax=ax3, cbar=False, annot=True)
        # gs1.tight_layout(fig, rect=[0, 0.20, 1, 1])
        # gs2 = gridspec.GridSpec(3, 1)
        # ax4 = fig.add_subplot(gs2[0])
        # ax5 = fig.add_subplot(gs2[1])
        # ax6 = fig.add_subplot(gs2[2])
        # sns.heatmap(ce_loss, square=True, xticklabels=False, yticklabels=["L"], ax=ax4,
        #             cbar=False, annot=True, vmin=0, vmax=10)
        # sns.heatmap(log_dot_loss, square=True, xticklabels=False, yticklabels=["L"], ax=ax5,
        #             cbar=False, annot=True, vmin=0, vmax=10)
        # sns.heatmap(dot_loss, square=True, xticklabels=False, yticklabels=["L"], ax=ax6,
        #             cbar=False, annot=True, vmin=0, vmax=10)
        # gs2.tight_layout(fig, rect=[0, -0.05, 1, 0.31])
        # left = min(gs1.left, gs2.left)
        # right = max(gs1.right, gs2.right)
        #
        # gs1.update(left=left, right=right)
        # gs2.update(left=left, right=right)
        # plt.show()

#         run_example(sess, model, example)


def run_example(sess, model, example):
        input_ids = example["input_ids"]
        input_dicts = example["input_dicts"]
        label_ids = example["label_ids"]
        seq_length = example["seq_length"]

        prediction = sess.run(
            [model.prediction],
            feed_dict={model.input_ids: input_ids,
                       model.input_dicts: input_dicts,
                       model.label_ids: label_ids,
                       model.seq_length: seq_length,
                       model.dropout_keep_prob: 1}
        )

        return prediction


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("config_file")
    # flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("pretrain_dir")
    tf.app.run()
