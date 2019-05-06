import os

import tensorflow as tf
from prepare import prepare_form_config

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_path", None,
    "The input data dir. Should contain the .tf_record files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "export_model_dir", None,
    "The input data dir. Should contain the .tf_record files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "model_version", None,
    "The input data dir. Should contain the .tf_record files (or other data files) "
    "for the task.")


def main(_):
    model_class, config, dim_info, dict_builder, processor, tokenizer, data_augmenter = prepare_form_config(FLAGS)

    with tf.get_default_graph().as_default():
        # 定义你的输入输出以及计算图

        input_ids = tf.placeholder(dtype=tf.int64, shape=[None, None, dim_info.input_dim], name='input_ids')
        input_dicts = tf.placeholder(dtype=tf.int64, shape=[None, None, dim_info.dict_dim], name='input_dicts')
        if dim_info.label_dim == 1:
            label_ids = tf.placeholder(dtype=tf.int64, shape=[None, None], name='label_ids')
        else:
            label_ids = tf.placeholder(dtype=tf.int64, shape=[None, None, dim_info.label_dim], name='label_ids')
        seq_length = tf.placeholder(dtype=tf.int64, shape=[None], name='seq_length')

        dropout_keep_prob = tf.constant(1, dtype=tf.float32, name='dropout_keep_prob')
        features = {
            "input_ids": input_ids,
            "input_dicts": input_dicts,
            "label_ids": label_ids,
            "seq_length": seq_length
        }

        model = model_class(config, features, dropout_keep_prob, None, None)

        _, _, prediction, _ = model.get_all_results()
        output_result = model.prediction

        saver = tf.train.Saver()

        # 导入你已经训练好的模型.ckpt文件
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path,
                                      os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            # 定义导出模型的各项参数
            # 定义导出地址
            export_path_base = FLAGS.export_model_dir
            tf.gfile.MakeDirs(FLAGS.export_model_dir)
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(FLAGS.model_version)))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # 定义Input tensor info，需要前面定义的input_images
            tensor_info_input_ids = tf.saved_model.utils.build_tensor_info(input_ids)
            tensor_info_input_dicts = tf.saved_model.utils.build_tensor_info(input_dicts)
            tensor_info_label_ids = tf.saved_model.utils.build_tensor_info(label_ids)
            tensor_info_seq_length = tf.saved_model.utils.build_tensor_info(seq_length)

            # 定义Output tensor info，需要前面定义的output_result
            tensor_info_output = tf.saved_model.utils.build_tensor_info(output_result)

            # 创建预测签名
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_ids': tensor_info_input_ids,
                            'input_dicts': tensor_info_input_dicts,
                            'label_ids': tensor_info_label_ids,
                            'seq_length': tensor_info_seq_length},
                    outputs={'result': tensor_info_output},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict': prediction_signature})

            # 导出模型
            builder.save(as_text=True)
            print('Done exporting!')


if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    flags.mark_flag_as_required("export_model_dir")
    flags.mark_flag_as_required("model_version")
    flags.mark_flag_as_required("model")
    tf.app.run()
