import os
import platform
from datetime import datetime

import grpc
import jieba
import tornado.ioloop
import tornado.web

from process import data_processor, feature_builder, dataset

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np

from process.feature_builder import DimInfo
from serving.serving_utils import build_from_config, LazyConstructor

flags = tf.flags

FLAGS = flags.FLAGS

## Dataset Input parameters
flags.DEFINE_string("config_file", "config/web/config.json",
                    "the domain use for part label training")


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        with open('./web/main.html', 'rb') as f:
            self.write(f.read())
            self.finish()


class ModelsHandler(tornado.web.RequestHandler):
    def initialize(self, model_options):
        self.model_options = model_options

    def get(self):
        self.finish({'message': 'ok',
                     'state': "success",
                     'models': self.model_options})


class SegmentHandler(tornado.web.RequestHandler):
    def initialize(self, server_url, model_setting, cxt_feature_extractors, dict_feature_extractors):
        self.server_url = server_url
        self.model_setting = model_setting
        processor = data_processor.PlainLineDataProcessor()
        feature_builder_constructor = LazyConstructor(lambda extractors_name: feature_builder.FeatureBuilder(
            extractors={"input_ids": cxt_feature_extractors[extractors_name[0]],
                        "input_dicts": dict_feature_extractors[extractors_name[1]]},
            label_map=processor.get_labels()))

        def convert(x, solution):
            x = [sent for sent in x.split("\n") if len(sent.strip()) > 0]
            # x = [x]
            examples = processor.get_examples(data=x, example_type="multi")
            examples = list(examples)
            setting = model_setting[solution]
            feat_builder = feature_builder_constructor[setting["context_feature_extractor"],
                                                       setting["dictionary_feature_extractor"]]
            extractors = feat_builder.extractors

            multitag = setting["multitag"]

            features = feat_builder.build_features_from_examples(examples)
            dim_info = DimInfo({feature_name: feature.dim for feature_name, feature in extractors.items()},
                               4 if multitag else 1)

            padded_features, padded_label_ids, seq_length = dataset.PaddingDataset.padding_features(features,
                                                                                                    dim_info=dim_info)
            padded_input_ids = padded_features["input_ids"]
            padded_input_dicts = padded_features["input_dicts"]
            features = {
                "input_ids": padded_input_ids,
                "input_dicts": padded_input_dicts,
                "seq_length": np.array(seq_length),
                "label_ids": padded_label_ids
            }
            batch_size = padded_input_ids.shape[0]
            max_len = padded_input_ids.shape[1]
            return examples, features, batch_size, max_len

        self.convert = convert

    async def post(self):
        def id2label(lid):
            if lid == 0:
                return "B"
            elif lid == 1:
                return "M"
            elif lid == 2:
                return "E"
            else:
                return "S"

        def seg(inp):
            x, y = inp
            if y == 2 or y == 3:
                return x + " "
            else:
                return x

        def batch_seg(inp):
            x, y = inp
            return "".join(map(seg, zip(x.guid, y))).split()

        input_text = self.get_argument("input")
        solution = self.get_argument("solution")
        model = self.model_setting[solution]["model"]
        if model == "jieba":
            self.finish({'message': 'ok',
                         'state': "success",
                         'results': [jieba.lcut(input_text)]
                         })
            return
        standard = self.model_setting[solution]["standard"]
        examples, features, batch_size, max_len = self.convert(input_text, solution)
        if batch_size > 250 or batch_size * max_len > 51200:
            self.finish({'message': f'lines: {batch_size} > 250 or lines * max_line: {batch_size * max_len} > 51200',
                         'state': "fail"
                         })
            return

        start_time = datetime.now()
        response = await self.inference(features, model, standard)
        now_time = datetime.now()
        print(f"{(now_time - start_time).total_seconds():.2f} sec")
        results = np.array(response.outputs["result"].int_val).reshape((batch_size, -1))
        examples = examples
        self.finish({'message': 'ok',
                     'state': "success",
                     'results': list(map(batch_seg, zip(examples, results)))
                     })

    async def inference(self, features, model, standard):
        channel = grpc.insecure_channel(self.server_url)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = f"{model}/{standard}"  # 模型名称
        request.model_spec.signature_name = "predict"  # 签名名称
        # "images"是你导出模型时设置的输入名称
        request.inputs["input_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["input_ids"], dtype=tf.int64, shape=features["input_ids"].shape))
        request.inputs["input_dicts"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["input_dicts"], dtype=tf.int64,
                                              shape=features["input_dicts"].shape))
        request.inputs["seq_length"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["seq_length"], dtype=tf.int64,
                                              shape=features["seq_length"].shape))
        request.inputs["label_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["label_ids"], dtype=tf.int64, shape=features["label_ids"].shape))
        response = stub.Predict(request, 10.0)  # 5 secs timeout
        return response


def main(_):
    system = platform.system()
    serving_path = f"{os.getcwd()}/serving"
    os.system("docker kill cws_container")
    server_url, model_options, model_setting, cxt_feature_extractors, dict_feature_extractors, _ = build_from_config(
        serving_path)
    if system == "Windows":
        cmd = f'start /b docker run -t --name="cws_container" --rm -p 8500:8500 -p 8501:8501 ' \
            f'--mount type=bind,source="{serving_path}/models",target=/serving/models ' \
            f'--mount type=bind,source="{serving_path}/models.config",target=/serving/models.config -t ' \
            'tensorflow/serving --model_config_file=/serving/models.config'
    elif system == "Linux":
        cmd = f'docker run -t --name="cws_container" --rm -p 8500:8500 -p 8501:8501 ' \
            f'--mount type=bind,source="{serving_path}/models",target=/serving/models ' \
            f'--mount type=bind,source="{serving_path}/models.config",target=/serving/models.config -t ' \
            'tensorflow/serving --model_config_file=/serving/models.config &'
    else:
        raise SystemError("only support Windows and Linux")
    os.system(cmd)
    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/v0/segment", SegmentHandler, dict(server_url=server_url,
                                              model_setting=model_setting,
                                              cxt_feature_extractors=cxt_feature_extractors,
                                              dict_feature_extractors=dict_feature_extractors)),
        (r"/models", ModelsHandler, dict(model_options=model_options)),
        (r"/css/(.*)", tornado.web.StaticFileHandler, dict(path="web/css")),
    ])
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    tf.app.run()
