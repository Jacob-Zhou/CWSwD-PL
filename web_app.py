import re
from datetime import datetime
import json
import os
import platform

import grpc
import six
import tornado.ioloop
import tornado.web
from tornado.routing import URLSpec

import dictionary_builder
import process
import tokenization

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np

import utils
from utils import DimInfo

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


class Config:
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BaselineConfig` from a Python dictionary of parameters."""
        config = cls()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


class SegmentHandler(tornado.web.RequestHandler):
    def initialize(self, server_url, model_setting, tokenizers, dict_builders):
        self.server_url = server_url
        self.model_setting = model_setting
        processor = process.PlainLineProcessor()

        # todo
        def convert(x, model, standard):
            x = [sent for sent in x.split("\n") if len(sent.strip()) > 0]
            examples = processor.get_multi_example(x)
            examples = list(examples)
            setting = model_setting[model][standard]
            tokenizer = tokenizers[setting["tokenizer"]]
            dict_builder = dict_builders[setting["dict_builder"]]
            multitag = setting["multitag"]
            features = process.convert_examples_to_features(examples, tokenizer=tokenizer, dict_builder=dict_builder,
                                                            label_map=processor.get_labels())
            dim_info = DimInfo(tokenizer.dim, dict_builder.dim if dict_builder else 1, 4 if multitag else 1)
            padded_input_ids, padded_input_dicts, padded_label_ids, seq_length = utils.padding_features(
                features, dim_info=dim_info)
            features = {
                "input_ids": padded_input_ids,
                "input_dicts": padded_input_dicts,
                "seq_length": np.array(seq_length),
                "label_ids": padded_label_ids
            }
            batch_size = padded_input_ids.shape[0]
            return examples, features, batch_size

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
        model = self.get_argument("model")
        standard = self.get_argument("standard")
        examples, features, batch_size = self.convert(input_text, model, standard)
        response = await self.inference(features, model, standard)
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
        ## todo able to change to other model
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


def build_from_config(serving_path):
    serving_config_path = os.path.join(serving_path, "serving.json")
    with open(serving_config_path, "r", encoding="utf-8") as reader:
        text = reader.read()
    serving_json = json.loads(text)
    server_url = serving_json["server_url"]
    tokenizers_config = serving_json["tokenizers"]
    dict_builders_config = serving_json["dict_builders"]
    models = serving_json["models"]
    config_template_path = os.path.join(serving_path, serving_json["config_template_path"])
    with open(config_template_path, "r", encoding="utf-8") as reader:
        text = reader.read()
    config_template_json = json.loads(text)
    model_options = []
    model_configs = []
    model_setting = {}
    for model in models:
        model_option = {"value": model["name"], "label": model["alias"], "children": []}
        model_setting[f"{model['name']}"] = {}
        for standard in model["standards"]:
            model_option["children"].append({"value": standard["name"], "label": standard["alias"]})
            versions = ", ".join(
                map(lambda x: config_template_json["versions"].format(**{"version": x}), standard["versions"]))
            model_configs.append({"model": model["name"],
                                  "standard": standard["name"],
                                  "model_base_path": model["base_path"],
                                  "standard_path": standard["path"],
                                  "versions": versions})
            model_setting[f"{model['name']}"][f"{standard['name']}"] = {
                "tokenizer": standard["tokenizer"],
                "dict_builder": standard["dict_builder"],
                "multitag": model['multitag']
            }
        model_options.append(model_option)

    model_configs_inside = ", ".join(
        map(lambda x: config_template_json["config"].replace("'", '"').format(**x), model_configs))
    model_configs_text = config_template_json["model_config_list"].format(**{"configs": model_configs_inside})
    model_config_path = os.path.join(serving_path, "models.config")
    with open(model_config_path, "w", encoding="utf-8") as cfg_out:
        cfg_out.write(model_configs_text)

    class LazyLoader:
        def __init__(self, build_func):
            self.configs = {}
            self.items = {}
            self.build_func = build_func

        def __getitem__(self, item):
            if item in self.items:
                return self.items[item]
            else:
                if item in self.configs:
                    self.items[item] = self.build_func(self.configs[item])
                    return self.items[item]
                else:
                    raise KeyError(item)

        def __setitem__(self, key, value):
            if isinstance(value, Config):
                self.configs[key] = value
            else:
                raise ValueError("value should be type Config")

    tokenizers = LazyLoader(lambda cfg: tokenization.TrTeWinBiTokenizer(
        vocab_file=cfg.vocab_file, bigram_file=cfg.bigram_file,
        type_file=cfg.type_file, bigram_type_file=cfg.bigram_type_file,
        do_lower_case=cfg.do_lower_case, window_size=cfg.window_size))
    for tokenizer in tokenizers_config:
        tokenizer_config = Config.from_json_file(os.path.join(serving_path, tokenizer["config"]))
        tokenizers[tokenizer["name"]] = tokenizer_config

    dict_builders = LazyLoader(lambda cfg: dictionary_builder.DefaultDictionaryBuilder(
        cfg.dict_file,
        min_word_len=cfg.min_word_len,
        max_word_len=cfg.max_word_len))
    for dict_builder in dict_builders_config:
        dict_builder_config = Config.from_json_file(os.path.join(serving_path, dict_builder["config"]))
        dict_builders[dict_builder["name"]] = dict_builder_config

    return server_url, model_options, model_setting, tokenizers, dict_builders


def main(_):
    system = platform.system()
    serving_path = f"{os.getcwd()}/serving"
    os.system("docker kill cws_container")
    server_url, model_options, model_setting, tokenizers, dict_builders = build_from_config(serving_path)
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
        (r"/segment", SegmentHandler, dict(server_url=server_url,
                                           model_setting=model_setting,
                                           tokenizers=tokenizers,
                                           dict_builders=dict_builders)),
        (r"/models", ModelsHandler, dict(model_options=model_options)),
        (r"/css/(.*)", tornado.web.StaticFileHandler, dict(path="web/css")),
    ])
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    tf.app.run()
