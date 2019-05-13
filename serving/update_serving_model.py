import tensorflow as tf
from google.protobuf import text_format
from tensorflow_serving.apis import model_service_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2
from tensorflow_serving.util import status_pb2

import grpc
import os

from serving.serving_utils import build_from_config


def main(_):
    channel = grpc.insecure_channel(':8500')
    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    request = model_management_pb2.ReloadConfigRequest()

    model_server_config = model_server_config_pb2.ModelServerConfig()
    _, _, _, _, _, model_configs_text = build_from_config(os.getcwd())
    text_format.Parse(model_configs_text, model_server_config)
    request.config.CopyFrom(model_server_config)
    responese = stub.HandleReloadConfigRequest(request, 10)
    if responese.status.error_code == 0:
        print("successful update model")
    else:
        print("fail")
        print(responese.status.error_code)
        print(responese.status.error_message)


if __name__ == "__main__":
    tf.app.run()
