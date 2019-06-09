import os
import platform
from base64 import b64encode
from uuid import uuid4

import pymysql
import tornado.ioloop
import tornado.web

import tensorflow as tf

from serving.serving_utils import build_from_config
from web.handler import SegmentHandler, PageHandler, UserHandler, AnalysisHandler, FeedbackHandler

flags = tf.flags

FLAGS = flags.FLAGS

## Dataset Input parameters
flags.DEFINE_string("config_file", "config/web/config.json",
                    "the domain use for part label training")


def main(_):
    system = platform.system()
    serving_path = f"{os.getcwd()}/serving"
    # os.system("docker kill cws_container")
    server_url, model_options, model_setting, cxt_feature_extractors, dict_feature_extractors, _ = build_from_config(
        serving_path)
    # if system == "Windows":
    #     cmd = f'start /b docker run -t --name="cws_container" --rm -p 8500:8500 -p 8501:8501 ' \
    #         f'--mount type=bind,source="{serving_path}/models",target=/serving/models ' \
    #         f'--mount type=bind,source="{serving_path}/models.config",target=/serving/models.config -t ' \
    #         'tensorflow/serving --model_config_file=/serving/models.config'
    # elif system == "Linux":
    #     cmd = f'docker run -t --name="cws_container" --rm -p 8500:8500 -p 8501:8501 ' \
    #         f'--mount type=bind,source="{serving_path}/models",target=/serving/models ' \
    #         f'--mount type=bind,source="{serving_path}/models.config",target=/serving/models.config -t ' \
    #         'tensorflow/serving --model_config_file=/serving/models.config &'
    # else:
    #     raise SystemError("only support Windows and Linux")
    # os.system(cmd)
    db = pymysql.connect("localhost", "root", "123456", "ocws")
    db.autocommit(True)
    app = tornado.web.Application([
        (r"/([^/.]*)", PageHandler),
        (r"/api/v0/user", UserHandler, dict(db=db)),
        (r"/api/v0/feedback", FeedbackHandler, dict(db=db)),
        (r"/api/v0/analysis", AnalysisHandler, dict(db=db)),
        (r"/api/v0/segment", SegmentHandler, dict(server_url=server_url,
                                                  model_options=model_options,
                                                  model_setting=model_setting,
                                                  cxt_feature_extractors=cxt_feature_extractors,
                                                  dict_feature_extractors=dict_feature_extractors,
                                                  db=db)),
        (r'/(favicon.ico)', tornado.web.StaticFileHandler, dict(path="web/")),
        (r"/css/(.*)", tornado.web.StaticFileHandler, dict(path="web/css")),
        (r"/js/(.*)", tornado.web.StaticFileHandler, dict(path="web/js")),
    ], cookie_secret=b64encode(uuid4().bytes + uuid4().bytes))
    app.listen(8888, xheaders=True)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    tf.app.run()
