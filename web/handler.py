from collections import defaultdict
from datetime import datetime, timedelta

import grpc
import jieba
import numpy as np
import tensorflow as tf
import tornado.web
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

from process import data_processor, feature_builder, dataset
from process.feature_builder import DimInfo
from serving.serving_utils import LazyConstructor

import hashlib
import math
import urllib


class FeedbackHandler(tornado.web.RequestHandler):
    def initialize(self, db):
        self.db = db

    async def post(self):
        text = self.get_argument("text", default=None)
        rate = self.get_argument("rate", default=None)
        model = self.get_argument("model", default=None)
        standard = self.get_argument("standard", default=None)

        quoted_name = self.get_cookie("username")
        if quoted_name is not None:
            username = urllib.parse.unquote(quoted_name)
        else:
            username = None
        group = self.get_cookie("group")
        token = self.get_cookie("token")
        if not UserHandler.check(self.db, username, group, token,
                                 lambda conn: conn.finish({'message': 'user error', 'state': "fail"})):
            group = None
            username = None

        if text is None or rate is None or model is None:
            self.finish({'message': 'None Argument',
                         'state': "fail"})

        self.insert_feedback(self.db, username, text, rate, model, standard)
        self.finish({'message': 'ok',
                     'state': "success"})

    @staticmethod
    def insert_feedback(db, user_name, text, rate, model, standard):
        cursor = db.cursor()
        if user_name is None:
            d_u_id = 0
        else:
            sql = "SELECT U_ID FROM user_table WHERE U_NAME=%s"
            cursor.execute(sql, user_name)
            results = cursor.fetchall()
            if len(results) <= 0:
                return False
            d_u_id = results[0][0]

        sql = f"INSERT INTO feedback_table(U_ID, TEXT, MODEL, STD, RATE) VALUES " \
            f"(%(d_u_id)s, %(text)s, %(model)s, %(std)s, %(rate)s)"
        cursor.execute(sql, {"d_u_id": d_u_id, "text": text, "model": model, "std": standard, "rate": rate})
        # db.commit()


class AnalysisHandler(tornado.web.RequestHandler):
    def initialize(self, db):
        self.db = db

    async def get(self):
        query = self.get_argument("query", default=None)
        cursor = self.db.cursor()
        if query == "model":
            # bar labels & data
            sql = "SELECT STR_DATA, COUNT(*) FROM data_table WHERE DATA_TYPE=\"MODEL_SELECT\" GROUP BY STR_DATA"
            cursor.execute(sql)
            results = cursor.fetchall()
            labels = []
            data = []
            for result in results:
                labels.append(result[0])
                data.append(result[1])
            self.finish({'message': 'ok',
                         'state': "success",
                         'labels': labels,
                         'data': data})
            return
        elif query == "std":
            # bar labels & data
            sql = "SELECT STR_DATA, COUNT(*) FROM data_table WHERE DATA_TYPE=\"STD_SELECT\" GROUP BY STR_DATA"
            cursor.execute(sql)
            results = cursor.fetchall()
            labels = []
            data = []
            for result in results:
                labels.append(result[0])
                data.append(result[1])
            self.finish({'message': 'ok',
                         'state': "success",
                         'labels': labels,
                         'data': data})
            return
        elif query == "input":
            # scatter: [{"x": 1, "y": 2}, {"x": 1, "y": 2}]
            sql = "SELECT INT_DATA FROM data_table WHERE DATA_TYPE=\"TEXT_CHAR_LEN\" ORDER BY D_ID"
            cursor.execute(sql)
            char_len_results = cursor.fetchall()
            sql = "SELECT INT_DATA FROM data_table WHERE DATA_TYPE=\"TEXT_LINE_LEN\" ORDER BY D_ID"
            cursor.execute(sql)
            line_len_results = cursor.fetchall()
            len_map = defaultdict(int)
            data = []
            for cl, ll in zip(char_len_results, line_len_results):
                len_map[cl[0], ll[0]] += 1
            maxv = max(len_map.values())
            minv = min(len_map.values())
            log_xn = math.log(maxv) - math.log(minv)
            xn = 50 - 5
            for key, value in len_map.items():
                data.append({"x": key[0], "y": key[1], "r": xn * (math.log(value) - math.log(minv))/log_xn + 5, "v": value})
            self.finish({'message': 'ok',
                         'state': "success",
                         'points': data})
            return
        elif query == "runTime":
            # time line: [{"t": 1, "y": 2}, {"t": 1, "y": 2}]
            sql = "SELECT DATE(LOG_DATETIME), count(LOG_DATETIME) FROM data_table WHERE DATA_TYPE=\"MODEL_SELECT\" " \
                  "AND DATE_SUB(CURDATE(), INTERVAL 30 DAY) < date(LOG_DATETIME) " \
                  "GROUP BY DATE(LOG_DATETIME) ORDER BY DATE(LOG_DATETIME)"
            cursor.execute(sql)
            results = cursor.fetchall()
            data = []
            labels = []
            for i in reversed(range(30)):
                labels.append((datetime.now() - timedelta(days=i)).date().isoformat())
            for result in results:
                data.append({"x": result[0].isoformat(), "y": result[1]})
            self.finish({'message': 'ok',
                         'state': "success",
                         'labels': labels,
                         'points': data})
            return
        elif query == "feedback":
            offset = self.get_argument("offset", default=0)
            size = self.get_argument("size", default=5)
            # time line: [{"t": 1, "y": 2}, {"t": 1, "y": 2}]
            sql = "SELECT MODEL, STD, TEXT, RATE FROM feedback_table LIMIT %s, %s"
            cursor.execute(sql, [int(offset), int(size)])
            results = cursor.fetchall()
            data = []
            for result in results:
                data.append({"model": result[0], "standard": result[1], "text": result[2], "rate": result[3]})
            self.finish({'message': 'ok',
                         'state': "success",
                         'data': data})
            return

        self.finish({'message': 'unknown request',
                     'state': "fail"})

    @staticmethod
    def insert_data(db, user_name, type_name, data):
        cursor = db.cursor()
        if user_name is None:
            d_u_id = 0
        else:
            sql = "SELECT U_ID FROM user_table WHERE U_NAME=%s"
            cursor.execute(sql, user_name)
            results = cursor.fetchall()
            if len(results) <= 0:
                return False
            d_u_id = results[0][0]
        sql_type_name = None
        ##
        #  TEXT: STR_DATA
        #  TEXT_CHAR_LEN: INT_DATA
        #  TEXT_LINE_LEN: INT_DATA
        #  MODEL_SELECT: STR_DATA
        #  STD_SELECT: STR_DATA
        #  RUN_TIME: TIME_DATA
        #  FEEDBACK_TEXT: STR_DATA
        ##

        if isinstance(data, dict):
            data_type_set = {"INT_DATA", "STR_DATA", "FLOAT_DATA", "TIME_DATA", "DATETIME_DATA"}
            type_list = []
            data_list = []
            for k, v in data.items():
                if k not in data_type_set:
                    return False
                else:
                    type_list.append(k)
                    data_list.append(v)
        else:
            if type_name == "TEXT_CHAR_LEN":
                sql_type_name = "INT_DATA"
            elif type_name == "TEXT_LINE_LEN":
                sql_type_name = "INT_DATA"
            elif type_name == "RUN_TIME":
                sql_type_name = "TIME_DATA"
            else:
                sql_type_name = "STR_DATA"
            type_list = [sql_type_name]
            data_list = [data]

        sql = f"INSERT INTO data_table(U_ID, DATA_TYPE, {', '.join(type_list)}) VALUES (%s, %s, {', '.join(['%s'] * len(data_list))})"
        cursor.execute(sql, [d_u_id, type_name] + data_list)
        # db.commit()


class SegmentHandler(tornado.web.RequestHandler):
    def initialize(self, server_url, model_options, model_setting, cxt_feature_extractors, dict_feature_extractors, db):
        self.db = db
        self.server_url = server_url
        self.model_setting = model_setting
        self.model_options = model_options
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

    async def get(self):
        query = self.get_argument("query", default=None)
        if query is None or query == "options":
            self.finish({'message': 'ok',
                         'state': "success",
                         'models': self.model_options})
        else:
            self.finish({'message': 'unknown request',
                         'state': "fail"})

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

        quoted_name = self.get_cookie("username")
        if quoted_name is not None:
            username = urllib.parse.unquote(quoted_name)
        else:
            username = None
        group = self.get_cookie("group")
        token = self.get_cookie("token")
        if not UserHandler.check(self.db, username, group, token,
                                 lambda conn: conn.finish({'message': 'user error', 'state': "fail"})):
            group = None
            username = None

        input_text = self.get_argument("input", default="")
        solution = self.get_argument("solution", default=None)
        if solution is None:
            self.finish(
                {'message': 'please select model',
                 'state': "fail"
                 })
            return
        text_len = len(input_text)
        line_len = len(input_text.split("\n"))
        if group is None:
            if line_len > 5 or text_len > 100:
                self.finish(
                    {'message': f'lines: {line_len} > 5 or lines * max_line: {text_len} > 100',
                     'state': "fail"
                     })
                return
        elif group == "0" or group == "2":
            if line_len > 100 or text_len > 2000:
                self.finish(
                    {'message': f'lines: {line_len} > 100 or lines * max_line: {text_len} > 2000',
                     'state': "fail"
                     })
                return
        else:
            if line_len > 50 or text_len > 1000:
                self.finish(
                    {'message': f'lines: {line_len} > 50 or lines * max_line: {text_len} > 1000',
                     'state': "fail"
                     })
                return
        AnalysisHandler.insert_data(self.db, username, "TEXT", {"STR_DATA": input_text})
        AnalysisHandler.insert_data(self.db, username, "TEXT_CHAR_LEN", {"INT_DATA": text_len})
        AnalysisHandler.insert_data(self.db, username, "TEXT_LINE_LEN", {"INT_DATA": line_len})
        model = self.model_setting[solution]["model"]
        AnalysisHandler.insert_data(self.db, username, "MODEL_SELECT", {"STR_DATA": model})
        if model == "jieba":
            self.finish({'message': 'ok',
                         'state': "success",
                         'results': [jieba.lcut(input_text)],
                         'model': model,
                         'standard': None
                         })
            return
        standard = self.model_setting[solution]["standard"]
        AnalysisHandler.insert_data(self.db, username, "STD_SELECT", {"STR_DATA": standard})
        examples, features, batch_size, max_len = self.convert(input_text, solution)

        start_time = datetime.now()
        response = await self.inference(features, model, standard)
        now_time = datetime.now()
        spend_time = now_time - start_time
        print(f"{spend_time.total_seconds():.2f} sec")
        AnalysisHandler.insert_data(self.db, username, "RUN_TIME", {"TIME_DATA": spend_time})
        results = np.array(response.outputs["result"].int_val).reshape((batch_size, -1))
        examples = examples
        self.finish({'message': 'ok',
                     'state': "success",
                     'results': list(map(batch_seg, zip(examples, results))),
                     'model': model,
                     'standard': standard
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


class PageHandler(tornado.web.RequestHandler):
    def get(self, page):
        with open('./web/main.html', 'rb') as f:
            self.write(f.read())
            self.finish()


class UserHandler(tornado.web.RedirectHandler):
    def initialize(self, db):
        self.db = db

    async def get(self):
        quoted_name = self.get_cookie("username")
        if quoted_name is not None:
            username = urllib.parse.unquote(quoted_name)
        else:
            username = None
        group = self.get_cookie("group")
        token = self.get_cookie("token")
        if UserHandler.check(self.db, username, group, token, lambda conn: conn.finish({'message': 'fail',
                                                                                        'state': "error"})):
            self.finish({'message': 'check',
                         'state': "success"
                         })
            return
        else:
            self.clear_cookie("username")
            self.clear_cookie("group")
            self.clear_cookie("token")
            self.finish({'message': 'invalid',
                         'state': "success"
                         })
            return

    @staticmethod
    def get_token(u_id, username):
        token_md5 = hashlib.md5(f"{u_id}&&{username}&&{datetime.now().timestamp()}".encode('utf-8'))
        return token_md5.hexdigest()

    @staticmethod
    def check(db, username, group, token, callback=None):
        if username is None or group is None or token is None:
            return False
        cursor = db.cursor()
        check_log_sql = "SELECT U_NAME, U_GROUP, unix_timestamp(LOG_TIME), TOKEN, unix_timestamp(NOW()), user_table.U_ID " \
                        "FROM user_table INNER JOIN log_table ON user_table.U_ID=log_table.U_ID where U_NAME =%s"
        try:
            cursor.execute(check_log_sql, username)
            results = cursor.fetchall()
            if len(results) <= 0:
                return
            result = results[0]
            d_name = result[0]
            d_group = result[1]
            d_log = result[2]
            d_token = result[3]
            d_now = result[4]
            d_t_id = result[5]
            if username == d_name and group == str(d_group) and token == d_token and d_now - d_log < 7 * 24 * 60 * 60:
                return True
            else:
                # sql = "DELETE FROM log_table WHERE U_ID = %s"
                # cursor.execute(sql, d_t_id)
                # db.commit()
                return False
        except IOError:
            callback()
            print("Error: unable to fetch data")

    async def post(self):
        query = self.get_argument("query", default=None)
        cursor = self.db.cursor()
        if query == "is_used":
            username = self.get_argument("username", default=None)
            if username is None:
                self.finish({'message': 'username is null',
                             'state': "fail"
                             })
                return
            name_count_sql = "SELECT COUNT(*) AS nums FROM `user_table` WHERE `U_NAME`=%s"
            cursor.execute(name_count_sql, username)
            result = cursor.fetchall()[0]
            if result[0] > 0:
                self.finish({'message': 'used',
                             'state': "success"
                             })
                return
            else:
                self.finish({'message': 'unused',
                             'state': "success"
                             })
                return
        elif query == "login":
            username = self.get_argument("username", default=None)
            password = self.get_argument("password", default=None)
            if username is None or password is None:
                self.finish({'message': 'username or password is null',
                             'state': "fail"
                             })
                return
            try:
                login_sql = "SELECT U_PASSWORD, U_ID, U_GROUP FROM `user_table` where `U_NAME` =%s"
                cursor.execute(login_sql, username)
                results = cursor.fetchall()
                if len(results) <= 0:
                    self.finish({'message': 'no that user',
                                 'state': "fail"
                                 })
                    return
                result = results[0]
                d_password = result[0]
                d_u_id = result[1]
                d_group = result[2]
                if password == d_password:
                    token = self.get_token(d_u_id, username)
                    sql = "SELECT COUNT(*) AS nums FROM log_table WHERE U_ID=%s"
                    cursor.execute(sql, d_u_id)
                    nums = cursor.fetchall()[0][0]
                    if nums > 0:
                        sql = "UPDATE log_table SET TOKEN=%(token)s, LOG_TIME=NOW() WHERE U_ID=%(u_id)s"
                        cursor.execute(sql, {"token": token, "u_id": d_u_id})
                    else:
                        sql = "INSERT INTO log_table(TOKEN, U_ID) VALUES (%(token)s, %(u_id)s)"
                        cursor.execute(sql, {"token": token, "u_id": d_u_id})
                    # self.db.commit()
                    self.set_cookie("username", urllib.parse.quote(username), expires_days=7)
                    self.set_cookie("group", str(d_group), expires_days=7)
                    self.set_cookie("token", token, expires_days=7)
                    self.finish({'message': 'ok',
                                 'state': "success"
                                 })
                    return
                else:
                    self.finish({'message': 'password or username not right',
                                 'state': "fail"
                                 })
                    return
            except IOError:
                print("Error: unable to fetch data")
                self.finish({'message': 'unable to fetch data',
                             'state': "fail"
                             })
                return
        elif query == "logout":
            quoted_name = self.get_cookie("username")
            if quoted_name is not None:
                username = urllib.parse.unquote(quoted_name)
            else:
                username = None
            self.clear_cookie("username")
            self.clear_cookie("group")
            self.clear_cookie("token")
            if username is None:
                self.finish({'message': 'ok',
                             'state': "success"
                             })
                return
            sql = "SELECT U_ID FROM `user_table` where `U_NAME` =%s"
            cursor.execute(sql, username)
            results = cursor.fetchall()
            if len(results) <= 0:
                self.finish({'message': 'no that user',
                             'state': "fail"
                             })
                return
            result = results[0]
            d_t_id = result[0]
            sql = "DELETE FROM log_table WHERE U_ID = %s"
            cursor.execute(sql, d_t_id)
            # self.db.commit()
            self.finish({'message': 'ok',
                         'state': "success"
                         })
            return
        elif query == "register":
            username = self.get_argument("username", default=None)
            password = self.get_argument("password", default=None)
            email = self.get_argument("email", default=None)
            # print(username)
            # print(password)
            # print(email)
            # print("register")
            if username is None or password is None or email is None:
                self.finish({'message': 'username or password or email is null',
                             'state': "fail"
                             })
                return
            try:
                sql = "SELECT COUNT(*) FROM `user_table` where `U_NAME` =%s"
                cursor.execute(sql, username)
                result = cursor.fetchall()[0]
                nums = result[0]
                if nums > 0:
                    self.finish({'message': 'name was used',
                                 'state': "fail"
                                 })
                    return
                reg_sql = "INSERT INTO user_table (U_NAME, U_PASSWORD, U_EMAIL, U_GROUP) " \
                          "VALUES (%(username)s, %(password)s, %(email)s, %(group)s)"
                # 0 -> admin, 1 -> normal, 2 -> vip
                cursor.execute(reg_sql, {"username": username, "password": password, "email": email, "group": 1})
                # self.db.commit()
                login_sql = "SELECT U_ID FROM `user_table` where `U_NAME` =%s"
                cursor.execute(login_sql, username)
                results = cursor.fetchall()
                if len(results) <= 0:
                    self.finish({'message': 'some thing wrong',
                                 'state': "fail"
                                 })
                    return
                result = results[0]
                d_u_id = result[0]
                token = self.get_token(d_u_id,username)
                sql = "INSERT INTO log_table(TOKEN, U_ID) VALUES (%(token)s, %(u_id)s)"
                cursor.execute(sql, {"token": token, "u_id": d_u_id})
                # self.db.commit()
                self.set_cookie("username", urllib.parse.quote(username), expires_days=7)
                self.set_cookie("group", str(1), expires_days=7)
                self.set_cookie("token", token, expires_days=7)
                self.finish({'message': 'ok',
                             'state': "success"
                             })
                return
            except IOError:
                print("Error: unable to fetch data")
                self.finish({'message': 'unable to fetch data',
                             'state': "fail"
                             })
                return
