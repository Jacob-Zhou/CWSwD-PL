import json
import os

from process import context_feature_extractor, dictionary_feature_extractor
from utils import Config


def build_from_config(serving_path):
    serving_config_path = os.path.join(serving_path, "serving.json")
    with open(serving_config_path, "r", encoding="utf-8") as reader:
        text = reader.read()
    serving_json = json.loads(text)
    server_url = serving_json["server_url"]
    cxt_feature_extractors_config = serving_json["context_feature_extractors"]
    dict_feature_extractors_config = serving_json["dictionary_feature_extractors"]
    models = serving_json["models"]
    config_template_path = os.path.join(serving_path, serving_json["config_template_path"])
    with open(config_template_path, "r", encoding="utf-8") as reader:
        text = reader.read()
    config_template_json = json.loads(text)
    model_options = {"children": []}
    model_configs = []
    model_setting = {}
    for model in models:
        for standard in model["standards"]:
            versions = ", ".join(
                map(lambda x: config_template_json["versions"].format(**{"version": x}), standard["versions"]))
            model_configs.append({"model": model["name"],
                                  "standard": standard["name"],
                                  "model_base_path": model["base_path"],
                                  "standard_path": standard["path"],
                                  "versions": versions})

    solutions = serving_json["solutions"]
    tree_walker(solutions, "", model_options, model_setting)
    model_options = model_options["children"]

    model_configs_inside = ", ".join(
        map(lambda x: config_template_json["config"].replace("'", '"').format(**x), model_configs))
    model_configs_text = config_template_json["model_config_list"].format(**{"configs": model_configs_inside})
    model_config_path = os.path.join(serving_path, "models.config")
    with open(model_config_path, "w", encoding="utf-8") as cfg_out:
        cfg_out.write(model_configs_text)

    cxt_feature_extractors = LazyLoader(lambda cfg: context_feature_extractor.TrTeWinBiContextFeatureExtractor(
        vocab_file=cfg.vocab_file, bigram_file=cfg.bigram_file,
        type_file=cfg.type_file, bigram_type_file=cfg.bigram_type_file, window_size=cfg.window_size))
    for cxt_feature_extractor in cxt_feature_extractors_config:
        cxt_feature_extractor_config = Config.from_json_file(
            os.path.join(serving_path, cxt_feature_extractor["config"]))
        cxt_feature_extractors[cxt_feature_extractor["name"]] = cxt_feature_extractor_config

    dict_feature_extractors = LazyLoader(lambda cfg: dictionary_feature_extractor.DefaultDictionaryFeatureExtractor(
        cfg.dict_file,
        min_word_len=cfg.min_word_len,
        max_word_len=cfg.max_word_len))
    for dict_feature_extractor in dict_feature_extractors_config:
        dict_feature_extractor_config = Config.from_json_file(
            os.path.join(serving_path, dict_feature_extractor["config"]))
        dict_feature_extractors[dict_feature_extractor["name"]] = dict_feature_extractor_config

    return server_url, model_options, model_setting, cxt_feature_extractors, dict_feature_extractors, model_configs_text


def tree_walker(nodes, prefix, model_option, model_setting):
    for node in nodes:
        solution_name = f"{prefix}{'' if prefix else '/'}{node['name']}"
        if node["isleaf"]:
            model_setting[solution_name] = {
                "model": node["model"],
                "standard": node["standard"],
                "context_feature_extractor": node["context_feature_extractor"],
                "dictionary_feature_extractor": node["dictionary_feature_extractor"],
                "multitag": node['multitag']
            }
            model_option["children"].append({"value": solution_name, "label": node["alias"]})
        else:
            new_option = {"value": node['name'], "label": node["alias"], "children": []}
            tree_walker(node["children"], solution_name, new_option, model_setting)
            model_option["children"].append(new_option)


class LazyConstructor:
    def __init__(self, build_func):
        self.items = {}
        self.build_func = build_func

    def __getitem__(self, item):
        if item in self.items:
            return self.items[item]
        else:
            self.items[item] = self.build_func(item)
            return self.items[item]


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
