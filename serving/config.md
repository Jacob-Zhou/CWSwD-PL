├── server_url                       #TensorFlow Serving服务器地址及端口
├── config_template_path             #配置生成模板的地址
├── context_feature_extractors[]     #上下文特征提取器配置
│   ├── name
│   └── config
├── dictionary_feature_extractors[]  #词典特征提取器配置
│   ├── name
│   └── config
├── models                           #模型配置
│   ├── name                         #模型名称
│   ├── base_path                    #模型基础位置
│   └── standards[]                  #分词标准
│       ├── name                     #标准名称
│       ├── versions                 #模型版本
│       └── path                     #模型版本
└── solutions[]                      #解决方案, 包含了文本预处理配置与模型配置
    ├── name                         #解决方案名称
    ├── alias                        #解决方案代号, 用于在前端模型选择时显示
    ├── isleaf
    ├── model                        #解决方案中所使用的模型, 当isleaf为true时必须提供
    ├── standard                     #解决方案的分词标准, 当isleaf为true时必须提供
    ├── context_feature_extractor    #解决方案中所用的上下文特征提取器, 当isleaf为true时必须提供
    ├── dictionary_feature_extractor #解决方案中所用的词典特征提取器, 当isleaf为true时必须提供
    └── multitag
    └── children[]                   #解决方案的子方案, 当isleaf为false时必须提供




    