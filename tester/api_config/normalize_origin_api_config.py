import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]

def get_notsupport_config():
    not_support_files = [
        "tester/api_config/api_config_merged_not_support_amp.txt",
        "tester/api_config/api_config_merged_not_support_arange.txt",
        "tester/api_config/api_config_merged_not_support_empty.txt",
        "tester/api_config/api_config_merged_not_support_flatten.txt",
        "tester/api_config/api_config_merged_not_support_getset_item.txt",
        "tester/api_config/api_config_merged_not_support_reshape.txt",
        "tester/api_config/api_config_merged_not_support_slice.txt",
        "tester/api_config/api_config_merged_not_support_topk.txt",
        "tester/api_config/api_config_merged_not_support_zeros.txt",
        "tester/api_config/api_config_merged_not_support.txt"
    ]
    configs = set()

    for flie in not_support_files:
        with open(DIR_PATH+"/"+flie, "r") as f:
            origin_configs = f.readlines()
            f.close()

        for config in origin_configs:
            configs.add(config)
    return configs

logs = [
"/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_1.txt",
"/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_2.txt",
"/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_3.txt",
"/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_4.txt"
]

configs = set()

for log in logs:
    with open(log, "r") as f:
        origin_configs = f.readlines()
        f.close()

    for config in origin_configs:
        configs.add(config)

configs = configs - get_notsupport_config()

with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_all.txt", "w") as f:
    for config in sorted(configs):
        f.write(config)


# with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_all3.txt", "w") as f:
#     with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_not_support_concat_amp.txt", "w") as gs:
#         for config in sorted(configs):
#             if "concat" in config:
#                 lala = 0
#                 if "float32" in config:
#                     lala = lala + 1
#                 if "float64" in config:
#                     lala = lala + 1
#                 if "float16" in config:
#                     lala = lala + 1
#                 if "bfloat16" in config:
#                     lala = lala + 1
#                 if lala > 1:
#                     gs.write(config)
#                 else:
#                     f.write(config)
#             else:
#                 f.write(config)
#         f.close()
#         gs.close()
