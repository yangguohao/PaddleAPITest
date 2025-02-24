
logs = [
"/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_all8.txt",
"/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_not_support_getset_item.txt"
]

configs = set()

for log in logs:
    with open(log, "r") as f:
        origin_configs = f.readlines()
        f.close()

    for config in origin_configs:
        configs.add(config)


with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_all9.txt", "w") as f:
    with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_getset_item.txt", "w") as gss:
        with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_not_support_getset_item2.txt", "w") as gs:
            for config in sorted(configs):
                if ("__getitem__" in config or "__setitem__" in config):
                    if config.count("Tensor(") > 1:
                        gs.write(config)
                    else:
                        gss.write(config)
                else:
                    f.write(config)
            f.close()
            gs.close()


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
