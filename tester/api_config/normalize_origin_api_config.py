
with open("/host_home/wanghuan29/APItest3/PaddleAPITest/PaddleNLP_api/unittest_api_config.txt", "r") as f:
    origin_configs = f.readlines()
    f.close()


configs = set()
for config in origin_configs:
    configs.add(config)


with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_PaddleNLP_unittest.txt", "w") as f:
    for config in sorted(configs):
        f.write(config)
    f.close()
