
with open("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config.txt.1", "r") as f:
    origin_configs = f.readlines()
    f.close()


configs = set()
for config in origin_configs:
    configs.add(config)


with open("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config_PaddleX.txt", "w") as f:
    for config in sorted(configs):
        f.write(config)
    f.close()
