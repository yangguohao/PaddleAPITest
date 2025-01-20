
with open("origin_api_config.txt", "r") as f:
    origin_configs = f.readlines()
    f.close()


configs = set()
for config in origin_configs:
    configs.add(config)


with open("api_config.txt", "w") as f:
    for config in sorted(configs):
        f.write(config)
    f.close()
