export CUDA_VISIBLE_DEVICES=4
export FLAGS_use_system_allocator=1
python engine.py --api_config_file=tester/api_config/a.txt --accuracy=True >> tester/api_config/test_log/log.log 2>&1