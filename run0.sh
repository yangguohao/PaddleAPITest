export CUDA_VISIBLE_DEVICES=0
export FLAGS_use_system_allocator=1
for i in {1..10000}; do python engine.py --api_config_file=/home/PaddleAPITest/tester/api_config/api_config_merged_amp.txt --test_amp=True  >> tester/api_config/test_log/log.log 2>&1; done
