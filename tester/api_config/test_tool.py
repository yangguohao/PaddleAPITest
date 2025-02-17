# logs = [
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_6.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_7.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_1.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_6.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_7.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4.log",]

# with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/oom.log", "w") as oom:
#     with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/oom_not.log", "w") as oom_not:
#         for log in logs:
#             with open(log, "r") as f:
#                 lines = f.readlines()
#                 is_log_str = False
#                 log_str = ""
#                 for line in lines:
#                     if "paddle error" in line or "cuda error" in line:
#                         is_log_str = True
#                         log_str = line
#                     elif "test begin" in line:
#                         is_log_str = False
#                         if "CUDAAllocator" in log_str:
#                             oom.write(log_str)
#                         else:
#                             oom_not.write(log_str)
#                     elif is_log_str:
#                         log_str += line

# with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/accuracy.log", "w") as accuracy:
#     for log in logs:
#         with open(log, "r") as f:
#             lines = f.readlines()
#             is_log_str = False
#             log_str = ""
#             for line in lines:
#                 if "accuracy error" in line:
#                     is_log_str = True
#                     log_str = line
#                 elif "test begin" in line:
#                     is_log_str = False
#                     if "Mismatched elements" in log_str:
#                         accuracy.write(log_str)
#                 elif is_log_str:
#                     log_str += line


logs = [
"/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/core_dump.log",]

with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/cinn_core_dump.log", "w") as get_log:
    with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/other.log", "w") as other:
        for log in logs:
            with open(log, "r") as f:
                lines = f.readlines()
                is_log_str = False
                log_str = ""
                for line in lines:
                    if "test begin" in line:
                        if "cinn_op." in log_str or "cinn::" in log_str:
                            get_log.write(log_str)
                        else:
                            other.write(log_str)
                        log_str = line
                    else:
                        log_str += line
