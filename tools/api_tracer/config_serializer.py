import time
from collections import deque
from threading import Event, Thread
from types import EllipsisType
from typing import Any, Dict, List

import yaml
from framework_dialect import FrameworkDialect


class ConfigSerializer:
    """将API调用信息序列化并写入"""

    _serialize_handlers = {}

    def __init__(self, dialect: FrameworkDialect, output_path: str):
        self.dialect = dialect
        self.output_path = output_path
        self.file_handler_yaml = None
        self.file_handler_txt = None
        self.buffer: List[Dict] = []
        self.buffer_limit = 20000

        # asyncio
        self.log_queue = deque()
        self._stop_event = Event()
        self.writer_thread = Thread(target=self._writer_loop)
        self.total_calls_processed = 0

        self._serialize_handlers = {
            type(None): lambda x: x,
            bool: lambda x: x,
            int: lambda x: x,
            float: lambda x: x,
            str: lambda x: x,
            list: self._serialize_list,
            tuple: self._serialize_tuple,
            set: self._serialize_set,
            dict: self._serialize_dict,
            type: self._serialize_type,
            slice: self._serialize_slice,
            EllipsisType: self._serialize_ellipsis,
        }

    def open(self):
        self.file_handler_yaml = open(
            self.output_path + "/api_trace.yaml", "w", encoding="utf-8"
        )
        self.file_handler_txt = open(
            self.output_path + "/api_trace.txt", "w", encoding="utf-8"
        )
        self.writer_thread.start()

    def close(self):
        self._stop_event.set()
        self.writer_thread.join()
        self._flush_buffer()

        if self.file_handler_yaml:
            self.file_handler_yaml.close()
            self.file_handler_yaml = None
        if self.file_handler_txt:
            self.file_handler_txt.close()
            self.file_handler_txt = None

        print(f"[ConfigSerializer] Files closed, final save to {self.output_path}")

    def _writer_loop(self):
        """线程的工作循环，从队列获取数据并处理"""
        while not self._stop_event.is_set() or len(self.log_queue) > 0:
            try:
                call_record = self.log_queue.popleft()
                self.buffer.append(call_record)

                if len(self.buffer) >= self.buffer_limit:
                    self._flush_buffer()
            except IndexError:
                time.sleep(0.01)
            except Exception as e:
                print(f"[ConfigSerializer] Error in writer thread: {e}")

    def _flush_buffer(self):
        """将buffer内容写入文件并清空buffer"""
        if not self.buffer:
            return

        self.total_calls_processed += len(self.buffer)

        if self.file_handler_yaml:
            try:
                yaml.dump(
                    self.buffer,
                    self.file_handler_yaml,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                )
                self.file_handler_yaml.flush()
            except Exception as e:
                print(f"[ConfigSerializer] Error writing YAML file: {e}")

        if self.file_handler_txt:
            try:
                for call_record in self.buffer:
                    txt_line = self._format_txt_line(
                        call_record["api"], call_record["args"], call_record["kwargs"]
                    )
                    self.file_handler_txt.write(txt_line + "\n")
                self.file_handler_txt.flush()
            except Exception as e:
                print(f"[ConfigSerializer] Error writing TXT file: {e}")

        print(
            f"[ConfigSerializer] Flushed {len(self.buffer)} calls, total processed: {self.total_calls_processed}"
        )
        self.buffer.clear()

    def dump_call(self, api_name: str, args: tuple, kwargs: dict, output: Any = None):
        """记录一次API调用"""
        try:
            call_record = {
                "api": api_name,
                "args": [self._serialize_item(arg) for arg in args],
                "kwargs": {
                    key: self._serialize_item(value) for key, value in kwargs.items()
                },
                # "output_summary": self._serialize_item(output)
            }
            self.log_queue.append(call_record)
        except Exception as e:
            print(f"[ConfigSerializer] Error serializing call for '{api_name}': {e}")

    def _serialize_list(self, item: list) -> Dict:
        return {
            "type": "list",
            "value": [self._serialize_item(sub_item) for sub_item in item],
        }

    def _serialize_tuple(self, item: tuple) -> Dict:
        return {
            "type": "tuple",
            "value": [self._serialize_item(sub_item) for sub_item in item],
        }

    def _serialize_set(self, item: set) -> Dict:
        return {
            "type": "set",
            "value": [self._serialize_item(sub_item) for sub_item in sorted(item)],
        }

    def _serialize_dict(self, item: dict) -> Dict:
        return {
            "type": "dict",
            "value": {str(k): self._serialize_item(v) for k, v in item.items()},
        }

    def _serialize_type(self, item: type) -> Dict:
        return {"type": "type", "value": f"{item.__module__}.{item.__name__}"}

    def _serialize_slice(self, item: slice) -> Dict:
        return {
            "type": "slice",
            "value": {"start": item.start, "stop": item.stop, "step": item.step},
        }

    def _serialize_ellipsis(self, item: Any) -> Dict:
        return {"type": "ellipsis", "value": "..."}

    def _serialize_item(self, item: Any) -> Any:
        """递归序列化对象"""
        handler = self._serialize_handlers.get(type(item))
        if handler:
            return handler(item)

        special_serialization = self.dialect.serialize_special_type(item)
        if special_serialization is not None:
            return special_serialization

        try:
            return f"<Unserializable: {type(item).__name__}>"
        except Exception:
            return "<Unserializable: unknown type>"

    def _format_txt_line(self, api_name: str, args: list, kwargs: dict) -> str:
        """格式化API调用为最通用的TXT配置"""

        def format_arg(arg: Any) -> str:
            if arg is None or isinstance(arg, (bool, int, float)):
                return str(arg)
            if isinstance(arg, str):
                return f'"{arg}"'

            if isinstance(arg, dict) and "type" in arg:
                type_handlers = {
                    "list": lambda x: f"list[{', '.join(format_arg(item) for item in x['value'])}]",
                    "tuple": lambda x: f"tuple({', '.join(format_arg(item) for item in x['value'])})",
                    "set": lambda x: f"set({', '.join(format_arg(item) for item in x['value'])})",
                    "dict": lambda x: f"dict({', '.join(f'{k}={format_arg(v)}' for k, v in x['value'].items())})",
                    "type": lambda x: x["value"],
                    "slice": lambda x: f"slice({x['value']['start']}, {x['value']['stop']}, {x['value']['step']})",
                    "ellipsis": lambda x: "ellipsis(...)",
                }
                handler = type_handlers.get(arg["type"])
                if handler:
                    return handler(arg)

            special_format = self.dialect.format_special_type(arg)
            if special_format is not None:
                return special_format

            return str(arg)

        args_str = ", ".join(format_arg(arg) for arg in args)
        kwargs_str = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs.items())
        return f"{api_name}({args_str + (', ' + kwargs_str if kwargs_str else '')})"

    def get_apis_and_configs(self):
        api_apis = set()
        api_configs = set()
        api_counts = {}
        line_count = 0
        with open(self.output_path + "/api_trace.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    api_configs.add(line)
                    api_api = line.split("(", 1)[0]
                    api_apis.add(api_api)
                    api_counts[api_api] = api_counts.get(api_api, 0) + 1
                    line_count += 1
        print(f"[ConfigSerializer] Read {line_count} traces from api_trace.txt")

        with open(self.output_path + "/api_apis.txt", "w", encoding="utf-8") as f:
            for api in sorted(api_apis):
                f.write(api + "\n")
        print(f"[ConfigSerializer] Write {len(api_apis)} apis to api_apis.txt")

        with open(self.output_path + "/api_configs.txt", "w", encoding="utf-8") as f:
            for config in sorted(api_configs):
                f.write(config + "\n")
        print(f"[ConfigSerializer] Write {len(api_configs)} configs to api_configs.txt")

        total_calls = sum(api_counts.values())
        api_percentages = {
            api: (count / total_calls) * 100 for api, count in api_counts.items()
        }
        sorted_api_counts = sorted(api_counts.items(), key=lambda x: x[1], reverse=True)

        with open(self.output_path + "/api_statistics.txt", "w", encoding="utf-8") as f:
            f.write(f"Total APIs: {len(api_apis)}\n")
            f.write(f"Total API calls: {total_calls}\n\n")
            for api, count in sorted_api_counts:
                f.write(f"{api}: {count} calls ({api_percentages[api]:.2f}%)\n")
            print(f"[ConfigSerializer] Write detailed statistics to api_statistics.txt")
