from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Dict, List

import yaml
from framework_dialect import FrameworkDialect


class ConfigSerializer:
    """将API调用信息序列化并写入"""

    def __init__(self, dialect: FrameworkDialect, output_path: str):
        self.dialect = dialect
        self.output_path = output_path
        self.file_handler_yaml = None
        self.file_handler_txt = None
        self.buffer: List[Dict] = []
        self.buffer_limit = 20000

        # asyncio
        self.log_queue = Queue()
        self._stop_event = Event()
        self.writer_thread = Thread(target=self._writer_loop)
        self.total_calls_processed = 0

    def open(self):
        self.file_handler_yaml = open(
            self.output_path + "/api_trace.yaml", "a", encoding="utf-8"
        )
        self.file_handler_txt = open(
            self.output_path + "/api_trace.txt", "a", encoding="utf-8"
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
        while not self._stop_event.is_set() or not self.log_queue.empty():
            try:
                api_name, args, kwargs, output = self.log_queue.get(timeout=0.1)

                try:
                    call_record = {
                        "api": api_name,
                        "args": [self._serialize_item(arg) for arg in args],
                        "kwargs": {
                            key: self._serialize_item(value)
                            for key, value in kwargs.items()
                        },
                    }
                    self.buffer.append(call_record)
                except Exception as e:
                    print(
                        f"[ConfigSerializer] Error serializing call for '{api_name}': {e}"
                    )

                if len(self.buffer) >= self.buffer_limit:
                    self._flush_buffer()

                self.log_queue.task_done()
            except Empty:
                pass
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

    def dump_call(self, api_name: str, args: tuple, kwargs: dict, output: Any):
        """记录一次API调用"""
        self.log_queue.put((api_name, args, kwargs, output))

    def _serialize_item(self, item: Any) -> Any:
        """递归序列化对象"""
        if item is None or isinstance(item, (bool, int, float, str)):
            return item

        special_serialization = self.dialect.serialize_special_type(item)
        if special_serialization is not None:
            return special_serialization

        if isinstance(item, list):
            return {
                "type": "list",
                "value": [self._serialize_item(sub_item) for sub_item in item],
            }
            # return [self._serialize_item(sub_item) for sub_item in item]
        if isinstance(item, tuple):
            return {
                "type": "tuple",
                "value": [self._serialize_item(sub_item) for sub_item in item],
            }
        if isinstance(item, set):
            return {
                "type": "set",
                "value": [
                    self._serialize_item(sub_item) for sub_item in sorted(list(item))
                ],
            }
        if isinstance(item, dict):
            return {
                "type": "dict",
                "value": {str(k): self._serialize_item(v) for k, v in item.items()},
            }
        if isinstance(item, type):
            return {"type": "type", "value": f"{item.__module__}.{item.__name__}"}
        if isinstance(item, slice):
            return {
                "type": "slice",
                "value": {"start": item.start, "stop": item.stop, "step": item.step},
            }
        if item is ...:
            return {"type": "ellipsis", "value": "..."}

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

            special_format = self.dialect.format_special_type(arg)
            if special_format is not None:
                return special_format

            if isinstance(arg, dict) and "type" in arg:
                if arg["type"] == "list":
                    return (
                        f"list[{', '.join(format_arg(item) for item in arg['value'])}]"
                    )
                if arg["type"] == "tuple":
                    return (
                        f"tuple({', '.join(format_arg(item) for item in arg['value'])})"
                    )
                if arg["type"] == "set":
                    return (
                        f"set({', '.join(format_arg(item) for item in arg['value'])})"
                    )
                if arg["type"] == "dict":
                    return f"dict({', '.join(f'{k}={format_arg(v)}' for k, v in arg['value'].items())})"
                if arg["type"] == "type":
                    return arg["value"]
                if arg["type"] == "slice":
                    return f"slice({arg['value']['start']}, {arg['value']['stop']}, {arg['value']['step']})"
                if arg["type"] == "ellipsis":
                    return "ellipsis(...)"
            return str(arg)

        args_str = ", ".join(format_arg(arg) for arg in args)
        kwargs_str = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs.items())
        all_args = args_str + (", " + kwargs_str if kwargs_str else "")
        return f"{api_name}({all_args})"
