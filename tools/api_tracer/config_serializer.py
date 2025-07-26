from typing import Any, Dict, List

import yaml

from framework_dialect import FrameworkDialect


class ConfigSerializer:
    """将API调用信息序列化并写入"""

    def __init__(self, dialect: FrameworkDialect, output_path: str):
        self.dialect = dialect
        self.output_path = output_path
        self.file_handler = None
        self.buffer: List[Dict] = []

    def open(self):
        self.file_handler = open(self.output_path, "w", encoding="utf-8")

    def close(self):
        if self.file_handler:
            try:
                yaml.dump(
                    self.buffer,
                    self.file_handler,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                )
            except Exception as e:
                print(f"[ConfigSerializer] Error writing YAML file: {e}")
            finally:
                self.file_handler.close()
                self.file_handler = None
        print(
            f"[ConfigSerializer] API trace with {len(self.buffer)} calls saved to {self.output_path}"
        )

    def dump_call(self, api_name: str, args: tuple, kwargs: dict, output: Any):
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
            self.buffer.append(call_record)
        except Exception as e:
            print(f"[ConfigSerializer] Error serializing call for '{api_name}': {e}")

    def _serialize_item(self, item: Any) -> Any:
        """递归序列化对象"""
        # 1. 使用方言进行序列化
        special_serialization = self.dialect.serialize_special_type(item)
        if special_serialization is not None:
            return special_serialization

        # 2. 处理Python基本类型和集合
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        if isinstance(item, list):
            return [self._serialize_item(sub_item) for sub_item in item]
        if isinstance(item, tuple):
            return {
                "_type": "tuple",
                "value": [self._serialize_item(sub_item) for sub_item in item],
            }
        if isinstance(item, set):
            return {
                "_type": "set",
                "value": [
                    self._serialize_item(sub_item) for sub_item in sorted(list(item))
                ],
            }
        if isinstance(item, dict):
            return {str(k): self._serialize_item(v) for k, v in item.items()}
        if isinstance(item, type):
            return {"_type": "type", "value": f"{item.__module__}.{item.__name__}"}

        # 3. 无法处理时，返回描述性字符串
        try:
            return f"<Unserializable: {type(item).__name__}>"
        except Exception:
            return "<Unserializable: unknown type>"
