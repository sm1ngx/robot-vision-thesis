"""集中管理工作台使用的 YOLO 模型。当前阶段只开放默认模型。"""

import os
from functools import lru_cache
from pathlib import Path

from utils import DEFAULT_MODEL_PATH, PROJECT_ROOT


MODEL_CHOICES = {"YOLO11n（CPU，默认）": DEFAULT_MODEL_PATH}


@lru_cache(maxsize=2)
def load_model(model_path: str):
    """首次使用时加载模型，之后复用，避免每次点击都重新加载权重。"""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{path}。请检查 models/ 目录。")

    # Ultralytics 的设置只写入本项目，且必须在导入 YOLO 之前设置。
    config_dir = PROJECT_ROOT / "configs" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    from ultralytics import YOLO

    try:
        return YOLO(str(path))
    except Exception as error:
        raise RuntimeError(f"模型加载失败，请确认权重文件有效：{path}") from error


def get_model(model_name: str):
    """按界面上的名称选择模型；后续自定义权重可在这里扩展。"""
    if model_name not in MODEL_CHOICES:
        raise ValueError("请选择列表中的模型。")
    path = MODEL_CHOICES[model_name]
    return load_model(str(path)), path
