"""图片检测的业务逻辑：读图、CPU 推理、画框、保存结果。"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2

from utils import PROJECT_ROOT, draw_detections
from workbench.model_manager import get_model


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "images"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def detect_image(image_path: str, model_name: str, confidence: float, iou: float) -> dict:
    """返回可用于网页显示的数据；检测结果同时写入项目 outputs/images。"""
    if not image_path:
        raise ValueError("请先上传一张 JPG、JPEG 或 PNG 图片。")

    source_path = Path(image_path)
    if source_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("只支持 JPG、JPEG、PNG 图片。")
    if not source_path.is_file():
        raise FileNotFoundError("上传的图片文件不存在，请重新上传。")
    if not 0.1 <= confidence <= 1.0 or not 0.1 <= iou <= 1.0:
        raise ValueError("Confidence 和 IoU 阈值都必须在 0.1 到 1.0 之间。")

    # OpenCV 读进来的颜色顺序是 BGR，画框函数也使用 BGR。
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError("图片无法读取，可能已经损坏，请换一张图片。")

    model, model_path = get_model(model_name)

    # 明确指定 device="cpu"，避免依赖 NVIDIA CUDA。
    start = perf_counter()
    try:
        results = model.predict(image, conf=confidence, iou=iou, device="cpu", verbose=False)
    except Exception as error:
        raise RuntimeError("模型推理失败，请检查图片或模型文件。") from error
    inference_ms = (perf_counter() - start) * 1000

    annotated_bgr, detections = draw_detections(image.copy(), results[0])
    class_counts = dict(sorted(Counter(item["class"] for item in detections).items()))

    # 时间戳加随机短码，避免两次检测使用相同文件名。
    result_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_output = OUTPUT_DIR / f"{result_name}.jpg"
    json_output = OUTPUT_DIR / f"{result_name}.json"
    if not cv2.imwrite(str(image_output), annotated_bgr):
        raise OSError(f"结果图片保存失败：{image_output}")

    # JSON 保留每个目标的位置、类别、置信度，便于论文实验记录。
    record = {
        "filename": source_path.name,
        "model": model_path.name,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "inference_time_ms": round(inference_ms, 2),
        "total_objects": len(detections),
        "class_counts": class_counts,
        "detections": [
            {
                "class": item["class"],
                "confidence": round(item["confidence"], 4),
                "x1": item["bounding_box"][0],
                "y1": item["bounding_box"][1],
                "x2": item["bounding_box"][2],
                "y2": item["bounding_box"][3],
                "center_x": item["center"][0],
                "center_y": item["center"][1],
                "inference_time_ms": round(inference_ms, 2),
            }
            for item in detections
        ],
    }
    try:
        json_output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        image_output.unlink(missing_ok=True)
        raise

    # Gradio 期望 RGB 图片；输出到磁盘的 OpenCV 图片仍是 BGR。
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return {
        "image": annotated_rgb,
        "record": record,
        "image_path": str(image_output),
        "json_path": str(json_output),
    }
