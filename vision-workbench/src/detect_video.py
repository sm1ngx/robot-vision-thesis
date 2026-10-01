"""逐帧读取本地视频，执行 YOLO11n 目标检测并保存结果。"""

import argparse
import os
import time
from pathlib import Path

import cv2

PROJECT_ROOT_FOR_CONFIG = Path(__file__).resolve().parent.parent
ULTRALYTICS_CONFIG_PARENT = PROJECT_ROOT_FOR_CONFIG / "configs" / "ultralytics"
ULTRALYTICS_CONFIG_PARENT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_PARENT))

from ultralytics import YOLO

from utils import DEFAULT_MODEL_PATH, PROJECT_ROOT, draw_detections, ensure_parent_directory, put_fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLO11n 检测本地视频")
    parser.add_argument("--source", required=True, help="输入视频路径")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "videos" / "detected_video.mp4"),
        help="结果视频路径",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO 模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="最低置信度，默认 0.25")
    parser.add_argument("--no-display", action="store_true", help="不弹出显示窗口，适合无桌面环境")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not source_path.is_file():
        print(f"错误：找不到输入视频：{source_path}")
        return 1

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        print(f"错误：OpenCV 无法打开该视频：{source_path}")
        return 1

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    ensure_parent_directory(output_path)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        print(f"错误：无法创建输出视频：{output_path}")
        return 1

    model = YOLO(args.model)
    frame_number = 0
    smoothed_fps = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            start_time = time.perf_counter()

            # 每次循环处理视频的一帧。device="cpu" 保证不依赖 CUDA。
            results = model.predict(frame, conf=args.conf, device="cpu", verbose=False)
            annotated_frame, _ = draw_detections(frame.copy(), results[0])

            elapsed = time.perf_counter() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            # 简单平滑 FPS，避免数字变化太剧烈。
            smoothed_fps = current_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * current_fps
            put_fps(annotated_frame, smoothed_fps)

            writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow("YOLO11n Video Detection - press q to quit", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("用户按下 q，提前结束视频检测。")
                    break
    finally:
        # 即使用户提前退出，也正确释放视频和窗口资源。
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"处理完成，共处理 {frame_number} 帧。")
    print(f"结果视频已保存到：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
