"""调用电脑摄像头，实时执行 YOLO11n 目标检测。"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT_FOR_CONFIG = Path(__file__).resolve().parent.parent
ULTRALYTICS_CONFIG_PARENT = PROJECT_ROOT_FOR_CONFIG / "configs" / "ultralytics"
ULTRALYTICS_CONFIG_PARENT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_PARENT))

from ultralytics import YOLO

from utils import DEFAULT_MODEL_PATH, PROJECT_ROOT, draw_detections, ensure_parent_directory, put_fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLO11n 实时检测摄像头画面")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO 模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="最低置信度，默认 0.25")
    parser.add_argument("--save", action="store_true", help="同时保存检测后的视频")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 尝试打开摄像头。临时隐藏 OpenCV 冗长的底层日志，改用下面的中文提示。
    previous_log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    capture = cv2.VideoCapture(args.camera)
    cv2.utils.logging.setLogLevel(previous_log_level)

    # 没有摄像头时给出说明并正常退出。
    if not capture.isOpened():
        capture.release()
        print(f"错误：无法打开编号为 {args.camera} 的摄像头。")
        print("请检查摄像头连接、系统权限，或尝试参数 --camera 1。")
        return 1

    writer = None
    if args.save:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_fps = capture.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:
            camera_fps = 30.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "results" / "videos" / f"camera_{timestamp}.mp4"
        ensure_parent_directory(output_path)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            camera_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            print(f"错误：无法创建摄像头录像：{output_path}")
            return 1
        print(f"录像将保存到：{output_path}")

    model = YOLO(args.model)
    smoothed_fps = 0.0

    print("摄像头检测已启动，按 q 退出。")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("错误：摄像头已打开，但读取画面失败。")
                break

            start_time = time.perf_counter()
            results = model.predict(frame, conf=args.conf, device="cpu", verbose=False)
            annotated_frame, _ = draw_detections(frame.copy(), results[0])

            elapsed = time.perf_counter() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            smoothed_fps = current_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * current_fps
            put_fps(annotated_frame, smoothed_fps)

            if writer is not None:
                writer.write(annotated_frame)

            cv2.imshow("YOLO11n Camera Detection - press q to quit", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print("摄像头检测已结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
