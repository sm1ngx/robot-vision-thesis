"""视频和摄像头共用的跟踪循环。先读 track_video.py，再读本文件。"""

import json
import math
import os
import time
from pathlib import Path

import cv2

from tracking_utils import draw_tracks
from utils import DEFAULT_MODEL_PATH, PROJECT_ROOT, put_fps

TRACKER_PATH = PROJECT_ROOT / "configs" / "bytetrack.yaml"


def run_tracking(capture, output_path, *, show=True, camera=False,
                 conf=0.10, imgsz=640, max_frames=0, model_path=DEFAULT_MODEL_PATH):
    """读取连续帧，用同一个模型跟踪；本函数负责释放 capture 和输出资源。"""
    writer = None
    record_file = None
    frame_number = 0
    saved_frames = 0
    output_path = Path(output_path).expanduser().resolve()
    record_path = output_path.with_suffix(".jsonl")
    window_name = "YOLO11n + ByteTrack - press q to quit"

    try:
        # 所有检查放在 try 内，任何失败都会执行 finally 释放摄像头/文件。
        if not 0 < conf <= 1 or imgsz < 32 or max_frames < 0:
            raise ValueError("conf 必须在 (0, 1]，imgsz 至少 32，max-frames 不能为负数。")
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("输出视频请使用 .mp4 扩展名。")
        if output_path.exists() or record_path.exists():
            raise ValueError("输出视频或 JSONL 已存在，请用 --output 指定新名称。")
        if show and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            raise ValueError("没有桌面显示环境，请添加 --no-display。")
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise ValueError(f"找不到本地模型：{model_path}，请按 README 准备模型。")

        # 配置留在项目内，运行时禁止隐式安装依赖或联网下载。
        config_dir = PROJECT_ROOT / "configs" / "ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
        from ultralytics import YOLO

        model = YOLO(str(model_path))  # 只创建一次，同一段视频共用跟踪状态
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
            print("无法取得输入 FPS，输出视频暂按 30 FPS 编码。")

        history = {}  # ID -> 最近 30 帧的中心点；每次运行从空字典开始
        smoothed_fps = 0.0
        start_time = time.perf_counter()
        print("CPU 跟踪已启动。窗口按 q，终端按 Ctrl+C 退出。")

        while True:
            success, frame = capture.read()
            if not success:
                if frame_number == 0 or camera:
                    raise ValueError("无法读取画面，请检查输入文件或摄像头连接。")
                break  # 本地视频读到末尾

            frame_number += 1
            # 视频时间来自帧号；摄像头时间来自启动后的实际经过时间。
            timestamp = (time.perf_counter() - start_time if camera
                         else (frame_number - 1) / source_fps)
            frame_start = time.perf_counter()
            result = model.track(
                frame, persist=True, tracker=str(TRACKER_PATH), device="cpu",
                conf=conf, imgsz=imgsz, verbose=False,
            )[0]
            annotated, objects = draw_tracks(frame.copy(), result, history, frame_number)
            current_fps = 1.0 / max(time.perf_counter() - frame_start, 1e-6)
            smoothed_fps = (current_fps if frame_number == 1
                            else 0.9 * smoothed_fps + 0.1 * current_fps)
            put_fps(annotated, smoothed_fps)

            # 用实际首帧尺寸创建输出，避免摄像头报告的尺寸为 0。
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         source_fps, (width, height))
                if not writer.isOpened():
                    raise ValueError(f"不能创建视频：{output_path}")
                record_file = record_path.open("x", encoding="utf-8")

            writer.write(annotated)
            # 一行记录一帧；即使没有目标也记录 objects=[]，便于分析丢失情况。
            record = {"frame": frame_number, "time_seconds": round(timestamp, 4),
                      "processing_fps": round(smoothed_fps, 2), "objects": objects}
            record_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_frames += 1

            if show:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and frame_number >= max_frames:
                break
    except KeyboardInterrupt:
        print("收到 Ctrl+C，保存已经处理的部分并退出。")
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as error:
        print(f"跟踪失败：{error}")
        print("请检查输入、模型和项目依赖；已有部分输出可使用新文件名重新运行。")
        return 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if record_file is not None:
            record_file.close()
        if show:
            cv2.destroyAllWindows()

    print(f"结束，共保存 {saved_frames} 帧。")
    if saved_frames:
        print(f"视频：{output_path}\n逐帧记录：{record_path}")
    else:
        print("尚未保存有效帧。")
    return 0
