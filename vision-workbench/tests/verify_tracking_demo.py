"""检查真实推理的 demo 输出，而不是仅仅确认命令没有报错。"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="检查 tracking_demo 的完整跟踪输出")
    parser.add_argument("--video", required=True, help="跟踪后的 mp4，与同名 jsonl 放在一起")
    args = parser.parse_args()
    video = Path(args.video)
    with video.with_suffix(".jsonl").open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    assert len(records) == 60, "逐帧记录必须为 60 条"
    bus_observations = []
    for frame_number, record in enumerate(records, 1):
        assert record["frame"] == frame_number
        assert abs(record["time_seconds"] - (frame_number - 1) / 10) < 0.001
        ids = [obj["track_id"] for obj in record["objects"] if obj["track_id"] is not None]
        assert len(ids) == len(set(ids)), "同一帧不能有重复 ID"
        if frame_number <= 5 or 31 <= frame_number <= 34:
            assert record["objects"] == [], "空白帧不应有目标"
        for obj in record["objects"]:
            if obj["class"] == "bus" and obj["track_id"] is not None:
                bus_observations.append((frame_number, obj["track_id"], obj["center"]))

    assert len(bus_observations) >= 40, "公交车有效跟踪帧数过少"
    id_counts = Counter(item[1] for item in bus_observations)
    dominant_id, count = id_counts.most_common(1)[0]
    assert count / len(bus_observations) >= 0.9, "移动过程中公交车 ID 不够稳定"
    before = [item for item in bus_observations if item[0] <= 30]
    after = [item for item in bus_observations if item[0] >= 35]
    assert before and after and before[-1][1] == after[0][1], "短暂消失后 ID 未恢复"
    assert len({tuple(item[2]) for item in bus_observations}) > 10, "应实际观察到位置移动"

    capture = cv2.VideoCapture(str(video))
    decoded = 0
    try:
        assert capture.isOpened(), "输出视频无法打开"
        assert abs(capture.get(cv2.CAP_PROP_FPS) - 10) < 0.01
        while True:
            success, frame = capture.read()
            if not success:
                break
            assert frame.shape[:2] == (640, 480)
            decoded += 1
    finally:
        capture.release()
    assert decoded == 60, "必须实际解码全部 60 帧"
    print(f"验证通过：60 帧视频及 JSONL 完整；空白帧、时间和 ID 唯一性正常。")
    print(f"公交车跟踪 {len(bus_observations)} 帧，主 ID={dominant_id}，保持 {count} 帧。")
    print(f"消失前后 ID：{before[-1][1]} → {after[0][1]}；中心位置发生变化。")


if __name__ == "__main__":
    main()
