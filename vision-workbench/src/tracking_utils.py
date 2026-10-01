"""阶段 2：提取 ID、保存有限长度的中心点轨迹、画跟踪结果。"""

import cv2

from utils import calculate_center


def update_history(history, objects, frame_number, trail_length=30):
    """history 是字典：每个 ID 对应最近的 (帧号, x, y) 列表。"""
    for obj in objects:
        track_id = obj["track_id"]
        if track_id is None:
            continue  # 未分配 ID 的检测不建立轨迹
        if track_id not in history:
            history[track_id] = []
        x, y = obj["center"]
        history[track_id].append((frame_number, x, y))

    # 每次都清理旧点，避免摄像头运行很久后占用越来越多内存。
    for track_id in list(history):
        history[track_id] = [
            point for point in history[track_id]
            if point[0] > frame_number - trail_length
        ]
        if not history[track_id]:
            del history[track_id]


def draw_tracks(frame, result, history, frame_number):
    """返回画好的图像和可写入 JSON 的目标列表；没有目标时返回空列表。"""
    objects = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # 检测类别编号与跟踪 ID 是两回事；缺少 ID 时不能编造编号。
            track_id = int(box.id[0]) if box.id is not None else None
            class_id = int(box.cls[0])
            objects.append({
                "track_id": track_id,
                "class_id": class_id,
                "class": result.names[class_id],
                "confidence": round(float(box.conf[0]), 4),
                "bounding_box": [x1, y1, x2, y2],
                "center": list(calculate_center(x1, y1, x2, y2)),
            })

    update_history(history, objects, frame_number)
    for obj in objects:
        track_id = obj["track_id"]
        # 相同 ID 使用相同颜色，便于观察身份是否保持。
        color = (160, 160, 160) if track_id is None else (
            80 + track_id * 37 % 176,
            80 + track_id * 67 % 176,
            80 + track_id * 97 % 176,
        )
        points = history.get(track_id, [])
        for previous, current in zip(points, points[1:]):
            if current[0] == previous[0] + 1:
                # 中间丢失过目标时不连接，避免把未观测的运动画成事实。
                cv2.line(frame, previous[1:], current[1:], color, 2)

        x1, y1, x2, y2 = obj["bounding_box"]
        cx, cy = obj["center"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        identity = f"ID {track_id}" if track_id is not None else "ID pending"
        labels = [f"{identity} {obj['class']} {obj['confidence']:.2f}", f"center=({cx},{cy})"]
        # 两行文字加黑底；靠近右边缘时向左挪动，减少文字被截断。
        for row, label in enumerate(labels):
            (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = max(0, min(x1, frame.shape[1] - width - 4))
            text_y = min(frame.shape[0] - baseline - 2, max(50, y1) + row * 20)
            cv2.rectangle(frame, (text_x, text_y - height - 3),
                          (text_x + width + 3, text_y + baseline + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return frame, objects
