# Python、目标检测与目标跟踪学习笔记

这份笔记从初学者角度解释本项目中的真实代码。建议先打开 `src/detect_image.py`，一边看代码一边阅读。

## Web 图片检测的阅读顺序

新增的本地页面可以按 `app.py` → `src/workbench/image_service.py` → `src/workbench/model_manager.py` → `src/utils.py` 阅读。`app.py` 中 `gr.Image` 是上传/展示组件；点击按钮后，`start.click(...)` 会调用 `run_image_detection(...)`。这个函数把上传图片的临时路径和两个滑块的数值传给 `detect_image(...)`，再把返回值放到页面组件中。

`image_service.py` 中的 `cv2.imread(...)` 将图片读成像素数组。`model.predict(image, conf=confidence, iou=iou, device="cpu")` 做一次 CPU 推理；`confidence` 过滤低置信度框，`iou` 调整重叠框的抑制程度。`perf_counter()` 前后相减得到本次推理秒数，乘 1000 才是毫秒。`Counter(...)` 统计每类目标有几个。最后 `cv2.imwrite(...)` 保存画框图，`json.dumps(...)` 保存便于分析的文本数据。

OpenCV 的彩色数组使用 BGR 顺序，而 Gradio 展示数组使用 RGB，所以返回页面前执行 `cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)`。如果忘了这一步，图片可能出现红蓝颜色互换。

## 1. `import` 是什么

`import` 表示“把其他模块中已经写好的功能引入当前文件”。例如：

```python
import cv2
from ultralytics import YOLO
from utils import draw_detections
```

- `import cv2`：引入整个 OpenCV 模块，使用功能时写 `cv2.imread(...)`。
- `from ultralytics import YOLO`：只从 `ultralytics` 包中引入名为 `YOLO` 的类。
- `from utils import draw_detections`：从本项目的 `src/utils.py` 引入画检测结果的函数。

这样可以复用成熟库和其他文件的代码，不必把所有功能写在一个文件里。

## 2. `cv2` 是什么

`cv2` 是 OpenCV 的 Python 模块名。虽然安装包叫 `opencv-python`，代码中要写 `import cv2`。

本项目中的常见用法：

```python
image = cv2.imread(str(source_path))       # 读取图片
cv2.rectangle(...)                        # 画矩形框
cv2.circle(...)                           # 画中心点
cv2.putText(...)                          # 写文字
cv2.imwrite(str(output_path), image)       # 保存图片
```

视频和摄像头使用 `cv2.VideoCapture(...)` 逐帧读取，输出视频使用 `cv2.VideoWriter(...)`。

## 3. YOLO 是什么

YOLO 是一类实时目标检测方法，名字来自 “You Only Look Once”。在本项目中，它接收一张图片，然后回答：

- 画面中检测到了什么类别，例如 `person`、`bus`、`car`；
- 模型对这个判断有多大把握；
- 目标在图片中的矩形位置。

代码先创建模型对象：

```python
model = YOLO(args.model)
```

再把图片交给模型：

```python
results = model.predict(image, conf=args.conf, device="cpu", verbose=False)
```

`device="cpu"` 表示只使用 CPU，不要求 CUDA。

## 4. 模型是什么

可以把模型理解为“已经从大量示例中学到规律的一组参数和计算方法”。`models/yolo11n.pt` 是预训练模型的权重文件。它已经在 COCO 数据集上学习过常见物体，本项目直接用它做判断，不进行训练。

`yolo11n` 中的 `n` 表示 nano。它的体积和计算量较小，更适合本项目当前的 CPU 环境。

## 5. inference / 推理是什么

训练是让模型从数据中学习；推理是使用训练好的模型对新输入做预测。本项目这一行就是推理：

```python
results = model.predict(image, conf=args.conf, device="cpu", verbose=False)
```

输入是图片，输出 `results` 中保存模型找到的目标。本阶段只做推理，不训练模型。

## 6. frame 是什么

frame 是“帧”。视频不是一张连续活动的图片，而是许多静态图片快速连续播放。每一张静态图片就是一帧。

在 `detect_video.py` 中：

```python
success, frame = capture.read()
```

- `success` 表示这一帧是否读取成功；
- `frame` 是当前读取到的图像数组，可以送入 YOLO，也可以用 OpenCV 画框。

摄像头检测也是不断获取新的一帧，所以代码结构与视频检测相似。

## 7. bounding box 是什么

Bounding box 是包围目标的矩形，中文常叫边界框或检测框。模型可能在一张图中返回多个框，每个框对应一个检测目标。

`utils.py` 使用下面这行取得坐标：

```python
x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
```

然后用 OpenCV 画出矩形：

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
```

## 8. confidence 是什么

Confidence 是置信度，表示模型对某个检测结果的把握程度。本项目读取方法是：

```python
confidence = float(box.conf[0])
```

它通常是 0 到 1 之间的小数。`0.90` 表示把握较高，`0.26` 表示刚刚超过默认阈值 `0.25`。置信度不是“绝对正确的概率”，实际结果仍可能误检或漏检。

## 9. class 是什么

Class 是类别。YOLO 不只画出位置，还会给每个目标一个类别编号：

```python
class_id = int(box.cls[0])
class_name = result.names[class_id]
```

第一行得到数字编号，第二行把编号转换为可读名称，例如 `person`。Python 自己也有关键字 `class`，所以变量名使用 `class_id` 和 `class_name`，避免冲突。

## 10. `x1 y1 x2 y2` 是什么

图像使用像素坐标：左上角是 `(0, 0)`，向右是 x 增大，向下是 y 增大。

```text
(x1, y1) ┌────────────┐
         │   目标     │
         │            │
         └────────────┘ (x2, y2)
```

- `x1, y1`：检测框左上角；
- `x2, y2`：检测框右下角。

## 11. 中心坐标怎么算

矩形中心等于两侧坐标的平均值：

```python
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2
```

`//` 是整数除法。像素位置用整数表示，所以即使平均值是 `20.5`，这里也会得到 `20`。随后用红色圆点标出中心：

```python
cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
```

注意：这只是图片平面中的二维像素坐标，不是以米或毫米表示的现实空间位置。

## 12. `for` 循环在检测代码中做什么

一张图可能有很多目标。`for` 循环让同一段代码依次处理每个目标：

```python
for box in result.boxes:
    # 读取当前目标的位置、类别和置信度
    # 给当前目标画框和中心点
```

如果检测到了 5 个目标，循环体就执行 5 次。`box` 每次代表其中一个目标。

视频处理外层使用 `while True` 循环，因为要不断读取下一帧；每一帧内部又用 `for` 处理这一帧中的多个目标。

## 13. function / 函数是什么

函数是“有名字、可以重复调用的一段代码”。例如 `utils.py` 中：

```python
def calculate_center(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y
```

- `def` 表示开始定义函数；
- `calculate_center` 是函数名；
- 括号中的四个名字是输入参数；
- `return` 把结果交回调用位置。

调用方法是：

```python
center_x, center_y = calculate_center(x1, y1, x2, y2)
```

使用函数可以让主流程更清楚，也可以让图片、视频和摄像头程序共用同一套中心点算法。

## 建议的阅读顺序

1. 先读 `detect_image.py` 的 `main()`，理解一张图片从读取到保存的完整流程。
2. 再读 `utils.py` 的 `calculate_center()` 和 `draw_detections()`。
3. 运行图片检测并对照终端输出和结果图片。
4. 再读 `detect_video.py`，重点看 `while` 如何逐帧处理。
5. 最后读 `detect_camera.py`，比较它与视频检测的共同点和区别。

## 阶段 2：检测和跟踪的区别

检测回答“这一帧有哪些物体、在哪里”。跟踪还需要回答“下一帧里的这个物体，是否与上一帧中的某个物体是同一个”。检测框列表的第 1 个元素不一定始终是同一个人，因此不能用 `for` 循环中的序号当身份。

`track_video.py` 负责打开视频，`tracking.py` 的 `while True` 每次处理一帧：

```python
result = model.track(
    frame, persist=True, tracker=str(TRACKER_PATH), device="cpu",
    conf=conf, imgsz=imgsz, verbose=False,
)[0]
```

模型创建在循环外：`model = YOLO(str(model_path))`。`persist=True` 告诉它当前帧是同一视频的下一帧，应该保留前面的跟踪状态。如果每帧重新创建模型，ID 就难以连续。

## ByteTrack、关联与 ID

ByteTrack 是跟踪算法。它用运动预测和检测框重叠程度来尝试匹配前后帧的目标；高分框先关联，较低分框还能帮助恢复已有轨迹。因此第二阶段的检测阈值默认是 0.10，创建新 ID 的阈值则在 YAML 中设为 0.25。

例如两个人的 `class_id` 都可以是 0（person），但 `track_id` 分别为 3 和 7。类别编号来自模型，跟踪 ID 来自跟踪器。它不是一个人的永久身份，重启程序后编号会重新开始。

`tracking_utils.py` 中的真实代码：

```python
track_id = int(box.id[0]) if box.id is not None else None
```

`None` 表示暂时没有这个值。画面可能检测到物体，但尚未建立有效轨迹；代码会显示 `ID pending` 并在 JSON 中记为 `null`，不把检测序号冒充跟踪 ID。

## 字典、列表与轨迹

`history = {}` 是空字典。它用跟踪 ID 作为键，最近的位置列表作为值，例如：

```python
history = {3: [(10, 100, 200), (11, 103, 200)]}
```

这表示 ID 3 在第 10 帧的中心是 `(100, 200)`，第 11 帧是 `(103, 200)`。`history[track_id].append(...)` 向列表加入新观测。程序只保留最近 30 帧，避免长时间运行时内存不断增长。

`zip(points, points[1:])` 依次取相邻两个点。只有帧号相邻时，才用 `cv2.line` 连接。如果中间有空帧，不把缺失位置画成已知轨迹。

## 遮挡、丢失与 ID 切换

目标被遮挡时可能没有检测框，跟踪器会暂时保留内部状态。`track_buffer: 30` 控制丢失状态的保留更新次数，约 30 次更新不等于任何设备上都固定 1 秒。重新出现且关联成功时可以沿用 ID；离开过久、突然跳跃或两个物体交叉时，可能获得新 ID 或错误换 ID。

本阶段不使用额外的外观重识别模型。轨迹线来自检测框中心，不是物体真实重心，也没有转换成米、毫米或机械臂坐标。

## JSONL 与逐帧保存

`tracking.py` 的 `json.dumps(record)` 把 Python 字典转换为 JSON 文本。每处理完一帧，写入一行；一行的 `objects` 可能含多个目标，也可能是空列表。

```python
record_file.write(json.dumps(record, ensure_ascii=False) + "\n")
```

`ensure_ascii=False` 允许直接保存中文；`\n` 表示换行。查看文件时可以逐行调用 `json.loads(line)`，不要直接对整个 JSONL 文件使用一次 `json.load()`。

## `try/finally` 为什么重要

打开视频和摄像头会占用资源。`finally` 中的 `capture.release()`、`writer.release()` 和 `record_file.close()`，确保正常结束、按 q、Ctrl+C 或发生常见错误后都释放资源。MP4 需要正确关闭才能写完文件结构。

## 第二阶段今天先学懂的三个点

1. 类别 ID 与跟踪 ID 的区别，为什么要保留跨帧状态。
2. 字典如何按 ID 存储有限长度的中心点历史。
3. 空检测、ID 未确认和目标丢失如何体现在程序与 JSONL 中。

建议阅读顺序：`track_video.py` → `tracking.py` 的循环 → `tracking_utils.py`。
