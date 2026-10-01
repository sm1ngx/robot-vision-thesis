# 机器人视觉感知、跟踪与目标检测工作台

这是本科毕业设计《基于视觉语言模型的机器人目标理解与抓取系统设计与实现》的阶段 1、2 工程。OpenCV 读取画面，YOLO11n 在 CPU 上检测物体，ByteTrack 为连续帧中的目标分配跟踪 ID。

当前完成基础视觉感知与多目标跟踪。保留阶段 1 的检测入口；阶段 2 新增目标 ID、最近 30 帧的中心点轨迹和逐帧 JSONL 记录。尚未加入 ROS2、VLM、机械臂控制、空间定位或模型训练。

## 新增：本地 Web 工作台（第 1 阶段）

这次是在**现有项目内**新增图片检测网页，没有改动原有命令行入口。项目虚拟环境已安装 Gradio；重新配置环境时可在激活虚拟环境后运行 `python -m pip install -r requirements.txt`。需要 Python 3.10 和 `models/yolo11n.pt`；不需要 CUDA。

```bash
cd /home/sharpa/Documents/Codex/2026-10-02/wo/outputs/robot-vision-thesis/vision-workbench
source .venv/bin/activate
python app.py
```

浏览器打开 `http://127.0.0.1:7860`。上传 JPG、JPEG 或 PNG，调整 Confidence、IoU 阈值，点击“开始检测”。页面会显示原图、画框结果、目标总数、各类别数量、单次推理耗时和理论推理速度，并可下载画框 JPG 与 JSON；文件也保存在 `outputs/images/`。可用 `data/images/bus.jpg` 试运行。退出服务按终端 `Ctrl+C`；端口被占用可用 `python app.py --port 7861`。默认只监听本机，未开启公网分享。

注意：单张图片的“理论推理速度”仅由推理耗时换算，**不是摄像头实际 FPS**；首次点击还可能有模型预热开销。图片中心点仍是二维像素坐标，不能直接作为机械臂抓取坐标。

这次采用的结构（只创建已实现的模块，后续按阶段增加）：

```text
robot_vision_thesis/
├── app.py                         # Gradio 页面与启动入口
├── src/
│   ├── utils.py                   # 原有绘框和中心点工具，继续复用
│   ├── detect_*.py / track_*.py   # 原有命令行检测、跟踪入口，继续保留
│   └── workbench/
│       ├── model_manager.py       # 统一查找、缓存 YOLO 模型
│       └── image_service.py       # 图片读取、CPU 推理、统计与保存
├── models/yolo11n.pt              # 现有轻量模型
├── outputs/images/                # Web 图片结果与 JSON
├── tests/test_web_workbench.py    # 新增的输入检查与真实推理测试
└── docs/architecture.md           # 数据流和后续扩展说明
```

Web 图片数据流：`gr.Image 上传路径 → image_service.py 用 OpenCV 读取 → model_manager.py 取得 YOLO11n → YOLO 在 CPU 推理 → utils.py 画框与中心点 → 页面显示 + outputs/images/ 保存`。`app.py` 只组装界面和调用服务；以后 Web 摄像头、视频、模型对比和标注可各自加模块，不必把流程塞进同一个文件。

当前 Web 页面**仅支持图片检测**。电脑摄像头、上传视频、手机 RTSP/HTTP 流、自定义权重、多模型对比、拖拽标注与 YOLO 标签导出均尚未实现；页面中的“后续阶段”只是说明，不是假功能。已有命令行视频、摄像头、手机流和跟踪功能仍按下面的旧说明使用。

工作台测试：

```bash
python -m unittest discover -s tests -v
```

## 无机械臂的虚拟抓取演示

```bash
python src/simulate_tabletop.py
```

无需安装新仿真软件、摄像头或机械臂。程序会在 `results/videos/` 生成 90 帧、6 秒的 MP4 和同名 JPG 预览图。演示中的圆点目标、桌面尺寸、像素到米的比例、机械臂两段长度都是**预先设定的虚拟数值**。代码根据目标平面坐标计算两关节角，画出“靠近—夹持—移动—放下—返回”的过程。它不调用 YOLO、不读取真实相机、不控制实物，也不模拟重力、碰撞和三维夹爪，因此不等于真实抓取成功。说明见 [虚拟演示笔记](docs/simulation_notes.md)。

第三方软件、模型、示例素材和本项目代码的来源见 [来源与个人工作说明](docs/third_party_and_authorship.md)。

## 阶段 2：直接运行

### 手机摄像头作为视觉输入

手机和电脑连入同一局域网，在手机的视频流工具中启动推流，复制工具给出的 HTTP/MJPEG 或 RTSP 视频流地址。网页播放器地址不一定是流地址。[OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)可以读取 IP 视频流。以下 IP、端口和路径只是示意，以手机端显示的值为准：

```bash
python src/track_stream.py --url 'http://192.168.1.20:8080/video' --no-display --max-frames 120
python src/track_stream.py --url 'rtsp://192.168.1.20:8554/live' --no-display --max-frames 120
```

第一次建议使用 `--no-display --max-frames 120`，避免此前桌面窗口无响应干扰连接验证。成功后在 `results/videos/` 下得到 MP4 和同名 JSONL。去掉这两个参数可打开实时窗口，按 `q` 退出。若手机作为 USB 摄像头显示为 `/dev/videoN`，则使用 `python src/track_camera.py --camera N`。

带密码的地址可通过环境变量传入，减少密码出现在进程命令行参数中：

```bash
export ROBOT_VISION_STREAM_URL='rtsp://用户名:密码@手机IP:端口/实际路径'
python src/track_stream.py --no-display --max-frames 120
unset ROBOT_VISION_STREAM_URL
```

如果 CPU 推理跟不上手机推流，画面可能滞后。可先在手机端选约 640×480、10～15 FPS；此入口默认 `--imgsz 416`。手机画面只提供二维像素坐标；机械臂抓取还需要固定相机位置、标定、桌面目标位置估计和机械臂坐标转换。

在 Ubuntu 终端执行：

```bash
cd /home/sharpa/Documents/Codex/2026-10-02/wo/outputs/robot-vision-thesis/vision-workbench
source .venv/bin/activate
python src/track_video.py --source data/videos/tracking_demo.mp4
```

窗口按 `q` 退出。无显示环境使用 `--no-display`；CPU 较慢可尝试 `--imgsz 416`。每次默认生成带时间戳的 MP4 和同名 JSONL，终端会显示完整保存路径。

```bash
# 自己的视频
python src/track_video.py --source data/videos/my_video.mp4
# 摄像头，默认编号 0，自动保存视频和逐帧记录
python src/track_camera.py
# 无窗口运行，只处理前 60 帧
python src/track_video.py --source data/videos/tracking_demo.mp4 --no-display --max-frames 60
```

可使用 `--output results/videos/my_tracking.mp4` 指定新名称。如果 MP4 或同名 JSONL 已存在，程序会提示换名，保护已有文件。摄像头支持 `--camera 1`，终端可按 `Ctrl+C` 保存已处理部分并退出。真实摄像头已成功运行过，但长时间打开窗口时出现过无响应和进程终止；首次验证其他输入时建议先用 `--no-display --max-frames 120`。

两个新增入口共用 `src/tracking.py` 的循环：

```python
result = model.track(frame, persist=True, tracker=str(TRACKER_PATH),
                     device="cpu", conf=0.10, imgsz=640, verbose=False)[0]
```

`persist=True` 让同一段视频的连续帧保留跟踪状态。`configs/bytetrack.yaml` 控制关联规则；不需要下载新模型。现有虚拟环境已经安装阶段 2 唯一新增的依赖 `lap==0.5.12`。重新配置环境时照下文安装 `requirements.txt` 即可。

当前验收输出：`results/videos/tracking_demo_tracked.mp4` 和同名 `.jsonl`；截图在 `results/images/tracking_demo_preview.jpg`。详细验收记录见 `docs/stage2_verification.md`。

### 如何阅读逐帧记录

JSONL 是“每行一个 JSON 对象”，不是整份文件一个 JSON 数组。每行含 `frame`（从 1 开始）、`time_seconds`、`processing_fps` 和 `objects` 列表。目标字段包括 `track_id`、`class_id`、`class`、`confidence`、`bounding_box`、`center`。

没有目标时 `objects=[]`；检测到但未建立有效轨迹时 `track_id=null`，画面显示 `ID pending`。类别 ID 表示“是什么”，跟踪 ID 表示“这一段视频中的哪一个”。新的进程会重新编号。目标离开很久、严重遮挡或交叉后可能换 ID，不保证跨视频识别同一对象。

画面 FPS 是推理加绘图的平滑处理速度，不含解码、磁盘写入和窗口等待，也不是源视频 FPS。保存的 MP4 按源帧率编码。摄像头处理速度低于采集帧率时，保存视频的播放时长可能短于真实采集时长；JSONL 的摄像头时间戳记录实际经过时间。视频文件的时间戳使用 `(帧号-1)/源FPS`，适合恒定帧率视频，不保留可变帧率时间轴。输出仅包含画面，不保留音轨。

### 复现阶段 2 验证

```bash
python tests/make_tracking_demo.py
python src/track_video.py --source data/videos/tracking_demo.mp4 --output results/videos/my_verified.mp4 --no-display
python tests/verify_tracking_demo.py --video results/videos/my_verified.mp4
python -m unittest discover -s tests -v
```

测试素材由官方示例 [bus.jpg](https://ultralytics.com/images/bus.jpg) 平移生成，含空白帧和短暂消失；它是功能检查，不是真实运动基准。自有视频放入 `data/videos/` 后按相同命令运行，观察交叉、遮挡和快速移动时的表现。

## 当前系统流程

```text
图片 / 视频 / 摄像头
        ↓
      OpenCV
        ↓
   YOLO11n（CPU）
        ↓
目标类别、置信度、bounding box、中心坐标
        ↓
ByteTrack（视频/摄像头跟踪入口）：ID 与近期轨迹
        ↓
  画面显示与结果保存
```

最终目标是在后续阶段逐步形成：

```text
自然语言指令 → VLM / 任务理解 → 视觉目标检测 → 目标位置获取
→ 坐标转换 → ROS2 → 机械臂运动规划 → 目标抓取
```

## 项目目录

```text
robot_vision_thesis/
├── README.md                 # 项目总说明和运行教程
├── requirements.txt          # 阶段 1、2 的 Python 直接依赖
├── .gitignore                # 不提交虚拟环境、模型和运行结果
├── configs/bytetrack.yaml    # 跟踪阈值与丢失缓冲参数
├── data/
│   ├── images/               # 输入图片
│   └── videos/               # 输入视频
├── models/                   # YOLO11n 模型权重
├── results/
│   ├── images/               # 检测后的图片
│   └── videos/               # 检测/跟踪视频和 JSONL
├── src/
│   ├── detect_image.py       # 图片检测入口
│   ├── detect_video.py       # 视频检测入口
│   ├── detect_camera.py      # 摄像头检测入口
│   ├── track_video.py        # 阶段 2 视频跟踪入口
│   ├── track_camera.py       # 阶段 2 摄像头跟踪入口
│   ├── track_stream.py       # 手机网络视频流跟踪入口
│   ├── simulate_tabletop.py  # 二维机械臂虚拟抓取演示
│   ├── tracking.py           # CPU 跟踪、视频/JSONL 保存与资源释放
│   ├── tracking_utils.py     # 提取 ID、维护轨迹、画跟踪结果
│   └── utils.py              # 画框、中心点、FPS 等共用函数
├── docs/
│   ├── architecture.md       # 当前系统架构说明
│   ├── learning_notes.md     # 面向 Python 初学者的代码学习笔记
│   ├── stage2_verification.md # 阶段 2 实测记录与限制
│   └── thesis_notes.md       # 毕设阶段与路线记录
└── tests/
    ├── test_utils.py         # 中心坐标函数的基础测试
    ├── test_tracking.py      # 跟踪边界与输出保护测试
    ├── make_tracking_demo.py # 生成 60 帧移动测试视频
    └── verify_tracking_demo.py # 验证真实推理输出
```

阶段 2 暂用简单的 `tracking.py` 和 `tracking_utils.py`，后续变复杂时再整理为 `tracking/` 包。`depth/`、`vlm/`、`ros2/`、`robot_control/`、`grasp/` 和 `simulation/` 等模块留给后续阶段。

## 环境要求

- Ubuntu 22.04
- Python 3.10（本项目已在 Python 3.10.12 下验证）
- 普通 CPU；不要求 NVIDIA 显卡或 CUDA
- 摄像头仅在运行摄像头检测时需要

先确认 Python 版本：

```bash
python3 --version
```

若输出为 `Python 3.10.x`，即可继续。

## 创建虚拟环境和安装依赖

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 指定 PyTorch 官方 CPU 软件源，并固定使用带 `+cpu` 标记的版本，目的是避免引入 CUDA 运行库。所有包都安装在本项目的 `.venv/` 内，不会修改系统 Python。

以后每次打开新终端，都要先进入项目并激活环境：

```bash
cd robot_vision_thesis
source .venv/bin/activate
```

确认 PyTorch 使用 CPU：

```bash
python -c "import torch; print(torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
```

在本项目环境中，`CUDA 可用` 应显示 `False`。

## 模型

项目使用在 COCO 数据集上预训练的 `yolo11n.pt`。其中 `n` 表示 nano，是 YOLO11 系列中体积较小、适合 CPU 入门实验的版本。模型文件位于：

```text
models/yolo11n.pt
```

如果模型文件被删除，可在项目根目录重新下载：

```bash
cd models
../.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
cd ..
```

## 运行图片检测

把图片放入 `data/images/`，然后运行：

```bash
python src/detect_image.py --source data/images/bus.jpg
```

程序会在终端打印每个目标的类别、置信度、检测框和中心坐标，并把画好结果的图片保存为：

```text
results/images/detected_image.jpg
```

指定其他输出名或置信度阈值：

```bash
python src/detect_image.py \
  --source data/images/bus.jpg \
  --output results/images/my_result.jpg \
  --conf 0.30
```

## 运行视频检测

把视频放入 `data/videos/`，然后运行：

```bash
python src/detect_video.py --source data/videos/example.mp4
```

程序逐帧检测、显示检测框和 FPS。显示窗口中按 `q` 可以提前结束，已经处理的部分仍会写入：

```text
results/videos/detected_video.mp4
```

如果在无桌面窗口的终端中运行，可以关闭实时窗口，但仍保存结果：

```bash
python src/detect_video.py --source data/videos/example.mp4 --no-display
```

## 运行摄像头检测

```bash
python src/detect_camera.py
```

- 按 `q` 退出。
- 默认使用编号 `0` 的摄像头。
- 如果系统没有摄像头或没有权限，程序会打印明确错误并退出，不会因未处理异常而崩溃。
- 若有多个摄像头，可尝试 `python src/detect_camera.py --camera 1`。
- 如需保存摄像头检测视频，使用 `python src/detect_camera.py --save`。

## 每个 Python 文件负责什么

- `src/detect_image.py`：读取单张图片、调用 YOLO、画出结果并保存，是最适合第一个阅读的入口。
- `src/detect_video.py`：通过 `while` 循环逐帧读取视频，计算 FPS，并写出新视频。
- `src/detect_camera.py`：打开摄像头并持续读取实时画面；按 `q` 安全退出。
- `src/utils.py`：保存三个入口都会使用的简单函数，包括中心点计算、画框、打印结果与 FPS 显示。
- `src/track_video.py` / `src/track_camera.py`：读取参数并打开对应输入源。
- `src/track_stream.py`：打开手机网络视频流，复用已有跟踪循环。
- `src/simulate_tabletop.py`：使用虚拟平面坐标与两连杆逆运动学生成抓取流程动画。
- `src/tracking.py`：连续帧跟踪、保存视频和逐帧记录、保证退出时释放资源。
- `src/tracking_utils.py`：区分类别与 ID、绘制有限长度轨迹，自动清理过期历史。

## YOLO 和 OpenCV 的作用

YOLO 是目标检测模型。它接收一张图片，判断画面中有哪些已知类别的物体，并返回每个物体的类别、置信度和位置。当前使用的是已经训练好的 YOLO11n，只做推理，不训练新模型。

OpenCV 是图像和视频处理库。本项目用它读取、显示和保存图片/视频，打开摄像头，并在画面上绘制矩形、文字、中心点和 FPS。可以简单理解为：YOLO 负责“看懂目标”，OpenCV 负责“取得画面并把结果画出来”。

## 三个重要检测概念

### Bounding box

Bounding box（边界框或检测框）是包围目标的矩形。本项目使用四个像素坐标表示：

```text
(x1, y1) = 左上角
(x2, y2) = 右下角
```

图片左上角是坐标原点 `(0, 0)`，`x` 向右增大，`y` 向下增大。

### 置信度

置信度（confidence）是模型对这次判断的把握程度，通常在 0 到 1 之间。例如 `0.87` 可理解为模型对该检测结果有较高把握，但它不是绝对正确率。参数 `--conf 0.25` 会过滤低于 0.25 的结果。

### 中心坐标

检测框中心点的计算方式是：

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

代码使用整数除法 `// 2` 得到实际像素位置：

```python
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2
```

当前中心坐标只是二维图片中的像素坐标，还不是机械臂可直接使用的三维空间坐标。三维定位、相机标定和坐标转换属于后续阶段。

## 测试与检查

运行中心坐标和跟踪边界单元测试：

```bash
python -m unittest discover -s tests -v
```

检查三个入口能否正常加载：

```bash
python src/detect_image.py --help
python src/detect_video.py --help
python src/detect_camera.py --help
```

## 后续路线（本阶段不实现）

3. 相机标定与位置估计
4. ROS2 通信
5. 机器人仿真
6. 机械臂运动规划与抓取
7. VLM 自然语言目标理解
8. 完整视觉语言机器人抓取系统

详细路线见 `docs/thesis_notes.md`。继续学习代码时，建议依次阅读 `src/detect_image.py`、`src/utils.py`、`src/detect_video.py`、`src/detect_camera.py`。

第二阶段先阅读 `src/track_video.py`，再读 `src/tracking.py` 中的 `while` 循环，最后读 `src/tracking_utils.py`。算法与参数依据：[Ultralytics 官方跟踪文档](https://docs.ultralytics.com/modes/track/)。
