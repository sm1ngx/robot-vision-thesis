# 阶段 1、2 系统架构

## 本地 Web 工作台：当前第 1 阶段与规划

现有 `src/detect_*.py`、`src/track_*.py` 保持独立可运行。新增 `app.py` 是本地 Gradio 入口，`src/workbench/image_service.py` 负责图片数据处理，`src/workbench/model_manager.py` 负责统一模型获取，二者复用原有 `src/utils.py` 的画框与中心坐标代码。默认只在 `127.0.0.1:7860` 提供服务。

```text
网页上传图片与参数
  → app.py 回调
  → image_service.py：OpenCV 读图
  → model_manager.py：加载并缓存 models/yolo11n.pt
  → YOLO.predict(..., device="cpu")
  → utils.py：画框、类别、置信度、中心点
  → Gradio 展示 + outputs/images/ 下的 JPG/JSON
```

采用“一个入口 + 少量业务模块”的结构。后续各阶段才计划增加 `video_service.py`、`camera_service.py`、`network_camera.py`、`model_compare.py`、`annotation/` 等模块；当前这些文件不存在，也不表示功能已完成。以后可让新页面调用相同服务函数，未来若更换 FastAPI + React，模型和推理逻辑不必与页面一起重写。

Web 工作台实施顺序：①图片（已完成）；②电脑摄像头；③上传视频；④手机网络摄像头；⑤自定义权重；⑥模型 A/B；⑦人工标注与 YOLO 导出；⑧日志、性能和文档完善。没有 Ground Truth 时不能计算 Precision、Recall、mAP，只能比较检测结果和速度。

## 范围

阶段 1 建立二维视觉感知闭环，阶段 2 增加连续画面的目标关联。两阶段均使用同一个 YOLO11n 权重，推理设备固定为 CPU。

## 阶段 2 数据流与状态

`track_video.py` / `track_camera.py` / `track_stream.py` 打开输入 → `tracking.py` 逐帧读取 → YOLO11n 检测 → ByteTrack 关联 → `tracking_utils.py` 提取 ID、画轨迹 → MP4 / JSONL / 窗口。手机视频流入口支持 HTTP/MJPEG 或 RTSP 地址，共用原有模型与跟踪算法。

每次启动只创建一个 YOLO 对象，在循环中调用 `model.track(..., persist=True)`。ByteTrack 根据前一帧的位置状态和当前检测框做关联。不同视频分别启动进程，避免继承上一段视频的身份。

跟踪器保留丢失目标的状态由 `configs/bytetrack.yaml` 管理；画面轨迹用普通字典管理，只保留最近 30 帧的观测点，过期 ID 自动删除。目标未观测的帧不画预测框，轨迹线也不跨丢失区间连接。JSONL 流式写入，不把整段视频结果堆在内存中。

资源统一在 `try/finally` 中释放；视频输出在首帧推理成功后创建，尺寸来自实际画面。输入视频、已有输出受到保护。第一阶段入口保持原来的检测行为，第二阶段通过单独入口使用。

## 阶段 1 数据流

```text
输入源
├── 图片 ─── detect_image.py
├── 视频 ─── detect_video.py
└── 摄像头 ─ detect_camera.py
              │
              ▼
        OpenCV 读取为图像数组
              │
              ▼
      YOLO11n 在 CPU 上执行推理
              │
              ▼
  类别、置信度、(x1, y1, x2, y2)
              │
              ▼
      utils.py 计算中心点并绘图
              │
              ▼
      屏幕显示 / 图片或视频保存
```

## 设计选择

- 选择 YOLO11n：模型小，适合没有 CUDA 的学习环境。
- 显式使用 `device="cpu"`：行为清楚，不依赖 NVIDIA 环境。
- 使用三个入口脚本：图片、视频和摄像头的流程不同，分开后更容易逐行学习。
- 只抽取真正重复的代码到 `utils.py`：避免复制画框和中心点逻辑，同时不过度封装。
- 命令行参数优先：第一阶段参数较少，避免为了配置文件引入额外复杂度。

## 当前边界

中心点是二维像素坐标，不能直接代表物体到相机的距离，也不能直接发送给机械臂。要获得机器人坐标，后续还需要深度信息、相机标定、坐标系转换和机械臂标定。本阶段不包含这些功能。
