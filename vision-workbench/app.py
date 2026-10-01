"""机器人视觉工作台的本地网页入口：python app.py。"""

import argparse
import sys
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from workbench.image_service import detect_image
from workbench.model_manager import MODEL_CHOICES


def run_image_detection(image_path, model_name, confidence, iou):
    """接收界面输入并把检测结果分配给各个输出组件。"""
    try:
        result = detect_image(image_path, model_name, confidence, iou)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as error:
        return None, f"检测未完成：{error}", None, None, None

    record = result["record"]
    counts = record["class_counts"]
    count_text = "、".join(f"{name}：{count}" for name, count in counts.items()) or "暂无"
    inference_ms = record["inference_time_ms"]
    # 单张图片的 FPS 只是推理耗时的倒数，不代表摄像头的实时帧率。
    estimated_fps = 1000 / inference_ms if inference_ms > 0 else 0
    summary = (
        f"**检测完成** · 模型：{record['model']}  \n"
        f"目标总数：{record['total_objects']}  \n"
        f"类别统计：{count_text}  \n"
        f"本次推理：{inference_ms:.1f} ms  \n"
        f"理论推理速度：{estimated_fps:.1f} 次/秒（非实时摄像头 FPS）"
    )
    return (
        result["image"],
        summary,
        record,
        result["image_path"],
        result["json_path"],
    )


def build_app():
    """构造本地 Gradio 页面；此函数不启动服务器，方便测试。"""
    with gr.Blocks(title="机器人视觉目标检测与模型评估系统") as app:
        gr.Markdown("# 机器人视觉目标检测与模型评估系统")
        gr.Markdown("当前是 Web 工作台第一阶段：图片检测。视频、摄像头、模型对比和标注将分阶段加入。")

        with gr.Tab("图片检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    source = gr.Image(
                        label="上传图片（JPG / JPEG / PNG）",
                        type="filepath",
                        sources=["upload"],
                    )
                    model = gr.Dropdown(
                        label="模型",
                        choices=list(MODEL_CHOICES),
                        value=next(iter(MODEL_CHOICES)),
                        interactive=False,
                    )
                    confidence = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Confidence 阈值")
                    iou = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="IoU 阈值")
                    start = gr.Button("开始检测", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        original = gr.Image(label="原始图片", interactive=False)
                        annotated = gr.Image(label="检测结果（框 / 类别 / 置信度 / 中心点）", interactive=False)

                with gr.Column(scale=1):
                    summary = gr.Markdown("上传图片并点击“开始检测”。")
                    details = gr.JSON(label="检测数据与性能")
                    image_file = gr.File(label="下载检测图片")
                    json_file = gr.File(label="下载检测 JSON")

            # 上传后同步显示原图；按钮点击才执行 YOLO 推理。
            source.change(lambda path: path, inputs=source, outputs=original)
            start.click(
                run_image_detection,
                inputs=[source, model, confidence, iou],
                outputs=[annotated, summary, details, image_file, json_file],
            )

        with gr.Tab("后续阶段"):
            gr.Markdown(
                "电脑摄像头 → 上传视频 → 手机网络摄像头 → 自定义模型 → "
                "模型对比 → 人工标注与 YOLO 导出。**这些 Web 功能尚未实现。** "
                "原有命令行图片、视频、摄像头与跟踪入口继续可用。"
            )
    return app


def main():
    parser = argparse.ArgumentParser(description="启动本地机器人视觉工作台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    parser.add_argument("--port", type=int, default=7860, help="端口，默认 7860")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        allowed_paths=[str(PROJECT_ROOT / "outputs" / "images")],
    )


if __name__ == "__main__":
    main()
