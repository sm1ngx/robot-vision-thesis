"""在本机提供 MJPEG 流，验证手机网络视频流入口的实际读取与输出。"""

import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent


def main():
    image = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"))
    if image is None:
        raise SystemExit("缺少 data/images/bus.jpg")
    image = cv2.resize(image, (480, 640))
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg_bytes = encoded.tobytes()

    class StreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != "/video":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for _ in range(20):
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode())
                    self.wfile.write(jpeg_bytes + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.06)
            except (BrokenPipeError, ConnectionResetError):
                pass  # 测试客户端达到帧数上限后会关闭连接

    server = ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "phone_test.mp4"
            url = f"http://127.0.0.1:{server.server_port}/video"
            command = [sys.executable, "src/track_stream.py", "--url", url,
                       "--output", str(output), "--no-display", "--max-frames", "4"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                    text=True, timeout=60)
            if result.returncode:
                raise AssertionError(result.stdout + "\n" + result.stderr)
            with output.with_suffix(".jsonl").open(encoding="utf-8") as file:
                records = [json.loads(line) for line in file]
            assert [record["frame"] for record in records] == [1, 2, 3, 4]
            assert sum(len(record["objects"]) for record in records) > 0
            capture = cv2.VideoCapture(str(output))
            decoded = 0
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    assert frame.shape[:2] == (640, 480)
                    decoded += 1
            finally:
                capture.release()
            assert decoded == 4
            print("本地 MJPEG 流验证通过：读取、YOLO 跟踪、MP4 与 JSONL 均正常（4 帧）。")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
