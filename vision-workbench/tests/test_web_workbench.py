"""Web 工作台第一阶段的输入检查与实际图片检测测试。"""

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from workbench.image_service import detect_image


class ImageWorkbenchTests(unittest.TestCase):
    def test_missing_image_gives_clear_message(self):
        with self.assertRaisesRegex(ValueError, "请先上传"):
            detect_image("", "YOLO11n（CPU，默认）", 0.25, 0.7)

    def test_invalid_suffix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "只支持"):
            detect_image("sample.txt", "YOLO11n（CPU，默认）", 0.25, 0.7)

    def test_sample_image_creates_annotated_jpg_and_json(self):
        sample = PROJECT_ROOT / "data" / "images" / "bus.jpg"
        model = PROJECT_ROOT / "models" / "yolo11n.pt"
        if not sample.is_file() or not model.is_file():
            self.skipTest("缺少示例图片或模型；请先按 README 准备这两个文件")

        result = detect_image(str(sample), "YOLO11n（CPU，默认）", 0.25, 0.7)
        image_output = Path(result["image_path"])
        json_output = Path(result["json_path"])
        self.assertTrue(image_output.is_file())
        self.assertTrue(json_output.is_file())
        self.assertEqual(result["image"].ndim, 3)

        record = json.loads(json_output.read_text(encoding="utf-8"))
        self.assertEqual(record["total_objects"], len(record["detections"]))
        self.assertEqual(record["total_objects"], sum(record["class_counts"].values()))
        self.assertEqual(record["model"], "yolo11n.pt")


if __name__ == "__main__":
    unittest.main()
