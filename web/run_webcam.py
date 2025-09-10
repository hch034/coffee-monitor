import time
import json
from typing import List

import cv2

from src.config import CONFIG
from src.detector.yolo_detector import YoloPersonDetector, Detection
from src.classifier.dummy_classifier import DummyBehaviorClassifier
from src.utils.vis import draw_boxes
from src.utils.interface import pack_results


def format_fps_text(start_time: float, frames: int) -> str:
    """计算并返回 FPS 文本（保留 1 位小数）。"""
    dt = max(time.time() - start_time, 1e-6)
    fps = frames / dt
    return f"FPS: {fps:.1f}"


def main() -> None:
    """主入口：
    - 打开摄像头，读取逐帧图像
    - 调用 YOLO 进行人体检测
    - 对检测到的人体进行裁剪并送入占位分类器
    - 绘制可视化并支持按键打印标准 JSON 结果
    """
    cap = cv2.VideoCapture(CONFIG.video.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查设备与权限。")

    # 初始化检测与分类模块
    detector = YoloPersonDetector(
        model_name=CONFIG.detection.model_name,
        confidence_threshold=CONFIG.detection.confidence_threshold,
        iou_threshold=CONFIG.detection.iou_threshold,
    )
    classifier = DummyBehaviorClassifier()

    start_t = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头帧失败。")
            break

        # 检测人体（默认仅保留 person 类别）
        detections: List[Detection] = detector.detect(frame, CONFIG.detection.target_class_names)

        # 裁剪人体区域，送入占位分类器
        crops = []
        for x1, y1, x2, y2 in [d.box_xyxy for d in detections]:
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            crops.append(frame[y1c:y2c, x1c:x2c])
        action_preds = classifier.predict(crops) if crops else []

        # 叠加可视化信息：类别/置信度与动作结果（全部保留 1 位小数）
        boxes = [d.box_xyxy for d in detections]
        labels = []
        for idx, d in enumerate(detections):
            if idx < len(action_preds):
                act, acon = action_preds[idx]
                labels.append(f"{d.class_name} {d.confidence:.1f} | {act} {acon:.1f}")
            else:
                labels.append(f"{d.class_name} {d.confidence:.1f}")
        draw_boxes(frame, boxes, labels)

        # 可选显示 FPS
        if CONFIG.video.display_fps:
            txt = format_fps_text(start_t, frames + 1)
            cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 展示窗口
        cv2.imshow(CONFIG.video.window_name, frame)
        frames += 1

        key = cv2.waitKey(1) & 0xFF
        # 按 'P' 打印当前帧 JSON 结果（遵循接口规范）
        if key in (ord('p'), ord('P')):
            packed = pack_results(detections, action_preds)
            print(json.dumps(packed, ensure_ascii=False), flush=True)
        if key == 27:  # ESC 退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


