import time
import json
import logging
from pathlib import Path
from typing import List
from datetime import datetime

import cv2

from src.config import CONFIG
from src.pose.yolo11_pose_estimator import Yolo11PoseEstimator, PoseDetection
from src.utils.interface import pack_pose_results
from src.camera.camera_manager import CameraManager
from src.ui.key_handler import KeyHandler
from src.ui.alert_visualizer import AlertVisualizer
from src.behavior.parameter_manager import BehaviorParameterManager


def setup_logging(logs_dir: Path) -> None:
    """设置日志系统"""
    logs_dir.mkdir(exist_ok=True)
    
    # 配置日志：综合日志 + 警报专用日志（按日期）
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    app_fh = logging.FileHandler(logs_dir / 'app.log', encoding='utf-8')
    app_fh.setLevel(logging.INFO)
    app_fh.setFormatter(fmt)
    logger.addHandler(app_fh)

    alerts_name = f"alerts-{datetime.now().strftime('%Y%m%d')}.log"
    alert_fh = logging.FileHandler(logs_dir / alerts_name, encoding='utf-8')
    alert_fh.setLevel(logging.WARNING)
    alert_fh.setFormatter(fmt)
    logger.addHandler(alert_fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def main() -> None:
    """主入口：
    - 打开摄像头，读取逐帧图像
    - 使用YOLO11姿势估计进行人体检测和关键点检测
    - 基于关键点运动参数进行异常行为检测
    - 绘制火柴人可视化并支持按键打印标准 JSON 结果
    """
    # 设置日志系统
    logs_dir = Path('logs')
    setup_logging(logs_dir)
    
    # 初始化摄像头管理器
    camera_manager = CameraManager(CONFIG)
    if not camera_manager.initialize():
        print("摄像头初始化失败，程序退出")
        return

    # 初始化YOLO11姿势估计器
    pose_estimator = Yolo11PoseEstimator(
        model_name=CONFIG.pose.model_name,
        confidence_threshold=CONFIG.pose.confidence_threshold,
        image_size=CONFIG.pose.image_size,
        device=CONFIG.pose.device if hasattr(CONFIG.pose, 'device') else None,
    )
    
    # 初始化参数管理器
    parameter_manager = BehaviorParameterManager(CONFIG, pose_estimator)
    
    # 初始化UI组件
    key_handler = KeyHandler(CONFIG, pose_estimator)
    alert_visualizer = AlertVisualizer(CONFIG, logs_dir)

    start_t = time.time()
    frames = 0

    while True:
        # 读取摄像头帧
        ok, frame = camera_manager.read_frame()
        if not ok:
            # 小等待避免空转
            cv2.waitKey(5)
            continue

        # 记录本帧起始时间
        t0_frame = time.perf_counter()

        # 姿势估计和异常行为检测
        t_pose_start = time.perf_counter()
        pose_detections: List[PoseDetection] = pose_estimator.detect_poses(frame)
        # 估计帧间隔（秒），用于速度/加速度归一化
        dt = max(time.perf_counter() - t_pose_start, 1e-3)
        behavior_results = pose_estimator.detect_abnormal_behavior(pose_detections, dt=dt)
        t_pose_end = time.perf_counter()

        # 绘制火柴人可视化
        for pose_detection, behavior_result in zip(pose_detections, behavior_results):
            frame = pose_estimator.draw_pose_skeleton(frame, pose_detection, behavior_result)

        # 处理报警逻辑
        any_alert = alert_visualizer.process_alerts(pose_detections, behavior_results, frame)

        # 绘制报警横幅
        frame = alert_visualizer.draw_alert_banner(frame)

        # 计算耗时（ms），均保留 1 位小数
        pose_ms = (t_pose_end - t_pose_start) * 1000.0
        total_ms = (t_pose_end - t0_frame) * 1000.0

        # 绘制性能监控HUD
        frame = alert_visualizer.draw_performance_hud(frame, pose_ms, total_ms, start_t, frames + 1)

        # 绘制调试信息
        debug_text = key_handler.get_debug_text(pose_detections)
        frame = alert_visualizer.draw_debug_info(frame, debug_text)

        # 展示窗口
        cv2.imshow(CONFIG.video.window_name, frame)
        frames += 1

        # 处理按键输入
        key = cv2.waitKey(1) & 0xFF
        
        # 如果用户点击右上角关闭按钮，窗口会变为不可见，这里检测后退出
        if cv2.getWindowProperty(CONFIG.video.window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        # 按 'P' 打印当前帧 JSON 结果（遵循接口规范）
        if key in (ord('p'), ord('P')):
            packed = pack_pose_results(pose_detections, behavior_results)
            print(json.dumps(packed, ensure_ascii=False), flush=True)
            
        # 处理热键调参
        key_handler.handle_key(key)
        
        # ESC 或 Q 退出
        if key in (27, ord('q'), ord('Q')):
            break

    # 清理资源
    camera_manager.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()