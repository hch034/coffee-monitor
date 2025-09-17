"""
可视化报警模块
负责报警横幅显示、截图保存等可视化功能
"""

import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.pose.yolo11_pose_estimator import PoseDetection


class AlertVisualizer:
    """报警可视化器，负责报警相关的UI显示"""
    
    def __init__(self, config, logs_dir: Path):
        """
        初始化报警可视化器
        
        Args:
            config: 配置对象
            logs_dir: 日志目录路径
        """
        self.config = config
        self.logs_dir = logs_dir
        self.alert_hold_counter = 0
        self.last_alert_label = ""
        self.alert_snapshot_saved = False
    
    def process_alerts(self, pose_detections: List[PoseDetection], 
                      behavior_results: List[Tuple[str, float]], 
                      frame: np.ndarray) -> bool:
        """
        处理报警逻辑，包括日志记录和截图保存
        
        Args:
            pose_detections: 姿势检测结果列表
            behavior_results: 行为检测结果列表
            frame: 当前帧图像
            
        Returns:
            bool: 是否有报警
        """
        any_alert = False
        
        # 检查是否有异常行为
        for pose_detection, behavior_result in zip(pose_detections, behavior_results):
            if behavior_result[0] != 'NORMAL':
                any_alert = True
                # 记录报警日志
                x1, y1, x2, y2 = pose_detection.person_box
                logging.warning(
                    f"报警 {behavior_result[0]} 置信度={behavior_result[1]:.2f} 框=({x1},{y1},{x2},{y2})"
                )
                self.last_alert_label = behavior_result[0]
        
        # 处理报警横幅显示逻辑
        if any_alert:
            self.alert_hold_counter = self.config.pose.alert_hold_frames
            if self.config.pose.save_alert_snapshot and not self.alert_snapshot_saved:
                self._save_alert_snapshot(frame)
        else:
            if self.alert_hold_counter > 0:
                self.alert_hold_counter -= 1
            if self.alert_hold_counter == 0:
                self.alert_snapshot_saved = False
        
        return any_alert
    
    def _save_alert_snapshot(self, frame: np.ndarray):
        """保存报警截图"""
        alerts_dir = self.logs_dir / self.config.pose.alert_snapshot_subdir
        alerts_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d-%H%M%S')
        fname = alerts_dir / f"alert-{ts}.jpg"
        try:
            cv2.imwrite(str(fname), frame)
            logging.warning(f"报警截图已保存: {fname}")
            self.alert_snapshot_saved = True
        except Exception as e:
            logging.error(f"保存报警截图失败: {e}")
    
    def draw_alert_banner(self, frame: np.ndarray) -> np.ndarray:
        """
        在帧上绘制报警横幅
        
        Args:
            frame: 输入帧图像
            
        Returns:
            np.ndarray: 绘制了报警横幅的帧图像
        """
        if self.alert_hold_counter <= 0:
            return frame
        
        banner = f"{self.config.pose.alert_banner_text}：{self.last_alert_label}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 255), -1)
        
        if getattr(self.config.pose, 'use_chinese_font', False):
            try:
                # 用Pillow绘制中文，避免OpenCV不支持中文导致乱码
                from PIL import Image, ImageDraw, ImageFont
                font = ImageFont.truetype(self.config.pose.chinese_font_path, 20)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 6), banner, font=font, fill=(255, 255, 255))
                frame = np.array(img_pil)
            except Exception:
                cv2.putText(frame, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def draw_performance_hud(self, frame: np.ndarray, pose_ms: float, total_ms: float, 
                           start_time: float, frames: int) -> np.ndarray:
        """
        绘制性能监控HUD
        
        Args:
            frame: 输入帧图像
            pose_ms: 姿势估计耗时（毫秒）
            total_ms: 总耗时（毫秒）
            start_time: 开始时间
            frames: 帧数
            
        Returns:
            np.ndarray: 绘制了性能HUD的帧图像
        """
        # 显示FPS
        if self.config.video.display_fps:
            dt = max(time.time() - start_time, 1e-6)
            fps = frames / dt
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 显示耗时信息
        hud = f"POSE {pose_ms:.1f} ms | TOT {total_ms:.1f} ms"
        cv2.putText(frame, hud, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)
        
        return frame
    
    def draw_debug_info(self, frame: np.ndarray, debug_text: str) -> np.ndarray:
        """
        绘制调试信息
        
        Args:
            frame: 输入帧图像
            debug_text: 调试文本
            
        Returns:
            np.ndarray: 绘制了调试信息的帧图像
        """
        if debug_text:
            cv2.putText(frame, debug_text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 50), 2, cv2.LINE_AA)
        
        return frame
