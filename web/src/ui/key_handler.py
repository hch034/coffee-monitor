"""
热键调参模块
负责处理运行时热键输入，动态调整异常行为检测参数
"""

from typing import Callable, Dict, Any
import cv2


class KeyHandler:
    """热键处理器，负责运行时参数调整"""
    
    def __init__(self, config, pose_estimator):
        """
        初始化热键处理器
        
        Args:
            config: 配置对象，包含所有可调参数
            pose_estimator: 姿势估计器对象，用于应用参数变更
        """
        self.config = config
        self.pose_estimator = pose_estimator
        self._setup_key_mappings()
    
    def _setup_key_mappings(self):
        """设置热键映射关系"""
        self.key_mappings = {
            # 速度阈值调整
            ord('+'): self._increase_speed_threshold,
            ord('='): self._increase_speed_threshold,
            ord('-'): self._decrease_speed_threshold,
            ord('_'): self._decrease_speed_threshold,
            
            # 加速度阈值调整
            ord('['): self._decrease_acceleration_threshold,
            ord(']'): self._increase_acceleration_threshold,
            
            # 抬臂角度调整
            ord(';'): self._decrease_arm_raise_angle,
            ord('\''): self._increase_arm_raise_angle,
            
            # 决策窗口调整
            ord(','): self._decrease_decision_window,
            ord('.'): self._increase_decision_window,
            
            # 冷却时间调整
            ord('0'): self._decrease_cooldown,
            ord('9'): self._increase_cooldown,
        }
    
    def handle_key(self, key: int) -> bool:
        """
        处理按键输入
        
        Args:
            key: 按键码
            
        Returns:
            bool: 如果处理了按键返回True，否则返回False
        """
        if key in self.key_mappings:
            self.key_mappings[key]()
            return True
        return False
    
    def _increase_speed_threshold(self):
        """增加速度阈值"""
        self.config.pose.speed_threshold += 5.0
        self.pose_estimator.set_speed_threshold(self.config.pose.speed_threshold)
        print(f"speed_threshold -> {self.config.pose.speed_threshold}")
    
    def _decrease_speed_threshold(self):
        """减少速度阈值"""
        self.config.pose.speed_threshold = max(0.0, self.config.pose.speed_threshold - 5.0)
        self.pose_estimator.set_speed_threshold(self.config.pose.speed_threshold)
        print(f"speed_threshold -> {self.config.pose.speed_threshold}")
    
    def _increase_acceleration_threshold(self):
        """增加加速度阈值"""
        self.config.pose.acceleration_threshold += 10.0
        self.pose_estimator.set_acceleration_threshold(self.config.pose.acceleration_threshold)
        print(f"acceleration_threshold -> {self.config.pose.acceleration_threshold}")
    
    def _decrease_acceleration_threshold(self):
        """减少加速度阈值"""
        self.config.pose.acceleration_threshold = max(0.0, self.config.pose.acceleration_threshold - 10.0)
        self.pose_estimator.set_acceleration_threshold(self.config.pose.acceleration_threshold)
        print(f"acceleration_threshold -> {self.config.pose.acceleration_threshold}")
    
    def _increase_arm_raise_angle(self):
        """增加抬臂角度阈值"""
        self.config.pose.arm_raise_angle_deg += 5.0
        self.pose_estimator.set_arm_raise_angle_threshold(self.config.pose.arm_raise_angle_deg)
        print(f"arm_raise_angle_deg -> {self.config.pose.arm_raise_angle_deg}")
    
    def _decrease_arm_raise_angle(self):
        """减少抬臂角度阈值"""
        self.config.pose.arm_raise_angle_deg = max(0.0, self.config.pose.arm_raise_angle_deg - 5.0)
        self.pose_estimator.set_arm_raise_angle_threshold(self.config.pose.arm_raise_angle_deg)
        print(f"arm_raise_angle_deg -> {self.config.pose.arm_raise_angle_deg}")
    
    def _increase_decision_window(self):
        """增加决策窗口大小"""
        self.config.pose.decision_window += 1
        self.pose_estimator.set_decision_window(self.config.pose.decision_window)
        print(f"decision_window -> {self.config.pose.decision_window}")
    
    def _decrease_decision_window(self):
        """减少决策窗口大小"""
        self.config.pose.decision_window = max(1, self.config.pose.decision_window - 1)
        self.pose_estimator.set_decision_window(self.config.pose.decision_window)
        print(f"decision_window -> {self.config.pose.decision_window}")
    
    def _increase_cooldown(self):
        """增加冷却时间"""
        self.config.pose.cooldown_after_alert += 1
        self.pose_estimator.set_cooldown_after_alert(self.config.pose.cooldown_after_alert)
        print(f"cooldown_after_alert -> {self.config.pose.cooldown_after_alert}")
    
    def _decrease_cooldown(self):
        """减少冷却时间"""
        self.config.pose.cooldown_after_alert = max(0, self.config.pose.cooldown_after_alert - 1)
        self.pose_estimator.set_cooldown_after_alert(self.config.pose.cooldown_after_alert)
        print(f"cooldown_after_alert -> {self.config.pose.cooldown_after_alert}")
    
    def get_debug_text(self, pose_detections) -> str:
        """
        获取调试信息文本，用于HUD显示
        
        Args:
            pose_detections: 姿势检测结果列表
            
        Returns:
            str: 调试信息文本
        """
        if not pose_detections:
            return ""
        
        return (
            f"SPD>{self.config.pose.speed_threshold:.0f} ACC>{self.config.pose.acceleration_threshold:.0f} "
            f"ARM>{self.config.pose.arm_raise_angle_deg:.0f} WIN={self.config.pose.decision_window} "
            f"CD={self.config.pose.cooldown_after_alert}"
        )
