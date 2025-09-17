"""
摄像头管理模块
负责摄像头的打开、配置、重连等管理功能
"""

import cv2
from typing import Optional


class CameraManager:
    """摄像头管理器"""
    
    def __init__(self, config):
        """
        初始化摄像头管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.fail_count = 0
        self.max_fail_before_reopen = 20
    
    def _configure_camera(self, cap: cv2.VideoCapture) -> None:
        """配置摄像头参数"""
        # 优先尝试MJPG，减轻CPU解码压力，提高快速移动时的稳定性
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
        # 合理分辨率与FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.config.video.target_fps)
        # 尝试减小内部缓冲，降低延迟（部分后端支持）
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
    
    def open_camera_with_fallback(self, preferred_index: int) -> cv2.VideoCapture:
        """
        在 Windows 上尝试多种后端和索引打开摄像头，并设置常用分辨率。
        依次尝试后端：CAP_DSHOW, CAP_MSMF, 默认；索引：preferred, 0, 1, 2, 3。
        成功则返回已打开的 VideoCapture，否则抛出异常。
        """
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
        candidate_indices = []
        for idx in [preferred_index, 0, 1, 2, 3]:
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        for backend in backends:
            for cam_idx in candidate_indices:
                cap = cv2.VideoCapture(cam_idx, backend)
                if cap.isOpened():
                    self._configure_camera(cap)
                    ok, _ = cap.read()
                    if ok:
                        return cap
                    cap.release()
        raise RuntimeError("无法打开摄像头，请检查设备连接、隐私权限与占用情况（关闭相机应用/Teams/Zoom等）。")
    
    def initialize(self) -> bool:
        """
        初始化摄像头
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.cap = self.open_camera_with_fallback(self.config.video.camera_index)
            self.fail_count = 0
            return True
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
    
    def read_frame(self) -> tuple[bool, Optional[cv2.Mat]]:
        """
        读取摄像头帧
        
        Returns:
            tuple: (是否成功, 帧图像)
        """
        if self.cap is None:
            return False, None
        
        # 先多次grab刷新缓冲，减低因抖动造成的失败
        ok = False
        for _ in range(max(1, self.config.video.max_grab_per_loop)):
            ok = self.cap.grab()
        
        # 再retrieve
        if ok:
            ok, frame = self.cap.retrieve()
        else:
            frame = None
        
        if not ok:
            self.fail_count += 1
            if self.fail_count == 1:
                print("读取摄像头帧失败，进行重试……")
            if self.fail_count >= self.max_fail_before_reopen:
                print("连续读取失败，尝试重启摄像头……")
                if self._reconnect():
                    return self.read_frame()  # 重连成功后重新读取
                else:
                    return False, None
        else:
            # 成功读帧，清零失败计数
            self.fail_count = 0
        
        return ok, frame
    
    def _reconnect(self) -> bool:
        """
        重新连接摄像头
        
        Returns:
            bool: 重连是否成功
        """
        if self.cap:
            self.cap.release()
        
        try:
            self.cap = self.open_camera_with_fallback(self.config.video.camera_index)
            self.fail_count = 0
            print("摄像头已重连。")
            return True
        except Exception as e:
            print(f"重连失败：{e}")
            return False
    
    def release(self):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def is_opened(self) -> bool:
        """检查摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()
