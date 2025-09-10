from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ActionPrediction:
    """行为分类结果结构（如未来需要返回与框绑定的分类）。

    当前管线仅使用 (label, confidence) 列表；该结构为未来扩展保留。
    - bbox_xyxy: 可选的人体框坐标
    - label: 行为类别名称（如 NORMAL/P0/P1/P2 等）
    - confidence: 该类别的置信度
    """
    bbox_xyxy: Tuple[int, int, int, int]
    label: str
    confidence: float


class BaseBehaviorClassifier(ABC):
    """行为分类器接口定义。

    任意具体实现需遵循 `predict(crops)->[(label, conf), ...]` 约定，
    以便与检测模块输出顺序对齐，实现即插即用。
    """

    @abstractmethod
    def predict(self, person_crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """对传入的人体裁剪图像列表进行行为分类。

        约定：
        - 输入 `person_crops` 与检测到的行人一一对应，顺序一致。
        - 输出为 [(label, confidence), ...]，长度与输入一致。
        """
        raise NotImplementedError




