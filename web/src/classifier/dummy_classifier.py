from typing import List, Tuple
import numpy as np

from .base import BaseBehaviorClassifier


class DummyBehaviorClassifier(BaseBehaviorClassifier):
    """占位分类器。

    作用：
    - 用于在分类模型尚未接入前，打通端到端流程。
    - 始终输出 ("NORMAL", 0.5)；便于联调可视化与接口。
    """

    def predict(self, person_crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """返回与输入等长的固定结果列表。

        - person_crops: 人体裁剪图像列表
        - 返回: [("NORMAL", 0.5), ...]
        """
        return [("NORMAL", 0.50) for _ in person_crops]




