from typing import Iterable, Tuple
import cv2


Color = Tuple[int, int, int]


def draw_boxes(
    frame,
    boxes: Iterable[Tuple[int, int, int, int]],
    labels: Iterable[str] | None = None,
    color: Color = (0, 200, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
):
    if labels is None:
        labels = [""] * len(list(boxes))

    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )


