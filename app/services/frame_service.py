import base64
from pathlib import Path

import cv2


def extract_frames(video_path: Path, every_n_frames: int = 30) -> list[tuple[int, str]]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[tuple[int, str]] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode("utf-8")
            frames.append((idx, f"data:image/jpeg;base64,{b64}"))
        idx += 1

    cap.release()
    return frames
