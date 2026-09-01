import base64
from pathlib import Path

import cv2


def extract_frames(video_path: Path, frames_per_sec: float = 1.0) -> list[tuple[int, str]]:
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(video_fps / frames_per_sec))

    frames: list[tuple[int, str]] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode("utf-8")
            frames.append((idx, f"data:image/jpeg;base64,{b64}"))
        idx += 1

    cap.release()
    return frames
