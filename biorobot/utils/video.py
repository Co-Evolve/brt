"""Save frames returned by ``env.render`` (``render_mode="rgb_array"``) as a video."""

from pathlib import Path
from typing import Sequence

import numpy as np


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: float) -> None:
    """Write RGB ``uint8`` frames of shape ``(height, width, 3)`` to an MP4 file.

    Args:
        frames: Frames in RGB order, as returned by ``BRTEnv.render``. All frames must have the same size.
        path: Output file; parent directories are created.
        fps: Frames per second; use ``1 / env.config.control_timestep`` to play back in real time when rendering
            every control step.
    """
    import cv2

    if len(frames) == 0:
        raise ValueError("No frames to write.")
    height, width = frames[0].shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        for frame in frames:
            # OpenCV expects BGR; MuJoCo renders RGB.
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
