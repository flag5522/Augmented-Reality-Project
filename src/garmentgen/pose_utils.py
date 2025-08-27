from __future__ import annotations

import json
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def load_pose_image(path: str, size_wh: Optional[Tuple[int, int]] = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size_wh is not None:
        image = image.resize(size_wh)
    return image


def render_pose_from_coco(keypoints_json_path: str, size_wh: Tuple[int, int]) -> Image.Image:
    with open(keypoints_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expect COCO person keypoints in either single object or annotations[0]
    if isinstance(data, dict) and "people" in data:
        # OpenPose JSON format
        keypoints = data["people"][0]["pose_keypoints_2d"]
    elif isinstance(data, dict) and "annotations" in data:
        keypoints = data["annotations"][0]["keypoints"]
    elif isinstance(data, dict) and "keypoints" in data:
        keypoints = data["keypoints"]
    elif isinstance(data, list):
        keypoints = data[0]
    else:
        raise ValueError("Unsupported keypoints JSON format")

    # Normalize to list of (x,y,v)
    if isinstance(keypoints, list) and all(isinstance(v, (int, float)) for v in keypoints):
        # Flat list [x1,y1,v1,x2,y2,v2,...]
        pts = [(keypoints[i], keypoints[i + 1], keypoints[i + 2]) for i in range(0, len(keypoints), 3)]
    elif isinstance(keypoints, list) and all(isinstance(v, (list, tuple)) for v in keypoints):
        pts = [(p[0], p[1], p[2] if len(p) > 2 else 1.0) for p in keypoints]
    else:
        raise ValueError("Unrecognized keypoints structure")

    width, height = size_wh
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Skeleton edges for COCO-17 ordering
    edges = [
        (5, 7), (7, 9),      # left arm
        (6, 8), (8, 10),     # right arm
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16),  # right leg
        (5, 6),              # shoulders
        (11, 12),            # hips
        (5, 11), (6, 12),    # torso diagonals
        (0, 1), (1, 2), (2, 3), (3, 4),  # head/face approx
        (0, 5), (0, 6),
    ]

    def valid(idx: int) -> bool:
        if idx < 0 or idx >= len(pts):
            return False
        x, y, v = pts[idx]
        return v > 0 and 0 <= x < width and 0 <= y < height

    # Draw with OpenCV for speed if available
    try:
        import cv2

        for a, b in edges:
            if valid(a) and valid(b):
                ax, ay, _ = pts[a]
                bx, by, _ = pts[b]
                cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), (255, 255, 255), 4)

        for i, (x, y, v) in enumerate(pts):
            if v > 0 and 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (int(x), int(y)), 5, (255, 255, 255), -1)
    except Exception:
        # Fallback with PIL drawing
        from PIL import ImageDraw

        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        for a, b in edges:
            if valid(a) and valid(b):
                ax, ay, _ = pts[a]
                bx, by, _ = pts[b]
                draw.line([(ax, ay), (bx, by)], fill=(255, 255, 255), width=4)
        for x, y, v in pts:
            if v > 0 and 0 <= x < width and 0 <= y < height:
                r = 5
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255))
        return pil_img

    return Image.fromarray(canvas)

