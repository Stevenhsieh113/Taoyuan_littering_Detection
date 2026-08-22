#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
讀取 labels_8class 底下各影片資料夾的 YOLO 標註，
只保留 class_id == 7（trash_flying），
計算「有標註」的相鄰兩幀之間，垃圾中心點的像素距離，並匯出 CSV。

輸出欄位：資料夾路徑、資料夾名稱、第幾frame、距離
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from PIL import Image

# ==========================================
#               【設定區塊】
# ==========================================
LABELS_ROOT = Path(r"C:\桃園市政府\已標註和轉yolo\labels_8class")
IMAGES_ROOT = Path(r"C:\桃園市政府\已標註和轉yolo\image")
OUTPUT_CSV = Path("trash_flying_adjacent_frame_distances.csv")
TARGET_CLASS_ID = 7  # trash_flying
VALID_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

FRAME_RX = re.compile(r"(\d+)")
# ==========================================


def extract_frame_num(filename: str) -> int | None:
    nums = FRAME_RX.findall(Path(filename).stem)
    return int(nums[-1]) if nums else None


def parse_yolo_centers(txt_path: Path, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """解析 YOLO 行，只保留 class_id == 7，回傳中心點（像素）。"""
    centers: list[tuple[float, float]] = []
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            if class_id != TARGET_CLASS_ID:
                continue
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
        except ValueError:
            continue
        centers.append((cx_norm * img_w, cy_norm * img_h))
    return centers


def find_image_path(video_dir_name: str, txt_path: Path) -> Path | None:
    base = IMAGES_ROOT / video_dir_name / txt_path.stem
    for ext in VALID_IMAGE_SUFFIXES:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def min_center_distance(
    centers_a: list[tuple[float, float]],
    centers_b: list[tuple[float, float]],
) -> float:
    """兩幀若有多個 class 7 框，取中心點之間的最小距離。"""
    return min(
        math.hypot(ax - bx, ay - by)
        for ax, ay in centers_a
        for bx, by in centers_b
    )


def process_video_folder(video_dir: Path) -> list[dict]:
    rows: list[dict] = []
    folder_path = str(video_dir)
    folder_name = video_dir.name

    txt_files = sorted(video_dir.glob("*.txt"))
    frames_with_trash: list[tuple[int, Path, list[tuple[float, float]]]] = []

    for txt_path in txt_files:
        frame_num = extract_frame_num(txt_path.name)
        if frame_num is None:
            continue

        image_path = find_image_path(folder_name, txt_path)
        if image_path is None:
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        centers = parse_yolo_centers(txt_path, img_w, img_h)
        if not centers:
            continue

        frames_with_trash.append((frame_num, txt_path, centers))

    frames_with_trash.sort(key=lambda x: x[0])

    for i in range(1, len(frames_with_trash)):
        prev_frame, _, prev_centers = frames_with_trash[i - 1]
        curr_frame, _, curr_centers = frames_with_trash[i]
        distance = min_center_distance(prev_centers, curr_centers)

        rows.append(
            {
                "資料夾路徑": folder_path,
                "資料夾名稱": folder_name,
                "第幾frame": curr_frame,
                "距離": round(distance, 2),
            }
        )

    return rows


def main() -> None:
    if not LABELS_ROOT.exists():
        raise FileNotFoundError(f"找不到標註根目錄：{LABELS_ROOT}")

    video_dirs = sorted(d for d in LABELS_ROOT.iterdir() if d.is_dir())
    if not video_dirs:
        print(f"找不到任何子資料夾：{LABELS_ROOT}")
        return

    all_rows: list[dict] = []

    for video_dir in video_dirs:
        rows = process_video_folder(video_dir)
        all_rows.extend(rows)
        print(f"[OK] {video_dir.name}: {len(rows)} 筆相鄰幀距離 (class {TARGET_CLASS_ID})")

    fieldnames = ["資料夾路徑", "資料夾名稱", "第幾frame", "距離"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n完成！共 {len(all_rows)} 筆 -> {OUTPUT_CSV.resolve()}")
    print(f"處理 {len(video_dirs)} 個影片資料夾")


if __name__ == "__main__":
    main()
