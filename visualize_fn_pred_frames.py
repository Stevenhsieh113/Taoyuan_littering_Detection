#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
針對 FN 案例，在「預測 frame」上視覺化人/車與垃圾框。

資料來源：
- 人/車框：detect_results.csv
- 垃圾框：mock_trash_results.csv（類別 0/1，顯示時仍為原始 ID）
- 若該 frame 有疑似配對紀錄：suspected_littering_with_tracker.csv（加粗連線）
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ==========================================
#               【設定區塊】
# ==========================================
IMAGE_ROOT = Path(r"C:\桃園市政府\已標註和轉yolo\image")
DETECT_CSV = Path("detect_results.csv")
TRASH_CSV = Path("mock_trash_results.csv")
TRACKER_CSV = Path("suspected_littering_with_tracker.csv")
FN_DETAIL_CSV = Path("evaluation_report/06_fn_only.csv")
VIDEO_SUMMARY_CSV = Path("evaluation_report/03_by_video.csv")
OUTPUT_DIR = Path("evaluation_report/fn_pred_frame_visualizations")

CLASS_PERSON_CAR = {0, 1, 2, 3, 5, 7}

COLOR_PC = (0, 255, 0)       # 綠：人/車
COLOR_TRASH = (0, 0, 255)    # 紅：垃圾
COLOR_PAIR_PC = (0, 255, 255)
COLOR_PAIR_TR = (255, 0, 255)
COLOR_LINE = (255, 255, 0)
THICKNESS = 2
# ==========================================


def extract_frame_num(filename: str) -> int:
    nums = re.findall(r"(\d+)", Path(filename).stem)
    return int(nums[-1]) if nums else -1


def load_cv2_image(path: Path):
    try:
        pil = Image.open(path)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except OSError:
        return None


def find_image_path(video_name: str, frame_num: int) -> Path | None:
    folder = IMAGE_ROOT / video_name
    stem = f"frame_{frame_num:06d}"
    for ext in (".png", ".jpg", ".jpeg"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_detect_lookup() -> dict[tuple[str, int], list[dict]]:
    lookup: dict[tuple[str, int], list[dict]] = {}
    df = pd.read_csv(DETECT_CSV, encoding="utf-8-sig")
    for _, row in df.iterrows():
        if int(row["類別ID"]) not in CLASS_PERSON_CAR:
            continue
        video = Path(row["資料夾路徑"]).name
        frame = extract_frame_num(row["圖片檔名"])
        lookup.setdefault((video, frame), []).append(row.to_dict())
    return lookup


def load_trash_lookup() -> dict[tuple[str, int], list[dict]]:
    lookup: dict[tuple[str, int], list[dict]] = {}
    df = pd.read_csv(TRASH_CSV, encoding="utf-8-sig")
    for _, row in df.iterrows():
        video = Path(row["資料夾路徑"]).name
        frame = extract_frame_num(row["圖片檔名"])
        lookup.setdefault((video, frame), []).append(row.to_dict())
    return lookup


def load_tracker_lookup() -> dict[tuple[str, int], list[dict]]:
    lookup: dict[tuple[str, int], list[dict]] = {}
    if not TRACKER_CSV.exists():
        return lookup
    df = pd.read_csv(TRACKER_CSV, encoding="utf-8-sig")
    for _, row in df.iterrows():
        video = Path(row["資料夾路徑"]).name
        frame = extract_frame_num(row["圖片檔名"])
        lookup.setdefault((video, frame), []).append(row.to_dict())
    return lookup


def build_fn_pred_cases() -> list[dict]:
    """
    從評估報表整理：每支有漏抓的片，取第一個預測 frame 做視覺化。
    並附上該片所有漏抓 GT frame。
    """
    if not FN_DETAIL_CSV.exists() or not VIDEO_SUMMARY_CSV.exists():
        raise FileNotFoundError("請先執行 export_evaluation_report.py 產生 evaluation_report/")

    fn_df = pd.read_csv(FN_DETAIL_CSV, encoding="utf-8-sig")
    video_df = pd.read_csv(VIDEO_SUMMARY_CSV, encoding="utf-8-sig")
    video_map = {row["Video_Name"]: row for _, row in video_df.iterrows()}

    cases: list[dict] = []
    seen: set[str] = set()

    for video_name, group in fn_df.groupby("Video_Name"):
        if video_name not in video_map:
            continue
        vrow = video_map[video_name]
        pred_text = str(vrow.get("預測frame列表", "")).strip()
        if not pred_text or pred_text in {"—", "nan", "NaN", "None"}:
            continue
        try:
            pred_frame = int(float(pred_text.split("、")[0]))
        except ValueError:
            continue
        gt_fn_frames = group["GT起始frame"].astype(int).tolist()
        key = f"{video_name}|{pred_frame}"
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            {
                "video_name": video_name,
                "pred_frame": pred_frame,
                "gt_fn_frames": gt_fn_frames,
            }
        )
    return cases


def draw_box(img, x1, y1, x2, y2, color, label: str, thick: int = THICKNESS):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    cv2.putText(
        img,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def render_case(case: dict, detect_lookup, trash_lookup, tracker_lookup) -> Path | None:
    video = case["video_name"]
    pred_frame = case["pred_frame"]
    img_path = find_image_path(video, pred_frame)
    if img_path is None:
        print(f"[略過] 找不到圖片: {video} frame {pred_frame}")
        return None

    img = load_cv2_image(img_path)
    if img is None:
        print(f"[略過] 無法讀圖: {img_path}")
        return None

    key = (video, pred_frame)

    for row in detect_lookup.get(key, []):
        draw_box(
            img,
            row["左上X"],
            row["左上Y"],
            row["右下X"],
            row["右下Y"],
            COLOR_PC,
            f"PC cls{int(row['類別ID'])}",
        )

    for row in trash_lookup.get(key, []):
        draw_box(
            img,
            row["左上X"],
            row["左上Y"],
            row["右下X"],
            row["右下Y"],
            COLOR_TRASH,
            f"Trash cls{int(row['類別ID'])}",
        )

    for row in tracker_lookup.get(key, []):
        draw_box(
            img,
            row["左上X_人車"],
            row["左上Y_人車"],
            row["右下X_人車"],
            row["右下Y_人車"],
            COLOR_PAIR_PC,
            "PAIR PC",
            thick=3,
        )
        draw_box(
            img,
            row["左上X_垃圾"],
            row["左上Y_垃圾"],
            row["右下X_垃圾"],
            row["右下Y_垃圾"],
            COLOR_PAIR_TR,
            "PAIR Trash",
            thick=3,
        )
        c_pc = (int(row["中心X_人車"]), int(row["中心Y_人車"]))
        c_tr = (int(row["中心X_垃圾"]), int(row["中心Y_垃圾"]))
        cv2.line(img, c_pc, c_tr, COLOR_LINE, 2)
        cv2.putText(
            img,
            f"Dist {row['距離']:.1f}px",
            (c_pc[0], c_pc[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_LINE,
            2,
            cv2.LINE_AA,
        )

    gt_text = "、".join(str(x) for x in case["gt_fn_frames"])
    deltas = [abs(pred_frame - g) for g in case["gt_fn_frames"]]
    delta_text = "、".join(str(d) for d in deltas)
    header = f"Pred frame {pred_frame} | FN GT: {gt_text} | gap: {delta_text}"
    cv2.rectangle(img, (0, 0), (1280, 36), (0, 0, 0), -1)
    cv2.putText(
        img,
        header[:120],
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", video)
    out_path = OUTPUT_DIR / f"{safe_name}_pred_{pred_frame:06d}.png"
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(out_path)
    return out_path


def main() -> None:
    cases = build_fn_pred_cases()
    detect_lookup = load_detect_lookup()
    trash_lookup = load_trash_lookup()
    tracker_lookup = load_tracker_lookup()

    print(f"準備輸出 {len(cases)} 張 FN 預測 frame 視覺化...")
    saved = 0
    for case in cases:
        out = render_case(case, detect_lookup, trash_lookup, tracker_lookup)
        if out:
            saved += 1
            print(f"  -> {out.name}")

    print(f"\n完成，共 {saved} 張 -> {OUTPUT_DIR.resolve()}")
    print("圖例：綠=detect人/車  紅=mock垃圾  青/紫粗框=系統配對的那一組")


if __name__ == "__main__":
    main()
