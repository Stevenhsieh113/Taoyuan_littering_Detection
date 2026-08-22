#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 YOLO 標註中的 Trash (class 2) 拆分成 Trash / Trash_Flying
- 以 YOLO txt 為主體（座標精準）
- 用 XML 的 State 屬性判斷 Flying 或 None
- 只保留 Trash 相關的標註，移除 Person (0) 和 Vehicle (1)

新類別：
  0 = Trash (State=None)
  1 = Trash_Flying (State=Flying)
"""

import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path
import re

# ===================== 設定 =====================
BASE_DIR = r'C:\桃園市政府\已標註和轉yolo'
XML_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'labels_split')  # 輸出到新資料夾，不動原始檔

# xml：<stem>.xml
# txt：<stem>_txt（遞迴找底下所有 *.txt）
TXT_DIR_SUFFIX = '_txt'

# XML 檔名 → YOLO 資料夾名稱的對照（處理命名不一致的情況）
NAME_MAP = {
    '桃園影片 2025年11月17日 1280x720': '桃園影片_2025_11月17日_1280x720',
}

IMG_W, IMG_H = 1280, 720
# ================================================


def xml_box_to_yolo(xtl, ytl, xbr, ybr):
    """將 XML 的像素座標轉成 YOLO 格式（用於比對）"""
    cx = (xtl + xbr) / 2 / IMG_W
    cy = (ytl + ybr) / 2 / IMG_H
    w = (xbr - xtl) / IMG_W
    h = (ybr - ytl) / IMG_H
    return cx, cy, w, h


def coords_match(yolo_coords, xml_coords, tol=1e-4):
    """比對兩組 YOLO 格式座標是否一致（允許微小誤差）"""
    return all(abs(a - b) < tol for a, b in zip(yolo_coords, xml_coords))


def extract_frame_idx_from_txt_filename(txt_filename: str):
    """
    從 txt 檔名抽出 frame_idx 用來對齊 XML 的 box frame。
    例如：frame_000041.txt / 000041.txt / frame41.txt 都可抽到最後一段數字。
    """
    stem = Path(txt_filename).stem
    digit_groups = re.findall(r'\d+', stem)
    if not digit_groups:
        return None
    return int(digit_groups[-1])


def parse_xml_trash_states(xml_path):
    """
    解析 XML，回傳每個 frame 中 Trash box 的 State
    回傳格式: { frame_idx: [ (cx, cy, w, h, state), ... ] }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    trash_info = {}

    for track in root.findall('.//track'):
        if track.attrib['label'] != 'Trash':
            continue

        for box in track.findall('box'):
            if box.get('outside') == '1':
                continue

            frame_idx = int(box.attrib['frame'])
            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])

            # 取得 State 屬性
            state = 'None'
            for attr in box.findall('attribute'):
                if attr.attrib.get('name') == 'State':
                    state = attr.text
                    break

            cx, cy, w, h = xml_box_to_yolo(xtl, ytl, xbr, ybr)

            if frame_idx not in trash_info:
                trash_info[frame_idx] = []
            trash_info[frame_idx].append((cx, cy, w, h, state))

    return trash_info


def process_video(xml_name, folder_name):
    """處理一部影片的所有 frame"""
    xml_path = os.path.join(XML_DIR, xml_name + '.xml')
    src_dir = os.path.join(BASE_DIR, folder_name, 'obj_train_data')
    dst_dir = os.path.join(OUTPUT_DIR, folder_name)

    if not os.path.exists(xml_path):
        print(f"  ⚠️ XML 不存在: {xml_path}")
        return 0, 0, 0, 0

    if not os.path.exists(src_dir):
        print(f"  ⚠️ 資料夾不存在: {src_dir}")
        return 0, 0, 0, 0

    os.makedirs(dst_dir, exist_ok=True)

    # 解析 XML 取得每個 frame 的 Trash State
    trash_info = parse_xml_trash_states(xml_path)

    txt_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.txt')])

    total_trash = 0
    total_flying = 0
    total_removed = 0  # Person + Vehicle 被移除的數量
    unmatched = 0

    for txt_file in txt_files:
        # 取得 frame 編號 (frame_000041.txt → 41)
        frame_idx = int(txt_file.replace('frame_', '').replace('.txt', ''))

        # 讀取原始 YOLO 標註
        src_path = os.path.join(src_dir, txt_file)
        with open(src_path, 'r') as f:
            lines = f.readlines()

        # 取得這個 frame 的 XML Trash 資訊
        xml_trash_boxes = trash_info.get(frame_idx, [])

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])

            # 只保留 Trash (class 2)，移除 Person (0) 和 Vehicle (1)
            if cls_id != 2:
                total_removed += 1
                continue

            # 取得 YOLO 座標
            yolo_cx = float(parts[1])
            yolo_cy = float(parts[2])
            yolo_w = float(parts[3])
            yolo_h = float(parts[4])
            yolo_coords = (yolo_cx, yolo_cy, yolo_w, yolo_h)

            # 用座標比對 XML，找出這個 box 的 State
            matched_state = None
            for xml_cx, xml_cy, xml_w, xml_h, state in xml_trash_boxes:
                if coords_match(yolo_coords, (xml_cx, xml_cy, xml_w, xml_h)):
                    matched_state = state
                    break

            # 根據 State 決定新 class_id
            if matched_state == 'Flying':
                new_cls_id = 1  # Trash_Flying
                total_flying += 1
            elif matched_state is not None:
                new_cls_id = 0  # Trash (None)
                total_trash += 1
            else:
                # 配對不到就預設 Trash (None)
                new_cls_id = 0
                total_trash += 1
                unmatched += 1

            new_lines.append(f"{new_cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")

        # 寫入新的標註檔（即使是空的也要寫，代表這個 frame 沒有 Trash）
        dst_path = os.path.join(dst_dir, txt_file)
        with open(dst_path, 'w') as f:
            f.writelines(new_lines)

    return total_trash, total_flying, total_removed, unmatched


def process_single(xml_path: str, txt_root: str, dst_dir: str):
    """
    單一資料集處理流程：
    - 直接讀指定 xml
    - 從 txt_root 遞迴抓所有 *.txt
    - 對每個 txt 依檔名抽 frame_idx，並寫到 dst_dir（檔名保留原 txt 檔名）
    """
    os.makedirs(dst_dir, exist_ok=True)

    trash_info = parse_xml_trash_states(xml_path)
    txt_paths = sorted(Path(txt_root).rglob('*.txt'))

    total_trash = 0
    total_flying = 0
    total_removed = 0
    unmatched = 0

    if not txt_paths:
        print(f"  ⚠️ 在 txt_root 找不到 .txt：{txt_root}")
        return total_trash, total_flying, total_removed, unmatched

    for txt_path in txt_paths:
        frame_idx = extract_frame_idx_from_txt_filename(txt_path.name)
        if frame_idx is None:
            continue

        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        xml_trash_boxes = trash_info.get(frame_idx, [])
        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id != 2:
                total_removed += 1
                continue

            yolo_cx = float(parts[1])
            yolo_cy = float(parts[2])
            yolo_w = float(parts[3])
            yolo_h = float(parts[4])
            yolo_coords = (yolo_cx, yolo_cy, yolo_w, yolo_h)

            matched_state = None
            for xml_cx, xml_cy, xml_w, xml_h, state in xml_trash_boxes:
                if coords_match(yolo_coords, (xml_cx, xml_cy, xml_w, xml_h)):
                    matched_state = state
                    break

            if matched_state == 'Flying':
                new_cls_id = 1
                total_flying += 1
            elif matched_state is not None:
                new_cls_id = 0
                total_trash += 1
            else:
                new_cls_id = 0
                total_trash += 1
                unmatched += 1

            new_lines.append(f"{new_cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")

        dst_path = os.path.join(dst_dir, txt_path.name)
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return total_trash, total_flying, total_removed, unmatched


def main():
    print("=" * 80)
    print("🗑️  拆分 Trash 標註：Trash (0) / Trash_Flying (1)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 掃描所有 XML（直接在 BASE_DIR 底下）
    xml_paths = sorted(Path(XML_DIR).glob('*.xml'))

    print(f"\n找到 {len(xml_paths)} 個 XML 檔\n")
    print(f"{'影片':<50} {'Trash':>8} {'Flying':>8} {'移除':>8} {'未配對':>8}")
    print("-" * 90)

    grand_trash = 0
    grand_flying = 0
    grand_removed = 0
    grand_unmatched = 0

    for xml_path in xml_paths:
        xml_stem = xml_path.stem
        txt_root = os.path.join(BASE_DIR, xml_stem + TXT_DIR_SUFFIX)
        dst_dir = os.path.join(OUTPUT_DIR, xml_stem)

        if not os.path.exists(txt_root):
            print(f"  ⚠️ 找不到 txt 根目錄：{txt_root}（略過 {xml_path.name}）")
            continue

        trash, flying, removed, unmatched = process_single(str(xml_path), txt_root, dst_dir)
        grand_trash += trash
        grand_flying += flying
        grand_removed += removed
        grand_unmatched += unmatched

        print(f"{xml_stem:<50} {trash:>8} {flying:>8} {removed:>8} {unmatched:>8}")

    print("-" * 90)
    print(f"{'總計':<50} {grand_trash:>8} {grand_flying:>8} {grand_removed:>8} {grand_unmatched:>8}")

    print(f"\n✅ 完成！輸出位置: {OUTPUT_DIR}")
    print(f"\n新類別定義:")
    print(f"  0 = Trash (一般垃圾)")
    print(f"  1 = Trash_Flying (飛行中的垃圾)")

    if grand_unmatched > 0:
        print(f"\n⚠️ 有 {grand_unmatched} 個 Trash box 在 XML 中找不到配對，已預設為 class 0 (Trash)")


if __name__ == '__main__':
    main()
