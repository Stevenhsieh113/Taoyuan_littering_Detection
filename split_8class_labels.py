#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 YOLO 標註 (3 class) 展開成 8 class，參考 XML 的詳細屬性
- 以 YOLO txt 為主體（座標精準）
- 用 XML 的 Action/State 屬性判斷子類別

新類別：
  0 = person
  1 = person_holding
  2 = person_littering
  3 = vehicle
  4 = vehicle_holding
  5 = vehicle_littering
  6 = trash
  7 = trash_flying
"""

import xml.etree.ElementTree as ET
import os
from collections import Counter
import re
from pathlib import Path

# ===================== 設定 =====================
BASE_DIR = 'C:\桃園市政府\已標註和轉yolo'
XML_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'labels_8class')

# ===================== 資料夾/檔名規則 =====================
# xml：<stem>.xml
# txt：<stem>_txt（遞迴找底下所有 *.txt）
TXT_DIR_SUFFIX = '_txt'
# ============================================================

# XML 檔名 → YOLO 資料夾名稱的對照
NAME_MAP = {
    '桃園影片 2025年11月17日 1280x720': '桃園影片_2025_11月17日_1280x720',
}

IMG_W, IMG_H = 1280, 720

# 原始 YOLO 的 class 對照
ORIG_CLASSES = {0: 'Person', 1: 'Vehicle', 2: 'Trash'}

# 新的 8 class 定義
NEW_CLASSES = {
    'person':             0,
    'person_holding':     1,
    'person_littering':   2,
    'vehicle':            3,
    'vehicle_holding':    4,
    'vehicle_littering':  5,
    'trash':              6,
    'trash_flying':       7,
}

# 每個原始 label 對應的屬性名稱
ATTR_NAME_MAP = {
    'Person':  'Action',   # None / Holding / Littering
    'Vehicle': 'Action',   # None / Holding / Littering
    'Trash':   'State',    # None / Flying
}
# ================================================


def xml_box_to_yolo(xtl, ytl, xbr, ybr):
    cx = (xtl + xbr) / 2 / IMG_W
    cy = (ytl + ybr) / 2 / IMG_H
    w = (xbr - xtl) / IMG_W
    h = (ybr - ytl) / IMG_H
    return cx, cy, w, h


def coords_match(yolo_coords, xml_coords, tol=1e-4):
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


def get_new_class_id(base_label, attr_value):
    """根據 base_label 和屬性值決定新的 class_id"""
    label_lower = base_label.lower()

    if attr_value and attr_value.lower() not in ('none', ''):
        combo = f"{label_lower}_{attr_value.lower()}"
    else:
        combo = label_lower

    return NEW_CLASSES.get(combo, None), combo


def parse_xml_all_boxes(xml_path):
    """
    解析 XML，回傳每個 frame 中所有 box 的資訊
    回傳: { frame_idx: [ (base_label, cx, cy, w, h, attr_value), ... ] }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame_data = {}

    for track in root.findall('.//track'):
        base_label = track.attrib['label']
        attr_name = ATTR_NAME_MAP.get(base_label)

        for box in track.findall('box'):
            if box.get('outside') == '1':
                continue

            frame_idx = int(box.attrib['frame'])
            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])

            # 取得屬性值
            attr_value = 'None'
            if attr_name:
                for attr in box.findall('attribute'):
                    if attr.attrib.get('name') == attr_name:
                        attr_value = attr.text
                        break

            cx, cy, w, h = xml_box_to_yolo(xtl, ytl, xbr, ybr)

            if frame_idx not in frame_data:
                frame_data[frame_idx] = []
            frame_data[frame_idx].append((base_label, cx, cy, w, h, attr_value))

    return frame_data


def process_video(xml_name, folder_name):
    xml_path = os.path.join(XML_DIR, xml_name + '.xml')
    src_dir = os.path.join(BASE_DIR, folder_name, 'obj_train_data')
    dst_dir = os.path.join(OUTPUT_DIR, folder_name)

    if not os.path.exists(xml_path):
        print(f"  ⚠️ XML 不存在: {xml_path}")
        return Counter(), 0

    if not os.path.exists(src_dir):
        print(f"  ⚠️ 資料夾不存在: {src_dir}")
        return Counter(), 0

    os.makedirs(dst_dir, exist_ok=True)

    frame_data = parse_xml_all_boxes(xml_path)
    txt_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.txt')])

    class_counter = Counter()
    unmatched = 0

    for txt_file in txt_files:
        frame_idx = int(txt_file.replace('frame_', '').replace('.txt', ''))

        with open(os.path.join(src_dir, txt_file)) as f:
            lines = f.readlines()

        xml_boxes = frame_data.get(frame_idx, [])
        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            orig_cls = int(parts[0])
            yolo_coords = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))

            base_label = ORIG_CLASSES.get(orig_cls)
            if base_label is None:
                continue

            # 在 XML 中找配對的 box
            matched_attr = None
            for xml_label, xml_cx, xml_cy, xml_w, xml_h, xml_attr in xml_boxes:
                if xml_label == base_label and coords_match(yolo_coords, (xml_cx, xml_cy, xml_w, xml_h)):
                    matched_attr = xml_attr
                    break

            if matched_attr is not None:
                new_cls_id, combo = get_new_class_id(base_label, matched_attr)
            else:
                # 配對不到就用基礎 label
                new_cls_id, combo = get_new_class_id(base_label, 'None')
                unmatched += 1

            if new_cls_id is not None:
                new_lines.append(f"{new_cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                class_counter[combo] += 1

        with open(os.path.join(dst_dir, txt_file), 'w') as f:
            f.writelines(new_lines)

    return class_counter, unmatched


def process_single(xml_path: str, txt_root: str, dst_dir: str):
    """
    單一資料集處理流程：
    - 直接讀指定 xml
    - 從 txt_root 目錄遞迴抓所有 *.txt（不依賴 labels/train 結構）
    - 對每個 txt 依檔名抽 frame_idx，並寫到 dst_dir（檔名保留原 txt 檔名）
    """
    os.makedirs(dst_dir, exist_ok=True)

    frame_data = parse_xml_all_boxes(xml_path)

    txt_paths = sorted(Path(txt_root).rglob('*.txt'))
    class_counter = Counter()
    unmatched = 0

    if not txt_paths:
        print(f"  ⚠️ 在 txt_root 找不到 .txt：{txt_root}")
        return class_counter, unmatched

    for txt_path in txt_paths:
        frame_idx = extract_frame_idx_from_txt_filename(txt_path.name)
        if frame_idx is None:
            continue

        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        xml_boxes = frame_data.get(frame_idx, [])
        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            orig_cls = int(parts[0])
            yolo_coords = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))

            base_label = ORIG_CLASSES.get(orig_cls)
            if base_label is None:
                continue

            # 在 XML 中找配對的 box
            matched_attr = None
            for xml_label, xml_cx, xml_cy, xml_w, xml_h, xml_attr in xml_boxes:
                if xml_label == base_label and coords_match(yolo_coords, (xml_cx, xml_cy, xml_w, xml_h)):
                    matched_attr = xml_attr
                    break

            if matched_attr is not None:
                new_cls_id, combo = get_new_class_id(base_label, matched_attr)
            else:
                # 配對不到就用基礎 label
                new_cls_id, combo = get_new_class_id(base_label, 'None')
                unmatched += 1

            if new_cls_id is not None:
                new_lines.append(f"{new_cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                class_counter[combo] += 1

        dst_path = os.path.join(dst_dir, txt_path.name)
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return class_counter, unmatched


def main():
    print("=" * 80)
    print("🏷️  展開 YOLO 標註：3 class → 8 class")
    print("=" * 80)
    print("\n新類別定義:")
    for name, cid in sorted(NEW_CLASSES.items(), key=lambda x: x[1]):
        print(f"  {cid} = {name}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    xml_paths = sorted(Path(XML_DIR).glob('*.xml'))
    print(f"\n找到 {len(xml_paths)} 個 XML 檔\n")

    grand_counter = Counter()
    grand_unmatched = 0

    # 表頭
    short_names = list(NEW_CLASSES.keys())
    header = f"{'影片':<42}"
    for sn in short_names:
        header += f" {sn:>8}"
    header += f" {'未配對':>8}"
    print(header)
    print("-" * (42 + 9 * len(short_names) + 9))

    for xml_path in xml_paths:
        xml_stem = xml_path.stem
        txt_root = os.path.join(BASE_DIR, xml_stem + TXT_DIR_SUFFIX)
        dst_dir = os.path.join(OUTPUT_DIR, xml_stem)

        if not os.path.exists(txt_root):
            print(f"  ⚠️ 找不到 txt 根目錄：{txt_root}（略過 {xml_path.name}）")
            continue

        counter, unmatched = process_single(str(xml_path), txt_root, dst_dir)
        grand_counter += counter
        grand_unmatched += unmatched

        row = f"{xml_stem:<42}"
        for sn in short_names:
            row += f" {counter.get(sn, 0):>8}"
        row += f" {unmatched:>8}"
        print(row)

    print("-" * (42 + 9 * len(short_names) + 9))
    row = f"{'總計':<42}"
    for sn in short_names:
        row += f" {grand_counter.get(sn, 0):>8}"
    row += f" {grand_unmatched:>8}"
    print(row)

    print(f"\n✅ 完成！輸出位置: {OUTPUT_DIR}")
    if grand_unmatched > 0:
        print(f"⚠️ 有 {grand_unmatched} 個 box 在 XML 中找不到配對，已用基礎 label")


if __name__ == '__main__':
    main()
