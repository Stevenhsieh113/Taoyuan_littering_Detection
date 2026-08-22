import csv
import math
from pathlib import Path
from datetime import datetime
from PIL import Image


# 標註資料夾（可依需求修改）
LABELS_ROOT = Path(r"C:\桃園市政府\已標註和轉yolo\labels_8class")
IMAGES_ROOT = Path(r"C:\桃園市政府\已標註和轉yolo\image")
# 輸出檔名
OUTPUT_CSV = Path("label_distance_results.csv")
VALID_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_yolo_line(line: str):
    """
    解析 YOLO 標註行：class_id center_x center_y width height
    只會用到 class_id, center_x, center_y。
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    try:
        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
    except ValueError:
        return None

    return class_id, center_x, center_y


def main():
    txt_files = sorted(LABELS_ROOT.rglob("*.txt"))
    if not txt_files:
        print(f"找不到任何標註檔：{LABELS_ROOT}")
        return

    rows = []
    skipped_no_image = 0

    for txt_path in txt_files:
        objects = []

        with txt_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                parsed = parse_yolo_line(raw_line)
                if parsed is None:
                    continue
                class_id, cx, cy = parsed
                objects.append({"class_id": class_id, "cx": cx, "cy": cy})

        if not objects:
            continue

        # 找對應圖片：先用 labels 相對路徑對應到 image，再嘗試常見副檔名
        relative_txt = txt_path.relative_to(LABELS_ROOT)
        image_base = IMAGES_ROOT / relative_txt.with_suffix("")
        image_path = None
        for ext in VALID_IMAGE_SUFFIXES:
            candidate = image_base.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            skipped_no_image += 1
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        # 條件：同時有 0~5 任一類，且有 6 或 7
        group_a = [obj for obj in objects if 0 <= obj["class_id"] <= 5]
        group_b = [obj for obj in objects if obj["class_id"] in (6, 7)]
        if not group_a or not group_b:
            continue

        # 檔案中是否包含 2、5、7 任一類別
        has_257 = int(any(obj["class_id"] in (2, 5, 7) for obj in objects))

        relative_folder = txt_path.parent.relative_to(LABELS_ROOT)
        folder_name = str(relative_folder) if str(relative_folder) != "." else txt_path.parent.name

        # 計算任一(0~5)物體與任一(6/7)物體中心點距離
        for obj_a in group_a:
            for obj_b in group_b:
                cls_small, cls_large = sorted([obj_a["class_id"], obj_b["class_id"]])
                ax, ay = obj_a["cx"] * img_w, obj_a["cy"] * img_h
                bx, by = obj_b["cx"] * img_w, obj_b["cy"] * img_h
                distance = math.dist((ax, ay), (bx, by))

                rows.append(
                    {
                        "資料夾名稱": folder_name,
                        "檔名": txt_path.name,
                        "label(小)": cls_small,
                        "label(大)": cls_large,
                        "有沒有數字為2,5,7(有=1,沒有=0)": has_257,
                        "計算到的物體距離": round(distance, 6),
                    }
                )

    fieldnames = [
        "資料夾名稱",
        "檔名",
        "label(小)",
        "label(大)",
        "有沒有數字為2,5,7(有=1,沒有=0)",
        "計算到的物體距離",
    ]

    saved_path = OUTPUT_CSV
    try:
        with saved_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = OUTPUT_CSV.with_name(f"{OUTPUT_CSV.stem}_{timestamp}{OUTPUT_CSV.suffix}")
        with saved_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"⚠️ 原輸出檔可能被占用，已改存為：{saved_path.name}")

    print(f"完成，共輸出 {len(rows)} 筆距離資料 -> {saved_path.resolve()}")
    if skipped_no_image:
        print(f"另有 {skipped_no_image} 個標註檔找不到對應圖片，已略過。")


if __name__ == "__main__":
    main()
