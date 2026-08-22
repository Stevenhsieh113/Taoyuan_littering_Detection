import csv
from pathlib import Path
from PIL import Image

# ==========================================
#               【設定區塊】
# ==========================================
images_folder = r"C:\桃園市政府\已標註和轉yolo\image"  # 你的原始圖片資料夾
labels_folder = r"C:\桃園市政府\已標註和轉yolo\labels_split"  # 你的 YOLO txt 資料夾
output_csv = "mock_trash_results.csv" # 輸出的模擬垃圾 CSV 檔名
# ==========================================

# 準備寫入 CSV
with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as csv_file:
    writer = csv.writer(csv_file)
    # 標題與你的「人車 CSV」保持完全一致
    writer.writerow(["資料夾路徑", "圖片檔名", "類別ID", "信心度", "左上X", "左上Y", "右下X", "右下Y"])

    # 讀取所有的 txt 檔案
    txt_files = list(Path(labels_folder).rglob("*.txt"))
    print(f"📂 找到 {len(txt_files)} 個垃圾標註檔，準備轉換...\n")

    for txt_path in txt_files:
        # 假設圖片副檔名是 .png (請根據你的實際情況改成 .jpg 或其他)
        # 1. 算出這個 txt 檔在 labels 裡面的「相對路徑」
        relative_path = txt_path.relative_to(labels_folder)
        img_path = Path(images_folder) / relative_path.with_suffix(".png")
        
        # 確保對應的圖片存在，我們需要它的尺寸來還原座標
        if not img_path.exists():
            print(f"⚠️ 找不到 {relative_path}，跳過 {txt_path.name}")
            continue
            
        # 取得圖片尺寸
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # 讀取 txt 檔裡面的座標
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue # 略過格式錯誤的行
                
                # YOLO txt 格式: class_id, center_x, center_y, width, height (皆為 0~1 的比例)
                cls_id = int(parts[0])
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])
                
                # 1. 轉換為真實像素的「中心點與寬高」
                cx_pixel = cx_norm * img_width
                cy_pixel = cy_norm * img_height
                w_pixel = w_norm * img_width
                h_pixel = h_norm * img_height
                
                # 2. 推算出左上角 (x1, y1) 與 右下角 (x2, y2)
                x1 = cx_pixel - (w_pixel / 2)
                y1 = cy_pixel - (h_pixel / 2)
                x2 = cx_pixel + (w_pixel / 2)
                y2 = cy_pixel + (h_pixel / 2)
                
                # 信心度部分：因為是人工標註的絕對正確資料，我們模擬信心度為 1.0 (100%)
                confidence = 1.0 
                
                # 寫入 CSV，小數點取到整數或小數後兩位皆可
                writer.writerow([
                    str(img_path.parent),
                    img_path.name,
                    cls_id,
                    confidence,
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                ])
                
        print(f"✅ 完成轉換: {txt_path.name}")

print(f"\n🎉 轉換完畢！模擬垃圾資料已存入：{output_csv}")