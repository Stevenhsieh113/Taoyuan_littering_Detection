import os
from pathlib import Path
import pandas as pd

# ==========================================
#               【設定區塊】
# ==========================================
# 1. 你指定的 8 labels 標籤根目錄
DATASET_LABELS_DIR = r"C:\桃園市政府\已標註和轉yolo\labels_8class"

# 2. 你指定的輸出 CSV 檔名 (自動加上 .csv)
OUTPUT_CSV = "ground_truth_frome_8_class_gt.csv"

# 3. 圖片副檔名 (請確認你的圖片是 .png 還是 .jpg)
IMAGE_EXT = ".png" 

# 4. 定義構成違規事實的標籤 ID
TARGET_LABELS = {
    '2': 'person_littering',
    '5': 'vehicle_littering',
    '7': 'trash_flying'
}
# ==========================================

print(f"🔍 開始掃描標籤目錄: {DATASET_LABELS_DIR} ...")

records = []
root_path = Path(DATASET_LABELS_DIR)

# 使用 rglob("*.txt") 進行遞迴搜尋，無論資料夾有多深都能抓到
for txt_file in root_path.rglob("*.txt"):
    
    # 略過 YOLO 常見的類別定義檔
    if txt_file.name == "classes.txt":
        continue

    # 取得檔案所在的原始路徑 (例如: ...\labels_8class\路口A)
    raw_folder_path = str(txt_file.parent)
    
    # 💡 關鍵修改：直接把字串中的 'labels_8class' 替換成你的影像資料夾名稱
    # 請將 'images_8class' 換成你實際放圖片的資料夾名稱
    folder_path = raw_folder_path.replace('labels_8class', 'images_8class')

    # 如果你的影像跟標籤根本就混在同一個資料夾裡，那就直接用原本的路徑：
    # folder_path = raw_folder_path

    # 將 .txt 轉換為對應的圖片檔名
    image_filename = txt_file.stem + IMAGE_EXT

    detected_behaviors = []
    has_violation = False

    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = parts[0]
                if class_id in TARGET_LABELS:
                    has_violation = True
                    detected_behaviors.append(TARGET_LABELS[class_id])
                    
    except Exception as e:
        print(f"❌ 讀取錯誤 {txt_file.name}: {e}")
        continue

    # 如果該 Frame 包含任何違規事實，則記錄下來
    if has_violation:
        # 去重複並合併行為名稱
        behavior_label = " | ".join(sorted(list(set(detected_behaviors))))
        
        records.append({
            '資料夾路徑': folder_path,
            '圖片檔名': image_filename,
            '真實標籤': behavior_label
        })

# ==========================================
# 儲存與輸出
# ==========================================
if records:
    df = pd.DataFrame(records)
    # 按照資料夾與檔名排序，方便後續比對
    df = df.sort_values(by=['資料夾路徑', '圖片檔名'])
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✨ 處理完畢！")
    print(f"📊 總共發現 {len(records)} 個違規 Frame。")
    print(f"📁 檔案已存為: {OUTPUT_CSV}")
else:
    print("\n⚠️ 掃描完成，但在指定路徑中沒找到任何 ID 為 2, 5, 7 的標籤。")
    print("請檢查 .txt 檔內容或 TARGET_LABELS 設定是否正確。")