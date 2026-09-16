from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# ==========================================
# 【⚙️ 系統設定區塊】
# ==========================================
# 🧠 1. 垃圾專用模型權重（請把 best.pt 放到此路徑，或改成你的實際路徑）
MODEL_WEIGHTS = r"C:\桃園市政府\已標註和轉yolo\weights\yolo11l_trash_best.pt"

# 📂 2. 測試資料路徑（專案 image 資料夾）
INPUT_DIR = r"C:\桃園市政府\已標註和轉yolo\image"

# 💾 3. 輸出：檔名對齊 final_check_new_correct.py 的 FILE_TRASH
OUTPUT_CSV = r"C:\桃園市政府\已標註和轉yolo\mock_trash_results.csv"
CONF_THRESH = 0.25
# 單類別垃圾模型通常輸出 0；final_check 會再 +100 → 100
TRASH_CLASS_ID_FOR_RELATION = 0

# ==========================================

print("啟動真・YOLO 垃圾萃取器...")
print(f"載入模型: {MODEL_WEIGHTS}")

try:
    model = YOLO(MODEL_WEIGHTS)
except Exception as e:
    print(f"模型載入失敗: {e}")
    exit()

image_paths = list(Path(INPUT_DIR).rglob("*.png")) + list(Path(INPUT_DIR).rglob("*.jpg"))
print(f"總共找到 {len(image_paths)} 張圖片，準備開始推論...")

results_list = []

for count, img_path in enumerate(image_paths, 1):
    if count % 100 == 0:
        print(f"進度: {count} / {len(image_paths)}")

    results = model.predict(source=str(img_path), conf=CONF_THRESH, verbose=False)

    folder_path = str(img_path.parent)
    file_name = img_path.name

    for box in results[0].boxes:
        x1, y1, x2, y2 = [float(val) for val in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cls_id = TRASH_CLASS_ID_FOR_RELATION

        results_list.append(
            {
                "資料夾路徑": folder_path,
                "圖片檔名": file_name,
                "類別ID": cls_id,
                "信心度": round(conf, 4),
                "左上X": round(x1, 2),
                "左上Y": round(y1, 2),
                "右下X": round(x2, 2),
                "右下Y": round(y2, 2),
            }
        )

if results_list:
    df = pd.DataFrame(results_list)
    desired_columns = [
        "資料夾路徑",
        "圖片檔名",
        "類別ID",
        "信心度",
        "左上X",
        "左上Y",
        "右下X",
        "右下Y",
    ]
    df = df[desired_columns]

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n推論完成！總共抓到了 {len(df)} 個垃圾特徵。")
    print(f"檔案已匯出: {OUTPUT_CSV}")
    print("下一步：直接跑 final_check_new_correct.py（FILE_TRASH 已對應 mock_trash_results.csv）")
else:
    print("\n推論完成。在這個信心度門檻下，沒有抓到任何垃圾。")
