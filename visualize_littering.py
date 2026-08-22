import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
# 這是一支用於視覺化「疑似亂丟垃圾」行為的輔助程式。
# 在你的亂丟垃圾偵測系統中，這支程式扮演著「將模型（例如 YOLO）預測的結果，重新畫回原始圖片上」的角色。
# 它能讓你直接用肉眼檢查系統抓取的人車位置、垃圾位置，以及它們之間的距離判斷是否合理，
# 是非常實用的 Debug 與展示工具。
# ==========================================
#               【設定區塊】
# ==========================================
csv_file_path = "suspected_littering_with_tracker.csv" 
output_folder_path = r"C:\桃園市政府\已標註和轉yolo\littering_visualization_output_with_tracker\new"

# 框的顏色與設定
color_pc = (0, 255, 0)      # 綠色 (人車)
color_trash = (0, 0, 255)   # 紅色 (垃圾)
color_line = (255, 255, 0)  # 青色 (距離連線)
thickness = 2
# ==========================================

def load_pil_to_cv2(path):
    """支援中文路徑的圖片讀取函數"""
    try:
        pil_img = Image.open(path)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except:
        return None

def main():
    df = pd.read_csv(csv_file_path)
    out_path = Path(output_folder_path)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 開始處理 {len(df)} 筆視覺化紀錄...")

    for i, row in df.iterrows():
        img_full_path = Path(row['資料夾路徑']) / row['圖片檔名']
        img = load_pil_to_cv2(img_full_path)
        
        if img is None:
            continue

        # 1. 提取精確座標並轉為整數
        # 人車框
        x1_pc, y1_pc = int(row['左上X_人車']), int(row['左上Y_人車'])
        x2_pc, y2_pc = int(row['右下X_人車']), int(row['右下Y_人車'])
        
        # 垃圾框
        x1_tr, y1_tr = int(row['左上X_垃圾']), int(row['左上Y_垃圾'])
        x2_tr, y2_tr = int(row['右下X_垃圾']), int(row['右下Y_垃圾'])

        # 2. 繪製矩陣框
        cv2.rectangle(img, (x1_pc, y1_pc), (x2_pc, y2_pc), color_pc, thickness)
        cv2.rectangle(img, (x1_tr, y1_tr), (x2_tr, y2_tr), color_trash, thickness)

        # 3. 繪製中心點連線
        c_pc = (int(row['中心X_人車']), int(row['中心Y_人車']))
        c_tr = (int(row['中心X_垃圾']), int(row['中心Y_垃圾']))
        cv2.line(img, c_pc, c_tr, color_line, 1)

        # 4. 標註類別與距離資訊
        info = f"Dist: {row['距離']:.1f}px"
        cv2.putText(img, info, (c_pc[0], c_pc[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_line, 2)

        # 5. 儲存結果 (同樣支援中文路徑儲存)
        # 提取資料夾名稱 (例如從路徑中提取出 'KEK-3232')
        folder_name = Path(row['資料夾路徑']).name 
        
        # 將資料夾名稱與圖片檔名結合
        save_file = out_path / f"{folder_name}_{row['圖片檔名']}"
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(save_file)
        
        if i % 10 == 0:
            print(f"已完成: {i}/{len(df)}")

    print(f"\n✅ 視覺化完成！請至 {output_folder_path} 查看結果。")

if __name__ == "__main__":
    main()