from ultralytics import YOLO
import ultralytics
#檢查硬體和版本
ultralytics.checks()
import torch

# 推論裝置：本機 PyTorch 若為 CPU 版，請使用 "cpu"
DEVICE = "cpu"
print(f"推論裝置: {DEVICE}（torch.cuda.is_available() = {torch.cuda.is_available()}）")

import csv
from pathlib import Path
from ultralytics import YOLO

# 載入模型 (它會自動找到你剛剛下載的 yolo11n.pt)
model = YOLO("yolo11n.pt")
model.to(DEVICE)

# 確認 YOLO 使用的硬體設備
print(f"YOLO 目前使用的硬體是：{model.device}")

#是否要儲存圖片
SAVE_DRAWN_IMAGES = False

# input_folder = r"C:\Users\User\Desktop\專題dataset\dataset\images"
input_folder = r"C:\桃園市政府\已標註和轉yolo\image"
output_csv = 'detect_results.csv'   # 統整結果會存成這個檔案
output_image_folder = r"C:\Users\User\Desktop\專題dataset\畫框測試輸出"

valid_extensions = ['.jpg','.png','jpeg','.bmp']

if SAVE_DRAWN_IMAGES:
    Path(output_image_folder).mkdir(parents=True, exist_ok=True)
    print(f"📸 存圖模式已開啟！畫好框的圖片將存入: {output_image_folder}\n")

#建立並打開 CSV 檔案
with open(output_csv, mode='w', newline='',encoding='utf-8-sig') as csv_file:
    writer = csv.writer(csv_file)
    # 寫入 CSV 的第一列標題 (Header)
    writer.writerow(["資料夾路徑", "圖片檔名", "類別ID", "信心度", "左上X", "左上Y", "右下X", "右下Y"])
    #找出資料夾中所有符合副檔名的檔案
    folder_path = Path(input_folder)
    all_files = [f for f in folder_path.rglob('*') if f.suffix.lower() in valid_extensions]
    print(f"📂 總共在層層資料夾中找到 {len(all_files)} 張圖片，準備開始處理...\n")


    for file_path in all_files:
        try:
            #執行預測,(只抓人 0, 汽車 2, 機車 3, 公車 5, 卡車 7)
            results = model.predict(source=file_path, classes=[0, 2, 3, 5, 7], device=DEVICE, verbose=False)
            boxes = results[0].boxes

            if len(boxes)>0:
                for box in boxes:
                    cls_id = box.cls[0].item()
                    conf = box.conf[0].item()
                    x1,y1,x2,y2 = box.xyxy[0].tolist()

                    #寫入csv
                    writer.writerow([
                        str(file_path.parent),
                        file_path.name,         # 圖片本身的檔名
                        cls_id, 
                        round(conf, 3), 
                        round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                    ])

                    # 只有當總開關設為 True 時，才會執行存圖動作
                    if SAVE_DRAWN_IMAGES:
                        # 設定存檔路徑 (輸出資料夾 + 原本的圖片檔名)
                        save_path = Path(output_image_folder) / file_path.name
                        # results[0].save() 是 YOLO 內建超好用的函數，會直接把畫好框的圖存下來
                        results[0].save(filename=str(save_path))

                    print(f"✅ 完成: {file_path.name} (發現 {len(boxes)} 個目標)")

            else:
                print(f"➖ 略過: {file_path.name} (沒有抓到設定的目標)")


        except Exception as e:
            # 如果某張圖片損毀，程式會印出錯誤但繼續處理下一張，不會整個崩潰
            print(f"❌ 錯誤: 處理 {file_path.name} 時發生問題 ({e})")


print(f"\n🎉 批次處理完畢！所有結果已存入：{output_csv}")