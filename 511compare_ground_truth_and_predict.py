import pandas as pd
import re
from pathlib import Path

# ==========================================
#               【設定區塊】
# ==========================================
file_pred = "suspected_littering_with_tracker.csv"
file_gt = "deal_with_GT_litteringState.csv"   # 精選過的ground truth    
TOLERANCE_FRAMES = 40
# ==========================================

print("📊 啟動終極版模型效能評估 (免疫路徑 Bug)...\n")

def extract_frame_num(filename):
    match = re.search(r'(\d+)', str(filename))
    return int(match.group(1)) if match else -1

# 1. 讀取資料
try:
    df_pred = pd.read_csv(file_pred)
    df_gt = pd.read_csv(file_gt)
except FileNotFoundError as e:
    print(f"❌ 找不到檔案: {e}")
    exit()

# 2. 🌟 終極殺手鐧：提取「純影片名稱」與「Frame 數字」
# 不管前面路徑多長，Path(x).name 只會留下最後的資料夾名 (例如: '0619-RX' 或 '007-TZ')
df_pred['Video_Name'] = df_pred['資料夾路徑'].apply(lambda x: Path(x).name)
df_gt['Video_Name'] = df_gt['資料夾路徑'].apply(lambda x: Path(x).name)

df_pred['Frame_Num'] = df_pred['圖片檔名'].apply(extract_frame_num)
df_gt['Frame_Num'] = df_gt['圖片檔名'].apply(extract_frame_num)

# 3. 去重複邏輯 (改用 Video_Name)
df_pred_unique = df_pred.drop_duplicates(subset=['Video_Name', '垃圾追蹤ID'], keep='first')

gt_events = []
for video_name, group in df_gt.groupby('Video_Name'):
    frames = sorted(group['Frame_Num'].tolist())
    if not frames: continue
    
    current_event_start = frames[0]
    for f in frames[1:]:
        if f - current_event_start > TOLERANCE_FRAMES * 2:
            gt_events.append({'Video_Name': video_name, 'Frame_Num': current_event_start})
            current_event_start = f
    gt_events.append({'Video_Name': video_name, 'Frame_Num': current_event_start})

df_gt_unique = pd.DataFrame(gt_events)

# 4. 開始比對算分數
TP = 0
FP = 0
matched_gt_indices = set()

for _, pred in df_pred_unique.iterrows():
    pred_video = pred['Video_Name']
    pred_frame = pred['Frame_Num']
    
    possible_gts = df_gt_unique[df_gt_unique['Video_Name'] == pred_video]
    
    is_hit = False
    for idx, gt in possible_gts.iterrows():
        # 🌟 【關鍵修正】：檢查這個 GT 是不是已經被先前的預測配對過了？
        if idx in matched_gt_indices:
            continue # 如果被配對過了，就跳過，不能重複認領
            
        if abs(pred_frame - gt['Frame_Num']) <= TOLERANCE_FRAMES:
            is_hit = True
            matched_gt_indices.add(idx) # 🌟 消耗掉這個 GT
            break
            
    if is_hit:
        TP += 1
    else:
        FP += 1

FN = len(df_gt_unique) - len(matched_gt_indices)

# 5. 計算最終指標
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# ==========================================
# 輸出報告
# ==========================================
print("=====================================")
print("🏆 系統驗證報告 (2D Tracker Baseline)")
print("=====================================")
print(f"✅ 成功抓取 (True Positive, TP):  {TP} 件")
print(f"❌ 系統誤判 (False Positive, FP): {FP} 件 (假警報)")
print(f"⚠️ 系統漏抓 (False Negative, FN): {FN} 件")
print("-" * 37)
print(f"🎯 準確率 (Precision): {precision:.2%}")
print(f"🎣 召回率 (Recall):    {recall:.2%}")
print(f"⭐ F1-Score:           {f1_score:.2%}")
print("=====================================")