import pandas as pd
import math
import re
from pathlib import Path
# 找到某個被丟出垃圾的第一個 frame 並給予一個 id
# ==========================================
#               【⚙️ 系統設定區塊】
# ==========================================
# 📂 雙檔案路徑設定
FILE_OBJECTS = "detect_results.csv"          # 人車 CSV
FILE_TRASH = "mock_trash_results.csv"        # 垃圾 CSV
OUTPUT_CSV = "suspected_littering_with_tracker.csv"  # 產出的違規名單（完整欄位）
OUTPUT_FRAME_INDEX_CSV = "suspected_littering_frame_index.csv"  # 疑似違規對應：檔名與 frame 編號（精簡）

# 📏 演算法核心參數設定
THRESHOLD_PIXELS = 448.166   # [距離門檻] 人車與垃圾的中心點距離小於此值，觸發警報
PATIENCE_FRAMES = 15         # [緩衝壽命] 允許垃圾在畫面上消失的最大幀數 (對抗 YOLO 閃爍)
MERGE_RADIUS = 365           # [繼承半徑] 允許新垃圾繼承舊 ID 的最大空間範圍 (對抗彈跳)

# 📏 垃圾運動狀態（僅判斷垃圾框，不判斷人/車）
EPS_MOVE = 1                 # 垃圾中心點位移小於此值 (px) 視為「沒動」
N_STATIC = 10                # 連續幾幀沒動 → 標記為背景垃圾 (static)，不參與配對
N_CANDIDATE = 3              # 新垃圾至少觀察幾幀後，才允許升級為可配對 (active)

# 🌟 修正1：把 COCO 的人與所有車輛都包進來
CLASS_PERSON_CAR = [0, 1, 2, 3, 5, 7]

# 🌟 修正2：設定加上 100 偏移量後的垃圾 ID (100 代表原本的 0, 101 代表原本的 1)
CLASS_TRASH = [100, 101]
# ==========================================

print("啟動 2D 虛擬時空追蹤器 (含垃圾運動狀態過濾)...")


def extract_frame_num(filename):
    match = re.search(r"(\d+)", str(filename))
    return int(match.group(1)) if match else -1


def new_trash_track(frame_num, cx, cy, box, class_id):
    """建立新垃圾追蹤 ID 的初始狀態。"""
    return {
        "last_frame": frame_num,
        "cx": cx,
        "cy": cy,
        "box": box,
        "class_id": class_id,
        "still_count": 0,
        "ever_moved": False,
        "seen_frames": 1,
        "state": "candidate",
        "last_frame_dist": 0.0,
    }


def update_trash_motion(info, cx, cy):
    """
    依「垃圾中心點」跨幀位移更新狀態（不處理人/車）。
    candidate -> active（需 ever_moved 且觀察足夠幀）
    連續靜止 -> static（不參與配對）
    static 後若再移動 -> 回到 active
    """
    frame_dist = math.hypot(cx - info["cx"], cy - info["cy"])
    info["last_frame_dist"] = round(frame_dist, 2)

    if frame_dist >= EPS_MOVE:
        info["ever_moved"] = True
        info["still_count"] = 0
    else:
        info["still_count"] += 1

    if info["still_count"] >= N_STATIC:
        info["state"] = "static"
    elif info["state"] == "static" and frame_dist >= EPS_MOVE:
        info["state"] = "active"
    elif info["state"] == "candidate":
        if info["ever_moved"] and info["seen_frames"] >= N_CANDIDATE:
            info["state"] = "active"

    info["cx"] = cx
    info["cy"] = cy


# 1. 讀取並合併兩個 CSV 檔案
try:
    print("讀取人車資料中...")
    df_objects = pd.read_csv(FILE_OBJECTS)
    print("讀取垃圾資料中...")
    df_trash = pd.read_csv(FILE_TRASH)
    df_trash["類別ID"] = df_trash["類別ID"] + 100

    df_raw = pd.concat([df_objects, df_trash], ignore_index=True)
    print(f"成功合併！總共載入 {len(df_raw)} 筆偵測紀錄。")
except FileNotFoundError as e:
    print(f"找不到輸入檔案: {e}")
    exit()

print("自動計算 BBox 中心點座標...")
df_raw["中心X"] = (df_raw["左上X"] + df_raw["右下X"]) / 2
df_raw["中心Y"] = (df_raw["左上Y"] + df_raw["右下Y"]) / 2

df_raw["Video_Name"] = df_raw["資料夾路徑"].apply(lambda x: Path(x).name)
df_raw["Frame_Num"] = df_raw["圖片檔名"].apply(extract_frame_num)

suspected_records = []
suspected_frame_index_rows = []
skipped_static_pairing = 0

# ==========================================
# 🔄 核心邏輯：按影片逐一處理
# ==========================================
for video_name, video_group in df_raw.groupby("Video_Name"):

    active_trashes = {}
    dropped_trash_ids = set()
    next_trash_id = 1

    video_group = video_group.sort_values(by="Frame_Num")

    for frame_num, frame_group in video_group.groupby("Frame_Num"):

        active_trashes = {
            t_id: info
            for t_id, info in active_trashes.items()
            if frame_num - info["last_frame"] <= PATIENCE_FRAMES
        }

        people_cars = frame_group[frame_group["類別ID"].isin(CLASS_PERSON_CAR)]
        trashes = frame_group[frame_group["類別ID"].isin(CLASS_TRASH)]

        current_frame_trash_data = []

        for _, trash in trashes.iterrows():
            cx, cy = trash["中心X"], trash["中心Y"]
            box = (
                trash["左上X"],
                trash["左上Y"],
                trash["右下X"],
                trash["右下Y"],
            )
            class_id = trash["類別ID"]

            best_match_id = None
            min_dist = float("inf")

            for track_id, track_info in active_trashes.items():
                dist = math.hypot(cx - track_info["cx"], cy - track_info["cy"])
                if dist < MERGE_RADIUS and dist < min_dist:
                    min_dist = dist
                    best_match_id = track_id

            if best_match_id is not None:
                final_id = best_match_id
                info = active_trashes[final_id]
                info["last_frame"] = frame_num
                info["box"] = box
                info["class_id"] = class_id
                info["seen_frames"] += 1
                update_trash_motion(info, cx, cy)
            else:
                final_id = next_trash_id
                active_trashes[final_id] = new_trash_track(
                    frame_num, cx, cy, box, class_id
                )
                next_trash_id += 1

            info = active_trashes[final_id]
            if info["state"] == "active":
                current_frame_trash_data.append((final_id, info))
            else:
                skipped_static_pairing += 1

        if not people_cars.empty and current_frame_trash_data:
            folder_path = frame_group["資料夾路徑"].iloc[0]
            img_filename = frame_group["圖片檔名"].iloc[0]

            for _, pc in people_cars.iterrows():
                pc_cx, pc_cy = pc["中心X"], pc["中心Y"]

                for t_id, t_info in current_frame_trash_data:
                    if t_id in dropped_trash_ids:
                        continue

                    distance = math.hypot(pc_cx - t_info["cx"], pc_cy - t_info["cy"])

                    if distance <= THRESHOLD_PIXELS:
                        suspected_records.append(
                            {
                                "資料夾路徑": folder_path,
                                "圖片檔名": img_filename,
                                "類別ID_人車": pc["類別ID"],
                                "類別ID_垃圾": t_info["class_id"],
                                "距離": round(distance, 2),
                                "垃圾追蹤ID": t_id,
                                "垃圾狀態": t_info["state"],
                                "垃圾幀位移": t_info["last_frame_dist"],
                                "中心X_人車": pc_cx,
                                "中心Y_人車": pc_cy,
                                "左上X_人車": pc["左上X"],
                                "左上Y_人車": pc["左上Y"],
                                "右下X_人車": pc["右下X"],
                                "右下Y_人車": pc["右下Y"],
                                "中心X_垃圾": t_info["cx"],
                                "中心Y_垃圾": t_info["cy"],
                                "左上X_垃圾": t_info["box"][0],
                                "左上Y_垃圾": t_info["box"][1],
                                "右下X_垃圾": t_info["box"][2],
                                "右下Y_垃圾": t_info["box"][3],
                            }
                        )
                        suspected_frame_index_rows.append(
                            {
                                "資料夾路徑": folder_path,
                                "影片資料夾名": Path(folder_path).name,
                                "圖片檔名": str(img_filename).strip(),
                                "Frame_Num": int(frame_num),
                                "垃圾追蹤ID": int(t_id),
                                "距離": round(distance, 2),
                                "垃圾狀態": t_info["state"],
                            }
                        )
                        dropped_trash_ids.add(t_id)

# ==========================================
# 💾 匯出結果
# ==========================================
if suspected_records:
    df_output = pd.DataFrame(suspected_records)

    desired_columns = [
        "資料夾路徑",
        "圖片檔名",
        "類別ID_人車",
        "類別ID_垃圾",
        "距離",
        "垃圾追蹤ID",
        "垃圾狀態",
        "垃圾幀位移",
        "中心X_人車",
        "中心Y_人車",
        "左上X_人車",
        "左上Y_人車",
        "右下X_人車",
        "右下Y_人車",
        "中心X_垃圾",
        "中心Y_垃圾",
        "左上X_垃圾",
        "左上Y_垃圾",
        "右下X_垃圾",
        "右下Y_垃圾",
    ]
    df_output = df_output[desired_columns]

    df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    df_frames = pd.DataFrame(suspected_frame_index_rows)
    frame_cols = [
        "資料夾路徑",
        "影片資料夾名",
        "圖片檔名",
        "Frame_Num",
        "垃圾追蹤ID",
        "距離",
        "垃圾狀態",
    ]
    df_frames = df_frames[frame_cols]
    df_frames.to_csv(OUTPUT_FRAME_INDEX_CSV, index=False, encoding="utf-8-sig")

    print(f"\n追蹤與判定完成！")
    print(f"總共抓出 {len(df_output)} 筆疑似違規事件 (已排除重複與背景垃圾)。")
    print(f"背景/候選垃圾略過配對次數: {skipped_static_pairing}")
    print(f"垃圾運動參數: EPS_MOVE={EPS_MOVE}, N_STATIC={N_STATIC}, N_CANDIDATE={N_CANDIDATE}")
    print(f"結果已儲存至: {OUTPUT_CSV}")
    print(f"檔名與 Frame 對照已儲存至: {OUTPUT_FRAME_INDEX_CSV}")
else:
    print("\n追蹤完成。在目前的門檻設定下，沒有發現任何違規行為。")
    print(f"背景/候選垃圾略過配對次數: {skipped_static_pairing}")
