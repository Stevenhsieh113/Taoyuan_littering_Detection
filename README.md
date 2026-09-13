# Taoyuan_littering_Detection_第二階段流程
# 簡介
## input準備
split_trash_labels.py
split_8class_labels.py
get_txt.py

## 第二階段input
垃圾模擬：new_new_mock_trash_csv.py (替換成第一階段結果)
人車資料： test_yolo11n_find_car_and_person.py

## GT處理
ground_truth_frome_8_class_gt.py
deal_with_GT_litteringState.py


## 調參數(可跑可不跑)
#### 人車距離參數
calculate_dis.py + find_threshold.py (找中位數或Q3)
定好數字(中位數或Q3)後，調整final_check_new_correct.py 的 THRESHOLD_PIXELS數值。

#### 過濾背景垃圾
cal_trash_dist_between_two_frames.py
del_0.py
定好數字後，改的是 final_check_new_correct.py 的 EPS_MOVE（以及你可能一起調的 N_STATIC、N_CANDIDATE）。

## 第二階段執行
final_check_new_correct.py
511compare_ground_truth_and_predict.py

## 視覺化結果
visualize_littering.py（可跑可不跑）
visualize_fn_pred_frames.py（可跑可不跑）



-----
# 按流程順序執行

## 0. get_txt.py
讀取yolo格是(CVAT)匯出的zip，產生三種狀態的txt

## 1. split_trash_labels.py
把原始 YOLO 垃圾框（class 2）對上 XML 的 State，拆成一般垃圾 / 飛行垃圾，並去掉人、車。(有很多空白檔案是正常的，因為只有有出現Trash才會有內容)

輸入：根目錄 *.xml、對應的 某影片_txt/*.txt
輸出：labels_split/（0 = Trash，1 = Trash_Flying）

## 2. split_8class_labels.py
用 XML 的 Action/State，把 3 類 YOLO 標註展開成 8 類（人/車/垃圾及其 holding、littering、flying）。

輸入：根目錄 *.xml、對應的 某影片_txt/*.txt
輸出：labels_8class/（class 0–7，給後面做 GT）

## 3. new_new_mock_trash_csv.py(要替換成yolo第一階段結果)
把 labels_split 的 YOLO 框轉成和人車 CSV 相同格式，當成「垃圾偵測結果」（信心度固定 1.0）。

輸入：image/、labels_split/
輸出：mock_trash_results.csv (模擬垃圾資料)

## 4. test_yolo11n_find_car_and_person.py
用 YOLO11n 掃圖片，偵測人與車（COCO：0,1,2,3,5,7）。

輸入：image/、yolo11n.pt
輸出：detect_results.csv (人車資訊)

## 5. ground_truth_frome_8_class_gt.py
掃描 labels_8class，把「真的在亂丟」相關標籤（人 littering、車 littering、飛垃圾）整理成 GT 表。

輸入：labels_8class/
輸出：ground_truth_frome_8_class_gt.csv

## 6. deal_with_GT_litteringState.py
從完整 GT 裡只留下評估要用的列：真實標籤為 trash_flying | vehicle_littering。

輸入：ground_truth_frome_8_class_gt.csv
輸出：deal_with_GT_litteringState.csv
請注意：因為之前資料都為車和垃圾居多，所以之後有人丟垃圾的畫面，要補入person_littering| trash_flying

## 7. final_check_new_correct.py
主判定：同一幀人/車與垃圾夠近就配對，垃圾跨幀追 ID，並把幾乎沒動的背景垃圾過濾掉。

輸入：detect_results.csv、mock_trash_results.csv
輸出：suspected_littering_with_tracker.csv、suspected_littering_frame_index.csv

## 8.511compare_ground_truth_and_predict.py
用預測事件對 GT 事件算 TP / FP / FN（同一影片、幀差在容差內算命中），結果印在終端機。

輸入：suspected_littering_with_tracker.csv、deal_with_GT_litteringState.csv
輸出：終端機數字（Precision / Recall / F1），不寫 CSV


## 9.visualize_littering.py（可跑可不跑）
把判定結果畫回原圖：綠框人/車、紅框垃圾、中間連線，用來肉眼檢查配對合不合理。

輸入：suspected_littering_with_tracker.csv、對應原圖
輸出：littering_visualization_output_with_tracker/new/ 裡的畫框圖

## 10.visualize_fn_pred_frames.py（可跑可不跑）
專門看 FN：在「模型預測的那一幀」畫出該幀全部人/車與垃圾；若有配對成功的那一組會加粗。

輸入：detect_results.csv、mock_trash_results.csv、suspected_littering_with_tracker.csv、evaluation_report/06_fn_only.csv、03_by_video.csv、原圖
輸出：evaluation_report/fn_pred_frame_visualizations/


## 看Result
visualize_littering.py

------
## 調參數(人車距離門檻)
## calculate_dis.py
讀 8 類標註，算同一幀裡物體中心點之間的像素距離。

輸入：labels_8class/、image/
輸出：label_distance_results.csv
find_threshold.py

## find_threshold.py
把距離表裡「該幀有 2/5/7」的數字拿來算中位數、Q3，並畫箱型圖，當配對門檻參考。

輸入：label_distance_results.csv（或 label_distance_result.csv）、必須再有 label_distance_results_2.csv
輸出：終端機印中位數 / Q3、distance_boxplot.png

----
## 調參數(減少背景垃圾干擾)
## cal_trash_dist_between_two_frames.py
只看 class 7（trash_flying），算有標註的相鄰兩幀，垃圾中心點移動了幾像素。

輸入：labels_8class/、image/
輸出：trash_flying_adjacent_frame_distances.csv

## del_0.py
把上一支結果裡距離為 0 的列刪掉（座標完全沒變，多半是背景複製），方便看「真的有在動」的分布。

輸入：trash_flying_adjacent_frame_distances.csv
輸出：del_0.csv

