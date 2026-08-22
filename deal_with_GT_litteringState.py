#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
讀取 ground_truth_frome_8_class_gt.csv，篩選「真實標籤」為
trash_flying | vehicle_littering 的列，另存為 deal_with_GT_litteringState.csv。
不會修改原始 ground_truth_frome_8_class_gt.csv。
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "ground_truth_frome_8_class_gt.csv"
OUTPUT_CSV = BASE_DIR / "deal_with_GT_litteringState.csv"

TARGET_LABEL = "trash_flying | vehicle_littering"


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"找不到檔案：{INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    if "真實標籤" not in df.columns:
        print("CSV 缺少「真實標籤」欄位，請確認格式與 ground_truth_frome_8_class_gt.csv 一致。")
        return

    filtered = df[df["真實標籤"].astype(str) == TARGET_LABEL]
    filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"原始列數: {len(df)}")
    print(f"篩選後列數 ({TARGET_LABEL}): {len(filtered)}")
    print(f"已另存: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
