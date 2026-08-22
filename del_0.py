#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從 trash_flying 相鄰幀距離 CSV 移除距離為 0 的列，輸出至 del_0.csv。"""

import csv
from pathlib import Path

INPUT_CSV = Path("trash_flying_adjacent_frame_distances.csv")
OUTPUT_CSV = Path("del_0.csv")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到輸入檔：{INPUT_CSV.resolve()}")

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"輸入檔沒有欄位標題：{INPUT_CSV}")

        all_rows = list(reader)

    rows = [row for row in all_rows if float(row["距離"]) != 0]
    removed = len(all_rows) - len(rows)

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"輸入：{INPUT_CSV.name}")
    print(f"輸出：{OUTPUT_CSV.resolve()}")
    print(f"保留 {len(rows)} 筆，移除距離為 0 共 {removed} 筆")


if __name__ == "__main__":
    main()
