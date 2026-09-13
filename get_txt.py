#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
從 zip 內只解壓縮 labels/train 底下的 .txt 檔
輸出到同名資料夾：<zip檔名去副檔名>_txt

例如：
  大溪_20250608153937-紙屑_processed.zip
輸出到：
  大溪_20250608153937-紙屑_processed_txt/
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def extract_train_txt_from_zip(zip_path: Path) -> int:
    """
    解壓縮單一 zip：
    - 只取 labels/train/ 底下的 .txt
    - 不刪除既有檔案；若目標 txt 已存在就跳過
    - 盡量保留 zip 內 labels/train/ 後續的相對路徑
    """
    if not zip_path.exists():
        print(f"  ⚠️ 找不到 zip：{zip_path}")
        return 0

    out_dir = zip_path.with_suffix("").name + "_txt"
    out_dir_path = zip_path.parent / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    extracted_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_posix = member.replace("\\", "/")

            # 只處理 labels/train/ 底下的 txt
            if not member_posix.startswith("labels/train/"):
                continue
            if not member_posix.lower().endswith(".txt"):
                continue

            # 計算相對路徑（保留 labels/train/ 之後的路徑）
            rel_posix = member_posix[len("labels/train/") :]
            rel_path = PurePosixPath(rel_posix)

            # 避免路徑穿越：如包含 .. 就略過
            if ".." in rel_path.parts:
                print(f"  ⚠️ 略過可疑路徑：{member}")
                continue

            target_path = out_dir_path.joinpath(*rel_path.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 若已存在就不覆蓋，避免動到既有檔案
            if target_path.exists():
                continue

            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted_count += 1

    print(f"  ✅ {zip_path.name}：新增解壓 {extracted_count} 個 txt")
    return extracted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只解壓 zip 內 labels/train 的 txt 到同名資料夾（不動其他檔案）"
    )
    parser.add_argument(
        "--zip",
        nargs="*",
        default=None,
        help="指定要處理的 zip 檔名（可多個）。不提供則處理目前資料夾底下所有 *.zip。",
    )
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parent

    if args.zip:
        zip_paths = []
        for z in args.zip:
            candidate = Path(z)
            if not candidate.is_absolute():
                candidate = work_dir / candidate
            zip_paths.append(candidate)
    else:
        zip_paths = sorted(work_dir.glob("*.zip"))

    if not zip_paths:
        print(f"⚠️ 在資料夾找不到 zip：{work_dir}")
        return

    print(f"處理資料夾：{work_dir}")
    total = 0
    for zp in zip_paths:
        total += extract_train_txt_from_zip(zp)

    print(f"完成：新增解壓總計 {total} 個 txt")


if __name__ == "__main__":
    main()

