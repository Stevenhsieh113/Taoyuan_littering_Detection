import csv
from pathlib import Path
from statistics import median
import pandas as pd
import matplotlib.pyplot as plt


# 固定輸入檔（同規則、合併計算中位數）
PRIMARY_CSV_CANDIDATES = [
    Path("label_distance_result.csv"),
    Path("label_distance_results.csv"),
]
SECONDARY_CSV = Path("label_distance_results_2.csv")

FLAG_COLUMN = "有沒有數字為2,5,7(有=1,沒有=0)"
DISTANCE_COLUMN = "計算到的物體距離"
CSV_ENCODINGS = ("utf-8-sig", "cp950", "big5", "utf-8")
BOXPLOT_OUTPUT = Path("distance_boxplot.png")


def resolve_primary_csv_path() -> Path:
    for path in PRIMARY_CSV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "找不到距離結果檔，請確認以下其中之一存在："
        + ", ".join(str(p) for p in PRIMARY_CSV_CANDIDATES)
    )


def collect_distances_from_csv(csv_path: Path) -> list[float]:
    # 有些檔案雖然副檔名是 .csv，但實際內容是 xlsx (ZIP, 檔頭為 PK)
    header = csv_path.read_bytes()[:4]
    if header.startswith(b"PK"):
        df = pd.read_excel(csv_path)
        missing_columns = [c for c in (FLAG_COLUMN, DISTANCE_COLUMN) if c not in df.columns]
        if missing_columns:
            raise ValueError(f"檔案缺少必要欄位：{missing_columns}")

        flag_series = pd.to_numeric(df[FLAG_COLUMN], errors="coerce")
        distance_series = pd.to_numeric(df[DISTANCE_COLUMN], errors="coerce")
        valid_distances = distance_series[flag_series == 1].dropna()
        return valid_distances.astype(float).tolist()

    last_decode_error = None
    for encoding in CSV_ENCODINGS:
        try:
            distances = []
            with csv_path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV 沒有欄位標題。")

                missing_columns = [c for c in (FLAG_COLUMN, DISTANCE_COLUMN) if c not in reader.fieldnames]
                if missing_columns:
                    raise ValueError(f"CSV 缺少必要欄位：{missing_columns}")

                for row in reader:
                    flag = str(row.get(FLAG_COLUMN, "")).strip()
                    if flag != "1":
                        continue

                    distance_text = str(row.get(DISTANCE_COLUMN, "")).strip()
                    if not distance_text:
                        continue

                    try:
                        distances.append(float(distance_text))
                    except ValueError:
                        continue
            return distances
        except UnicodeDecodeError as e:
            last_decode_error = e
            continue

    if last_decode_error is not None:
        raise UnicodeDecodeError(
            last_decode_error.encoding,
            last_decode_error.object,
            last_decode_error.start,
            last_decode_error.end,
            f"無法解碼檔案 {csv_path}，已嘗試編碼：{CSV_ENCODINGS}",
        )
    return []


def main() -> None:
    primary_csv_path = resolve_primary_csv_path()
    if not SECONDARY_CSV.exists():
        raise FileNotFoundError(f"找不到第二份輸入檔：{SECONDARY_CSV}")

    distances = []
    distances.extend(collect_distances_from_csv(primary_csv_path))
    distances.extend(collect_distances_from_csv(SECONDARY_CSV))

    if not distances:
        print("沒有找到符合條件（有沒有數字為2,5,7 = 1）的有效距離資料。")
        return

    result = median(distances)
    q3 = pd.Series(distances).quantile(0.75)
    print(f"使用檔案：{primary_csv_path}, {SECONDARY_CSV}")
    print(f"符合條件筆數：{len(distances)}")
    print(f"距離中位數：{result:.6f}")
    print(f"距離 Q3 (75th percentile)：{q3:.6f}")

    # 產生並儲存 box plot
    plt.figure(figsize=(8, 5))
    plt.boxplot(distances, vert=True, patch_artist=True, labels=["Distance"])
    plt.title("Distance Distribution (Box Plot)")
    plt.ylabel("Distance (pixels)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(BOXPLOT_OUTPUT, dpi=150)
    plt.close()
    print(f"Box plot 已輸出：{BOXPLOT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
