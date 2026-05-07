import pandas as pd
from pathlib import Path
import sys

# Locate the data directory relative to this file (project root / data)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_files(data_dir: Path):
    train_fp = data_dir / "train_FD001.txt"
    test_fp = data_dir / "test_FD001.txt"
    rul_fp = data_dir / "RUL_FD001.txt"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train = pd.read_csv(train_fp, sep=r"\s+", header=None, names=columns)
    test = pd.read_csv(test_fp, sep=r"\s+", header=None, names=columns)
    rul = pd.read_csv(rul_fp, sep=r"\s+", header=None, names=["RUL"])

    return train, test, rul


def summarize(df: pd.DataFrame, name: str):
    print(f"--- {name} ---")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes.to_string())
    na_total = df.isna().sum().sum()
    print("total missing:", int(na_total))
    print("unique engine_ids:", int(df["engine_id"].nunique()))
    engines = df["engine_id"].unique()[:5].tolist()
    print("sample engine_ids:", engines)
    print(df.head().to_string(index=False))
    print()


def main():
    try:
        train, test, rul = load_files(DATA_DIR)
    except Exception as e:
        print("Error loading data:", e, file=sys.stderr)
        sys.exit(1)

    summarize(train, "Train")
    summarize(test, "Test")
    print("--- RUL ---")
    print("shape:", rul.shape)
    print(rul.head().to_string(index=False))

    # Quick consistency check: RUL length should match number of engines in test
    test_engines = test["engine_id"].nunique()
    rul_len = len(rul)
    print(f"test engines: {test_engines}, RUL rows: {rul_len}")


if __name__ == "__main__":
    main()