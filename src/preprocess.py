from pathlib import Path
from functools import reduce
import pandas as pd

THIS = Path(__file__).resolve()
ROOT = THIS.parent
for _ in range(4):  # walk up a few levels just in case
    if (ROOT / "columns_config.yaml").exists() and (ROOT / "data").exists():
        break
    ROOT = ROOT.parent
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

TARGET = "diabetes_prevalence"

def load_interim():
    dfs = []
    parqs = sorted(INTERIM.glob("*.parquet"))
    csvs  = sorted(INTERIM.glob("*_clean.csv"))

    if not parqs and not csvs:
        raise RuntimeError("No interim files found. Run clean.py first.")

    for p in parqs:
        dfs.append(pd.read_parquet(p))
    for c in csvs:
        dfs.append(pd.read_csv(c))

    return dfs


def outer_join_on_keys(dfs):
    def _merge(l, r):
        keys = [k for k in ["state_fips","state","year"] if k in l.columns and k in r.columns]
        if not keys: keys = [k for k in ["year"] if k in l.columns and k in r.columns]
        return pd.merge(l, r, on=keys, how="outer")
    return reduce(_merge, dfs)

def infer_year_bounds(panel):
    yrs = panel["year"].dropna().astype(int)
    return yrs.min(), yrs.max()

def time_splits(panel, target=TARGET):
    keep = panel.dropna(subset=[target]).copy()
    y_min, y_max = infer_year_bounds(keep)
    test_year = y_max
    val_year  = max(y_min, y_max - 1)
    train = keep[keep["year"].between(y_min, max(y_min, val_year - 1))]
    val   = keep[keep["year"] == val_year]
    test  = keep[keep["year"] == test_year]
    return train, val, test

def main():
    dfs = load_interim()
    panel = outer_join_on_keys(dfs)
    if "state_fips" in panel.columns:
        panel = panel[panel["state_fips"].notna()]
    panel = panel.sort_values(["year","state_fips"], ignore_index=True)
    panel.to_parquet(PROCESSED / "diabetes_panel.parquet", index=False)

    if TARGET in panel.columns:
        train, val, test = time_splits(panel, TARGET)
        for name, df in [("train", train), ("val", val), ("test", test)]:
            df.dropna(subset=["year"], inplace=True)
            X = df.drop(columns=[TARGET], errors="ignore")
            y = df[[TARGET]]
            X.to_parquet(PROCESSED / f"X_{name}.parquet", index=False)
            y.to_parquet(PROCESSED / f"y_{name}.parquet", index=False)

if __name__ == "__main__":
    main()
