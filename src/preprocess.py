from pathlib import Path
from functools import reduce
import pandas as pd

KEYS = ["state_fips","state","year"]

def infer_label_from_path(p):
    s = p.stem.lower()
    if "obesity" in s: return "obesity"
    if "inactivity" in s: return "inactivity"
    if "smoking" in s: return "smoking"
    if "diabetes" in s: return "diabetes"
    return "unknown"

def prepare_df(df, label):
    df = df.copy()
    keep = [c for c in KEYS if c in df.columns]

    ci_map = {}
    if "ci_low" in df.columns:  ci_map["ci_low"]  = f"{label}_ci_low"
    if "ci_high" in df.columns: ci_map["ci_high"] = f"{label}_ci_high"
    if ci_map:
        df = df.rename(columns=ci_map)

    if "percentage" in df.columns:
        target_name = "diabetes_prevalence" if label == "diabetes" else f"{label}_prevalence"
        df = df.rename(columns={"percentage": target_name})

    return df

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
    paths = sorted(INTERIM.glob("*_clean.csv"))  # CSVs written by clean.py
    if not paths:
        raise RuntimeError("No interim CSVs found. Run clean.py first.")
    return [pd.read_csv(p) for p in paths]


def outer_join_on_keys(dfs):
    def _merge(l, r):
        keys = [k for k in KEYS if k in l.columns and k in r.columns]
        if not keys:
            keys = [k for k in ["year"] if k in l.columns and k in r.columns]

        # Drop overlapping non-key columns from the right df to avoid duplicates
        overlap = set(l.columns).intersection(r.columns) - set(keys)
        if overlap:
            r = r.drop(columns=list(overlap))

        return pd.merge(l, r, on=keys, how="outer")

    from functools import reduce
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
    panel.to_csv(PROCESSED / "diabetes_panel.csv", index=False)

    if TARGET in panel.columns:
        train, val, test = time_splits(panel, TARGET)
        for name, df in [("train", train), ("val", val), ("test", test)]:
            df.dropna(subset=["year"], inplace=True)
            X = df.drop(columns=[TARGET], errors="ignore")
            y = df[[TARGET]]
            X.to_csv(PROCESSED / f"X_{name}.csv", index=False)
            y.to_csv(PROCESSED / f"y_{name}.csv", index=False)

if __name__ == "__main__":
    main()
