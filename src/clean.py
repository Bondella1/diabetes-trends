import re
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime

THIS = Path(__file__).resolve()
ROOT = THIS.parent
for _ in range(5):  # walk up a few levels just in case
    if (ROOT / "columns_config.yaml").exists() and (ROOT / "data").exists():
        break
    ROOT = ROOT.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
INTERIM.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# USPS -> FIPS
STATE_LOOKUP = {
    "AL":1,"AK":2,"AZ":4,"AR":5,"CA":6,"CO":8,"CT":9,"DE":10,"DC":11,"FL":12,"GA":13,"HI":15,"ID":16,
    "IL":17,"IN":18,"IA":19,"KS":20,"KY":21,"LA":22,"ME":23,"MD":24,"MA":25,"MI":26,"MN":27,"MS":28,
    "MO":29,"MT":30,"NE":31,"NV":32,"NH":33,"NJ":34,"NM":35,"NY":36,"NC":37,"ND":38,"OH":39,"OK":40,
    "OR":41,"PA":42,"RI":44,"SC":45,"SD":46,"TN":47,"TX":48,"UT":49,"VT":50,"VA":51,"WA":53,"WV":54,
    "WI":55,"WY":56
}
NAME_TO_USPS = { # for full state names in CDC files
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
    "Connecticut":"CT","Delaware":"DE","District of Columbia":"DC","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY",
    "Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
    "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
    "Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI",
    "South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}

YEAR_MIN, YEAR_MAX = 2014, 2023

def load_config():
    with open(ROOT / "columns_config.yaml", "r") as f:
        return yaml.safe_load(f)

def standardize_colnames(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+","_", regex=True).str.strip("_"))
    return df

def read_cdc_csv(path: Path) -> pd.DataFrame:
    # CDC "Line chart" CSV: first rows are titles, then the header with State/Year
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_idx = None
    for i, line in enumerate(text[:50]):  # scan a generous range
        if ("," in line) and ("State" in line) and ("Year" in line):
            header_idx = i
            break
    if header_idx is None:
        # Fallback: first comma-containing line
        for i, line in enumerate(text):
            if "," in line:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("Could not locate a CSV header with commas.")

    # Use the C engine first; if it complains, fall back to the python engine
    try:
        return pd.read_csv(path, skiprows=header_idx)
    except Exception:
        return pd.read_csv(path, skiprows=header_idx, engine="python")

def map_headers(df, cfg):
    rev = {}
    for group in ("standard_keys","targets","features"):
        for std_key, aliases in (cfg.get(group) or {}).items():
            for a in (aliases or []):
                rev[a.lower()] = std_key
    return df.rename(columns={c: rev.get(c, c) for c in df.columns})
def ensure_geo(df):
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
        df["state"] = df["state"].map(lambda x: NAME_TO_USPS.get(x, x)).str.upper()
    if "state_fips" not in df.columns and "state" in df.columns:
        df["state_fips"] = df["state"].map(STATE_LOOKUP)
    return df

def coerce_types(df):
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "state_fips" in df.columns:
        df["state_fips"] = pd.to_numeric(df["state_fips"], errors="coerce").astype("Int64")
    return df

def fix_units(df, cfg):
    as_percent = bool(cfg.get("store_as_percent", True))
    for c in df.columns:
        if any(k in c for k in ["prevalence","rate","pct","percentage","insecurity"]):
            s = pd.to_numeric(df[c], errors="coerce")
            if as_percent and s.max(skipna=True) is not None and s.max(skipna=True) <= 1.0:
                s = s * 100.0
            if (not as_percent) and s.max(skipna=True) is not None and s.max(skipna=True) > 1.0:
                s = s / 100.0
            df[c] = s.clip(lower=0)
    return df

def clean_one_file(path: Path, cfg):
    try:
        df = read_cdc_csv(path)
    except Exception:
        df = pd.read_csv(path)

    df = standardize_colnames(df)
    
    blob = (path.name + " " + path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]).lower()

    if ("obesity" in blob) or ("bmi" in blob):
        indicator = "obesity"
    elif ("inactivity" in blob) or ("physical inactivity" in blob):
        indicator = "inactivity"
    elif "smoking" in blob:
        indicator = "smoking"
    else:
        indicator = "diabetes"   # default

    rename_map = {
    "percentage": "diabetes_prevalence" if indicator == "diabetes" else f"{indicator}_prevalence",
    "lower_limit": "ci_low",
    "upper_limit": "ci_high",
    "_upper_limit": "ci_high",  # handles a leading space that becomes "_upper_limit"
    }
    df = df.rename(columns=rename_map)

    # Normalize known CDC column names
    df = df.rename(columns={"percentage":"diabetes_prevalence",
                            "lower_limit":"ci_low","upper_limit":"ci_high"})

    # Drop pseudo-rows
    if "state" in df.columns:
        s = df["state"].astype(str).str.strip().str.lower()
        df = df[~s.isin(["median of states","median_of_states","united states","united_states"])]

    df = map_headers(df, cfg)
    df = ensure_geo(df)
    df = coerce_types(df)
    df = fix_units(df, cfg)

    # Keep 2014–2023
    if "year" in df.columns:
        df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]

    # Save
    out_path = INTERIM / (path.stem + ".parquet")
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as e:
    # No pyarrow/fastparquet? Write CSV so we can keep going.
        out_csv = INTERIM / (path.stem + "_clean.csv")
        df.to_csv(out_csv, index=False)

        with open(REPORTS / "cleaning_log.md", "a") as log:
            log.write(f"- {datetime.now().isoformat()} | {path.name} → {out_path.name} | rows={len(df)}\n")
        return df

def main():
    cfg = load_config()

    print("CLEAN ROOT:", ROOT)
    print("RAW PATH :", RAW)
    print("FILES IN RAW:", [x.name for x in RAW.glob("*")])

    found_any = False
    for p in sorted(RAW.glob("*.[cC][sS][vV]")):   # matches .csv and .CSV
        found_any = True
        print("→ Processing:", p.name)
        try:
            df = clean_one_file(p, cfg)

            out_path = INTERIM / (p.stem + ".parquet")
            try:
                df.to_parquet(out_path, index=False)
                print("   wrote:", out_path)
            except Exception as e:
                # Parquet fallback to CSV
                out_csv = INTERIM / (p.stem + "_clean.csv")
                df.to_csv(out_csv, index=False)
                print("   parquet failed, wrote CSV instead:", out_csv)
                with open(REPORTS / "cleaning_log.md", "a") as log:
                    log.write(f"- {datetime.now().isoformat()} | PARQUET FAIL {p.name}: {e} | wrote {out_csv.name}\n")

        except Exception as e:
            print("   ERROR:", e)
            with open(REPORTS / "cleaning_log.md", "a") as log:
                log.write(f"- {datetime.now().isoformat()} | ERROR {p.name}: {e}\n")

    if not found_any:
        print("No CSVs found in", RAW)
        with open(REPORTS / "cleaning_log.md", "a") as log:
            log.write(f"- {datetime.now().isoformat()} | WARNING: no CSVs in data/raw\n")

if __name__ == "__main__":
    main()
