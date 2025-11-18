# diabetes-trends
Dataset - CDC US Diabetes surveillance system 2015- LATEST_YEAR
Venv used
# Diabetes Trends (US) — README (short)

## What this repo does

Cleans and merges CDC Diabetes Surveillance data (state-level, **2014–2023**) into an analysis-ready panel with risk-factor features, then produces train/val/test splits for modeling.

## Data sources (CDC Surveillance → “Line” download)

* **Diagnosed Diabetes (Age-adjusted %, Adults 18+)**
* **Obesity (BMI ≥30)**
* **Physical Inactivity**
* **Current Smoking**

Each is exported as a CSV covering all years and all states.

## Repo layout (key paths)

```
data/
  raw/        # drop raw CSVs here (from CDC “Line” download)
  interim/    # cleaned per-source CSVs (auto)
  processed/  # merged panel + X/y splits (auto)
reports/
  cleaning_log.md   # what the cleaner did
src/
  clean.py          # raw -> interim
  preprocess.py     # interim -> processed (panel + splits)
columns_config.yaml # header/unit mapping
data_dictionary.yaml# schema & units (doc-only)
```

## How to run (VS Code, no terminal)

1. **Put raw files** in `data/raw/`

   * Filenames can be:
     `DiabetesAtlas_AllStatesLineChartData.csv` (diabetes),
     `cdc_obesity_2014_2023.csv`, `cdc_inactivity_2014_2023.csv`, `cdc_smoking_2014_2023.csv`.
2. **Clean**

   * Open `src/clean.py` → click **Run ▶️**
   * Outputs: `data/interim/*_clean.csv` (+ log in `reports/cleaning_log.md`)
3. **Preprocess**

   * Open `src/preprocess.py` → **Run ▶️**
   * Outputs (CSV):

     * `data/processed/diabetes_panel.csv`
     * `X_train.csv`, `X_val.csv`, `X_test.csv`
     * `y_train.csv`, `y_val.csv`, `y_test.csv`

> We use **CSV-only outputs** for readability. Parquet files (if any) can be ignored.

## What you should see

* `diabetes_panel.csv` ≈ 50–51 states × 10 years ≈ **500–510 rows**
* Columns include: `state_fips, state, year, diabetes_prevalence, obesity_prevalence, inactivity_prevalence, smoking_prevalence` (+ CI columns per indicator).

## Adding more data later

1. Drop the new CSV in `data/raw/`.
2. If header names differ, add mappings in **`columns_config.yaml`**.
3. Re-run **clean.py** then **preprocess.py**.
4. Document the new field in **`data_dictionary.yaml`**.

## Troubleshooting (fast)

* **No interim files found**: make sure your raw CSVs are in `data/raw/` and named with `.csv`. Run `clean.py` first.
* **MergeError**: run the committed `preprocess.py`; it de-duplicates overlapping columns.
* **Weird years**: the cleaner filters to **2014–2023** automatically.
* **VS Code can’t open Parquet**: we write CSVs; use those.

## Hand-off for modeling/EDA

* Use `data/processed/diabetes_panel.csv` for EDA/trends.
* Use `X_train.csv`/`y_train.csv` (and val/test) for models.

## Notes

* All rates are **percent** (0–100) and **age-adjusted** where applicable.
* `reports/cleaning_log.md` captures file-by-file actions for reproducibility.

---

Questions? Ping Chinaza (data cleaning & preprocessing lead).
