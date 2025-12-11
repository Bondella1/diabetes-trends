import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 1. Load the data
CSV_PATH = '/Users/tanuskabiswakarma/CSC4740_project/diabetes_panel.csv' 

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found.")

df = pd.read_csv(CSV_PATH)

# Basic sanity checks - change column names below if file uses different names
expected_cols = {'state', 'year', 'diabetes_prevalence'}
if not expected_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {expected_cols}. Found: {df.columns.tolist()}")

# If prevalence is stored as proportion (0-1), convert to percent for readability
if df['diabetes_prevalence'].max() <= 1.01:
    df['diabetes_prevalence'] = df['diabetes_prevalence'] * 100


# 2. Map states to regions
south = {'AL','AR','DE','DC','FL','GA','KY','LA','MD','MS','NC','OK','SC','TN','TX','VA','WV'}
northeast = {'CT','ME','MA','NH','NJ','NY','PA','RI','VT'}
midwest = {'IL','IN','IA','KS','MI','MN','MO','NE','ND','OH','SD','WI'}
west = {'AK','AZ','CA','CO','HI','ID','MT','NV','NM','OR','UT','WA','WY'}

def assign_region(state_abbr):
    st = str(state_abbr).strip().upper()
    if st in south:
        return 'South'
    if st in northeast:
        return 'Northeast'
    if st in west:
        return 'West'
    if st in midwest:
        return 'Midwest'
    return 'Other'

df['state_abbr'] = df['state'].astype(str).str.strip()
if (df['state_abbr'].str.len() > 2).mean() > 0.5:
    print("Warning: >50% of 'state' values look like full names. If so, convert full names to 2-letter abbreviations before running.")
df['region'] = df['state_abbr'].apply(assign_region)

# 3. Aggregate: region-year averages

region_trends = (
    df.groupby(['year', 'region'], as_index=False)
      .agg(mean_prev=('diabetes_prevalence', 'mean'),
           se_prev=('diabetes_prevalence', lambda x: x.std(ddof=1) / np.sqrt(len(x))))
)

# Pivot for plotting convenience
pivot = region_trends.pivot(index='year', columns='region', values='mean_prev')


# 4. Plot region trends across years

plt.figure(figsize=(10, 6))
for region in ['South', 'Northeast', 'Midwest', 'West', 'Other']:
    if region in pivot.columns:
        plt.plot(pivot.index, pivot[region], marker='o', label=region)
plt.xlabel('Year')
plt.ylabel('Mean diabetes prevalence (%)')
plt.title('Mean adult diabetes prevalence by Census region (yearly averages)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('region_diabetes_trends.png', dpi=150)
print("Saved plot to region_diabetes_trends.png")


# 5. Fit an OLS growth model
#    Model: diabetes_prevalence ~ year_centered + C(region) + year_centered:C(region)
#    This gives region intercept differences and region-specific slopes.

# Center year to improve interpretability
df['year_c'] = df['year'] - df['year'].mean()

# Make region categorical and set reference
df['region'] = pd.Categorical(df['region'])


formula = 'diabetes_prevalence ~ year_c * C(region)'
ols_model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')  # robust SEs
print("\nOLS regression results (with year x region interaction):")
print(ols_model.summary())

# Interpretation helper
print("\nInterpretation tips:")
print("- The coefficient for C(region)[T.South] is the increase in prevalence (percentage points) for South vs baseline at the centered year.")
print("- The coefficient for year_c:C(region)[T.South] is the difference in slope (growth per year) for South vs baseline.")
print("- To test whether South has the largest prevalence, compare the region intercept coefficients and their significance.")
print("- To test whether Northeast and West are lower, check their C(region) coefficients (negative suggests lower than baseline).")


# 6. Fit a linear mixed effects model (states as random intercepts)

try:
    md = smf.mixedlm("diabetes_prevalence ~ year_c * C(region)", df, groups=df["state_abbr"])
    mdf = md.fit(reml=False)  # maximum likelihood
    print("\nLinear Mixed Effects (random intercept for state) results:")
    print(mdf.summary())
except Exception as e:
    print("\nMixedLM failed to run. Error:", e)
    print("You may need to install a more recent statsmodels or ensure there are enough observations per state.")
    mdf = None


# 7. Quick automated checks: which region has highest average overall and by most recent year

overall_by_region = df.groupby('region')['diabetes_prevalence'].mean().sort_values(ascending=False)
print("\nOverall average diabetes prevalence by region (descending):\n", overall_by_region)

most_recent_year = df['year'].max()
recent_by_region = (df[df['year'] == most_recent_year]
                    .groupby('region')['diabetes_prevalence']
                    .mean()
                    .sort_values(ascending=False))
print(f"\nAverage diabetes prevalence by region in the most recent year ({most_recent_year}):\n", recent_by_region)

# 8. pairwise contrasts to test 'South > Northeast' and 'South > West'

# Build contrast for intercept differences at centered year = 0 (year mean)
base = df['region'].cat.categories[0]  # baseline region used by the OLS design
print(f"\nCurrent baseline region for contrasts (OLS encoding): {base}")

# Helper to compare South vs X on intercept
def contrast_region_intercept(model, region_name):
    param_name = f"C(region)[T.{region_name}]"
    if param_name not in model.params.index:
        return None
    coef = model.params.get(param_name, 0.0)
    se = model.bse.get(param_name, np.nan)
    tval = coef / se
    pval = model.pvalues.get(param_name, np.nan)
    return {'region': region_name, 'coef': coef, 'se': se, 't': tval, 'p': pval}

for r in ['South', 'Northeast', 'West']:
    c = contrast_region_intercept(ols_model, r)
    if c is not None:
        print(f"Contrast intercept for {r}: coef={c['coef']:.3f}, se={c['se']:.3f}, t={c['t']:.3f}, p={c['p']:.3g}")
    else:
        print(f"No direct intercept parameter for {r} (it may be baseline or not present).")

# 9. Save summary outputs 

with open('ols_summary.txt', 'w') as f:
    f.write(ols_model.summary().as_text())
if mdf is not None:
    with open('mixedlm_summary.txt', 'w') as f:
        f.write(mdf.summary().as_text())

print("\nAnalysis complete. Outputs produced:")
print(" - region_diabetes_trends.png (plot)")
print(" - ols_summary.txt")
if mdf is not None:
    print(" - mixedlm_summary.txt")

