import pandas as pd
import matplotlib.pyplot as plt
 
# Step 1: Load CSV data
file_path = '/Users/tanuskabiswakarma/CSC4740_project/diabetes_panel.csv'  
data = pd.read_csv(file_path)

# Step 2: Clean column names
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
print("Columns in CSV:", data.columns)

# Step 3: Detect relevant columns
year_col = [col for col in data.columns if 'year' in col.lower()]
diabetes_col = [col for col in data.columns if 'diabetes_prevalence' in col.lower()]
state_col = [col for col in data.columns if 'state' in col.lower()]
risk_cols = [col for col in data.columns if 'obesity' in col.lower() or 'inactivity' in col.lower()]

if not year_col or not diabetes_col:
    raise ValueError("Could not find 'Year' or 'Diabetes' columns in your CSV.")

year_col = year_col[0]
diabetes_col = diabetes_col[0]
state_col = state_col[0] if state_col else None

# Line Graph 
plt.figure(figsize=(10, 6))
plt.plot(data[year_col], data[diabetes_col], marker='o', color='blue', linewidth=2)
plt.title('Diabetes Prevalence in the US (2014–2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Diabetes Prevalence (%)', fontsize=12)
plt.xticks(data[year_col], rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Bar Graph 
if state_col:
    latest_year = data[year_col].max()
    latest_data = data[data[year_col] == latest_year]
    
    plt.figure(figsize=(12, 6))
    plt.bar(latest_data[state_col], latest_data[diabetes_col], color='orange')
    plt.title(f'Diabetes Prevalence by State in {latest_year}', fontsize=16)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Diabetes Prevalence (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Bar Graph 
if state_col:
    latest_year = data[year_col].min()
    latest_data = data[data[year_col] == latest_year]
    
    plt.figure(figsize=(12, 6))
    plt.bar(latest_data[state_col], latest_data[diabetes_col], color='orange')
    plt.title(f'Diabetes Prevalence by State in {latest_year}', fontsize=16)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Diabetes Prevalence (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Scatter Plot 
for risk in risk_cols:
    plt.figure(figsize=(8, 6))
    plt.scatter(data[risk], data[diabetes_col], color='green', alpha=0.7)
    plt.title(f'Diabetes Prevalence vs {risk}', fontsize=16)
    plt.xlabel(f'{risk} (%)', fontsize=12)
    plt.ylabel('Diabetes Prevalence (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Scatter Plot for Smoking Prevalence
smoking_col = [col for col in data.columns if 'smoking_prevalence' in col.lower()]
if not smoking_col:
    raise ValueError("Could not find 'smoking_prevalence' column in your CSV.")
smoking_col = smoking_col[0]

plt.figure(figsize=(10, 6))
plt.scatter(data[smoking_col], data[diabetes_col], color='green', alpha=0.7)

plt.title('Diabetes Prevalence vs Smoking Prevalence', fontsize=16)
plt.xlabel('Smoking Prevalence (%)', fontsize=12)
plt.ylabel('Diabetes Prevalence (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
