import pandas as pd
from datetime import datetime
import os

# ==============================
# 1️⃣ File paths
# ==============================
BASE_PATH = r"D:\Projects\AIforGood\data"
ENV_FILE = os.path.join(BASE_PATH, "environment_scores.csv")
SOCIAL_FILE = os.path.join(BASE_PATH, "social_scores.csv")
GOV_FILE = os.path.join(BASE_PATH, "governance_scores.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "esg_summary.csv")

# ==============================
# 2️⃣ Load datasets
# ==============================
print("🔹 Loading data files...")
env_df = pd.read_csv(ENV_FILE)
soc_df = pd.read_csv(SOCIAL_FILE)
gov_df = pd.read_csv(GOV_FILE)
print("✅ Files loaded successfully.")

# ==============================
# 3️⃣ Standardize and clean date columns
# ==============================
def extract_date(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            df['date'] = pd.to_datetime(df[col], errors='coerce').dt.date
            return df
    raise KeyError(f"No valid date column found in {df.columns}")

env_df = extract_date(env_df, ['date', 'Date'])
soc_df = extract_date(soc_df, ['Timestamp', 'date', 'Date'])
gov_df = extract_date(gov_df, ['date', 'Date', 'Timestamp']) if 'date' not in gov_df.columns else gov_df

# Governance is aggregated citywide (since it may not have per-date entries)
if 'governance_score' in gov_df.columns:
    gov_mean = gov_df['governance_score'].mean()
else:
    gov_mean = 50.0  # fallback baseline

# ==============================
# 4️⃣ Prepare components
# ==============================
env_df['E_raw'] = env_df['daily_EIS']
soc_df['S_raw'] = soc_df['S_raw_0_100']
gov_df['G_raw'] = gov_mean

# Assign ESG weights
E_WEIGHT = 0.40
S_WEIGHT = 0.35
G_WEIGHT = 0.25

# ==============================
# 5️⃣ Merge datasets on date
# ==============================
merged_df = pd.merge(env_df, soc_df, on='date', how='inner')

# Governance score remains constant across dates (city-level)
merged_df['G_raw'] = gov_mean

# ==============================
# 6️⃣ Compute weighted ESG
# ==============================
merged_df['E_weighted'] = merged_df['E_raw'] * E_WEIGHT / 100
merged_df['S_weighted'] = merged_df['S_raw'] * S_WEIGHT / 100
merged_df['G_weighted'] = merged_df['G_raw'] * G_WEIGHT / 100

merged_df['ESG_Score'] = (
    merged_df['E_raw'] * E_WEIGHT +
    merged_df['S_raw'] * S_WEIGHT +
    merged_df['G_raw'] * G_WEIGHT
).round(2)

# ==============================
# 7️⃣ Save and display
# ==============================
final_cols = [
    'date', 'E_raw', 'S_raw', 'G_raw',
    'E_weighted', 'S_weighted', 'G_weighted', 'ESG_Score'
]
final_df = merged_df[final_cols].sort_values('date')
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ ESG summary generated successfully!")
print(f"📁 Saved to: {OUTPUT_FILE}\n")
print(final_df.head(10))
