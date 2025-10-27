import pandas as pd
import numpy as np

# ==========================================================
# 1️⃣ LOAD ALL DATASETS SAFELY
# ==========================================================
aqi_df = pd.read_csv(r'D:\Projects\AIforGood\data\openaq_buffalo.csv')
emission_df = pd.read_csv(r"D:\Projects\AIforGood\data\buffalo_emissions.csv")
green_df = pd.read_csv(r'D:\Projects\AIforGood\data\buffalo_green_cover_aug2025.csv')

# ==========================================================
# 2️⃣ CLEAN & STANDARDIZE TIMESTAMPS
# ==========================================================
def clean_date(df, col):
    df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
    df = df.dropna(subset=[col])
    df[col] = df[col].dt.date  # Keep only date part
    return df

aqi_df = clean_date(aqi_df, 'datetimeLocal')
emission_df = clean_date(emission_df, 'timestamp')
green_df = clean_date(green_df, 'timestamp')

# ==========================================================
# 3️⃣ CLEAN AND AGGREGATE RELEVANT COLUMNS
# ==========================================================

# --- Air Quality Data (average AQI per day)
aqi_daily = (
    aqi_df.groupby('datetimeLocal')['value']
    .mean()
    .reset_index()
    .rename(columns={'datetimeLocal': 'date', 'value': 'avg_aqi'})
)

# --- Emissions Data (average per day)
emission_daily = (
    emission_df.groupby('timestamp')[['CO2_emission_tons', 'NOx_emission_tons', 'SO2_emission_tons']]
    .mean()
    .reset_index()
    .rename(columns={'timestamp': 'date'})
)

# --- Green Cover Data (average per day)
green_daily = (
    green_df.groupby('timestamp')[['green_cover_percent', 'tree_density_per_sqkm', 'avg_surface_temp_C']]
    .mean()
    .reset_index()
    .rename(columns={'timestamp': 'date'})
)

# ==========================================================
# 4️⃣ MERGE ALL DATASETS ON DATE
# ==========================================================
env_df = (
    aqi_daily
    .merge(emission_daily, on='date', how='outer')
    .merge(green_daily, on='date', how='outer')
    .sort_values(by='date')
    .reset_index(drop=True)
)

# Fill missing values with column mean (since prototype)
env_df.fillna(env_df.mean(numeric_only=True), inplace=True)

# ==========================================================
# 5️⃣ NORMALIZE METRICS (0–1 SCALE)
# ==========================================================
def normalize(series, inverse=False):
    if series.max() == series.min():
        return pd.Series([0.5] * len(series))
    norm = (series - series.min()) / (series.max() - series.min())
    return 1 - norm if inverse else norm

env_df['norm_green_cover'] = normalize(env_df['green_cover_percent'])
env_df['norm_tree_density'] = normalize(env_df['tree_density_per_sqkm'])
env_df['norm_temp'] = normalize(env_df['avg_surface_temp_C'], inverse=True)
env_df['norm_aqi'] = normalize(env_df['avg_aqi'], inverse=True)
env_df['norm_co2'] = normalize(env_df['CO2_emission_tons'], inverse=True)
env_df['norm_nox'] = normalize(env_df['NOx_emission_tons'], inverse=True)
env_df['norm_so2'] = normalize(env_df['SO2_emission_tons'], inverse=True)

# ==========================================================
# 6️⃣ CALCULATE DAILY ENVIRONMENTAL IMPACT SCORE (EIS)
# ==========================================================
weights = {
    'green_cover': 0.20,
    'tree_density': 0.10,
    'temp': 0.15,
    'aqi': 0.20,
    'co2': 0.15,
    'nox': 0.10,
    'so2': 0.10
}

env_df['daily_EIS'] = (
    env_df['norm_green_cover'] * weights['green_cover'] +
    env_df['norm_tree_density'] * weights['tree_density'] +
    env_df['norm_temp'] * weights['temp'] +
    env_df['norm_aqi'] * weights['aqi'] +
    env_df['norm_co2'] * weights['co2'] +
    env_df['norm_nox'] * weights['nox'] +
    env_df['norm_so2'] * weights['so2']
) * 100  # Scale to 0–100

# ==========================================================
# 7️⃣ CALCULATE MONTHLY/OVERALL ENVIRONMENTAL SCORE
# ==========================================================
environment_score = env_df['daily_EIS'].mean()
esg_environment_component = 0.4 * environment_score

# ==========================================================
# 8️⃣ PRINT SUMMARY
# ==========================================================
print("🌍 ESG Environmental Model Summary")
print("=" * 60)
print(env_df[['date', 'avg_aqi', 'CO2_emission_tons', 'green_cover_percent', 'daily_EIS']].head(10))
print("\nAverage Environmental Score (E): {:.2f}/100".format(environment_score))
print("Environmental Contribution to ESG (0.4 × E): {:.2f}/100".format(esg_environment_component))

# ==========================================================
# 9️⃣ SAVE CLEAN MERGED DATA
# ==========================================================
env_df.to_csv(r'D:\Projects\AIforGood\data\environment_scores.csv', index=False)
print("\n✅ File saved at: D:\\Projects\\AIforGood\\data\\environment_scores.csv")
