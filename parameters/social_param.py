import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

# ==============================
# 1️⃣ File Configuration
# ==============================
HEALTH_FILE = r'D:\Projects\AIforGood\data\buffalo_health_facilities.csv'
INCOME_FILE = r'D:\Projects\AIforGood\data\neighborhoods_with_zips_and_income.csv'
CRIME_FILE = r'D:\Projects\AIforGood\data\cleaned_crime_incidents.csv'
OUTPUT_FILE = r'D:\Projects\AIforGood\data\social_disparity_scores.csv'
SUMMARY_FILE = r'D:\Projects\AIforGood\data\social_scores.csv'

# Ensure summary file directory exists
os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)

try:
    print("--- Step 1: Loading Data ---")
    df_health = pd.read_csv(HEALTH_FILE)
    df_health['Facility Zip Code'] = df_health['Facility Zip Code'].astype(str).str.slice(0, 5)

    df_income = pd.read_csv(INCOME_FILE)
    df_income['zipcode'] = df_income['zipcode'].astype(str).str.slice(0, 5)
    df_income['Median Income'] = pd.to_numeric(df_income['Median Income'], errors='coerce')

    df_crime = pd.read_csv(CRIME_FILE)
    df_crime['zip_code'] = df_crime['zip_code'].astype(str).str.slice(0, 5)
    df_crime['date'] = pd.to_datetime(df_crime['date'], errors='coerce')
    df_crime = df_crime.dropna(subset=['date'])

    # ==============================
    # 2️⃣ Iterate by Date
    # ==============================
    print("--- Step 2: Generating daily social scores ---")
    all_summaries = []

    for date, group in df_crime.groupby(df_crime['date'].dt.date):
        # Aggregate daily crime counts
        crime_by_zip = group.groupby('zip_code').size().to_frame('Crime_Incident_Count')

        # Aggregate health and income (static by ZIP)
        health_by_zip = df_health.groupby('Facility Zip Code').size().to_frame('Health_Facility_Count')
        income_by_zip = df_income.groupby('zipcode')['Median Income'].mean().to_frame('Avg_Median_Income')

        # Merge all
        df_social = pd.concat([health_by_zip, crime_by_zip, income_by_zip], axis=1).dropna()

        # Invert crime (lower = better)
        df_social['Crime_Score'] = df_social['Crime_Incident_Count'].max() - df_social['Crime_Incident_Count']

        features = ['Health_Facility_Count', 'Crime_Score', 'Avg_Median_Income']
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_social[features])
        df_scaled = pd.DataFrame(df_scaled, columns=features, index=df_social.index)

        # Cluster
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_social['Cluster'] = kmeans.fit_predict(df_scaled)

        centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
        centers['Combined'] = centers.sum(axis=1)
        centers = centers.sort_values('Combined')
        score_map = {centers.index[i]: (i / 2) * 100 for i in range(3)}

        df_social['S_Score_100'] = df_social['Cluster'].map(score_map)
        df_social['S_Score_Scaled'] = df_social['S_Score_100'] * 0.35

        # Compute city-level score for that date
        city_social_score = df_social['S_Score_100'].mean()
        weighted_component = city_social_score * 0.35

        all_summaries.append({
            'Date': date.strftime('%Y-%m-%d'),
            'S_raw_0_100': round(city_social_score, 2),
            'S_weight': 0.35,
            'S_weighted_component': round(weighted_component, 2)
        })

    # ==============================
    # 3️⃣ Save All Summaries
    # ==============================
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"✅ Multi-day Social Summary saved to: {SUMMARY_FILE}")

    print("\n--- PREVIEW ---")
    print(summary_df.head(10))

except Exception as e:
    print(f"⚠️ Error occurred: {e}")
