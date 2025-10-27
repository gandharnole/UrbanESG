"""
gov_param.py
Computes governance score (0–100) and ESG composite score.

Data inputs:
- 311_Service_Requests_cleaned.csv  → citizen complaints dataset
- cleaned_data_for_eda.csv          → budget / public finance dataset

Outputs:
- governance_scores.csv
- esg_scores.csv
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# --- Optional sentiment analysis (safe import)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    print("✅ NLTK VADER loaded for sentiment analysis.")
except Exception as e:
    print("⚠️  NLTK VADER unavailable; sentiment will be skipped.", e)
    sia = None

# --- File paths (✅ FIXED)
SERVICE_FILE = r"D:\Projects\AIforGood\data\311_Service_Requests_cleaned.csv"
BUDGET_FILE  = r"D:\Projects\AIforGood\data\cleaned_data_for_eda.csv"

# --- Check file existence
assert os.path.exists(SERVICE_FILE), f"Service file not found: {SERVICE_FILE}"
assert os.path.exists(BUDGET_FILE), f"Budget file not found: {BUDGET_FILE}"

# --- Load datasets
svc = pd.read_csv(SERVICE_FILE, low_memory=False)
bud = pd.read_csv(BUDGET_FILE, low_memory=False)

# --- Detect grouping column automatically
possible_keys = [
    'district','zone','borough','ward','neighborhood',
    'area','city','region','location','zip_code','zipcode'
]
GROUP_COL = None
for key in possible_keys:
    if key in svc.columns:
        GROUP_COL = key
        break

if GROUP_COL is None:
    print("⚠️  No location column found in 311 data. Grouping all records together.")
    svc['dummy_group'] = 'all'
    GROUP_COL = 'dummy_group'

if GROUP_COL not in bud.columns:
    print(f"⚠️  Adding '{GROUP_COL}' column to budget dataset for consistency.")
    bud[GROUP_COL] = 'all'

print(f"📍 Using group key: {GROUP_COL}")

# --- Helper functions
def compute_resolution_rate(df, groupby_col):
    if 'status' in df.columns:
        closed = df[df['status'].str.lower().isin(['closed','resolved','completed','closed - duplicate'])] \
                    .groupby(groupby_col).size()
    elif 'closed_date' in df.columns:
        closed = df[df['closed_date'].notna()].groupby(groupby_col).size()
    else:
        closed = pd.Series(dtype=float)
    total = df.groupby(groupby_col).size()
    rate = (closed / total).fillna(0).rename('complaint_resolution_rate')
    return rate.reset_index()

def compute_avg_resolution_time(df, groupby_col):
    if 'created_date' in df.columns and 'closed_date' in df.columns:
        created = pd.to_datetime(df['created_date'], errors='coerce')
        closed  = pd.to_datetime(df['closed_date'], errors='coerce')
        df['_res_days'] = (closed - created).dt.total_seconds() / (3600*24)
        avg = df.groupby(groupby_col)['_res_days'].mean().rename('avg_resolution_days')
        return avg.reset_index()
    return pd.DataFrame(columns=[groupby_col, 'avg_resolution_days'])

def compute_sentiment_from_texts(df, text_col, groupby_col):
    if sia is None or text_col not in df.columns:
        return pd.DataFrame(columns=[groupby_col, 'complaint_sentiment'])
    df = df.copy()
    df[text_col] = df[text_col].fillna("")
    df['_sent'] = df[text_col].apply(lambda t: sia.polarity_scores(str(t))['compound'])
    mean_sent = df.groupby(groupby_col)['_sent'].mean().rename('complaint_sentiment')
    return mean_sent.reset_index()

def compute_budget_metrics(bud_df, groupby_col):
    df = bud_df.copy()
    candidates = {
        'budget_allocated': ['budget_allocated','allocated_amount','budget'],
        'budget_spent': ['budget_spent','spent_amount','expenditure']
    }

    for k, opts in candidates.items():
        for o in opts:
            if o in df.columns:
                df[k] = pd.to_numeric(df[o], errors='coerce')
                break
        if k not in df.columns:
            df[k] = np.nan

    df['util_ratio'] = df['budget_spent'] / df['budget_allocated']
    df['util_ratio'] = df['util_ratio'].replace([np.inf, -np.inf], np.nan)
    agg = df.groupby(groupby_col, dropna=False)['util_ratio'].mean().fillna(0).rename('avg_budget_utilization')
    return agg.reset_index()

# --- Compute indicators
res_rate = compute_resolution_rate(svc, GROUP_COL)
res_time = compute_avg_resolution_time(svc, GROUP_COL)
sent = compute_sentiment_from_texts(svc, text_col='description', groupby_col=GROUP_COL)
budget = compute_budget_metrics(bud, GROUP_COL)

from functools import reduce
gov = reduce(lambda l, r: pd.merge(l, r, on=GROUP_COL, how='outer'), [res_rate, res_time, sent, budget]).fillna(0)

# --- Normalize and compute governance score
gov['complaint_resolution_rate'] = gov['complaint_resolution_rate'].clip(0,1)
gov['avg_resolution_days'] = gov['avg_resolution_days'].replace(0,np.nan)
MAX_DAYS = 60.0
gov['res_time_score'] = 1 - (gov['avg_resolution_days'].clip(0,MAX_DAYS) / MAX_DAYS)
gov['res_time_score'] = gov['res_time_score'].fillna(0).clip(0,1)

gov['sent_norm'] = (gov['complaint_sentiment'] + 1) / 2 if 'complaint_sentiment' in gov.columns else 0.5
gov['util_score'] = 1 - abs(gov['avg_budget_utilization'] - 1)
gov['util_score'] = gov['util_score'].clip(0,1)

weights = {'resolution':0.35,'sentiment':0.15,'time':0.25,'budget':0.25}
gov['governance_score_01'] = (
    gov['complaint_resolution_rate']*weights['resolution'] +
    gov['sent_norm']*weights['sentiment'] +
    gov['res_time_score']*weights['time'] +
    gov['util_score']*weights['budget']
)
gov['governance_score'] = (gov['governance_score_01'] * 100).round(2)

# --- Save governance scores
OUT1 = r"D:\Projects\AIforGood\AIforGood_Governance\governance_scores.csv"
gov.to_csv(OUT1, index=False)
print(f"✅ Governance scores saved to: {OUT1}")

# --- Compute final governance summary (no ESG calculation here) ---
gov['governance_score_scaled'] = (gov['governance_score'] * 0.25).round(2)

OUT_SUMMARY = r"D:\Projects\AIforGood\AIforGood_Governance\governance_summary.csv"
gov[['governance_score', 'governance_score_scaled']].to_csv(OUT_SUMMARY, index=False)

print(f"✅ Governance summary saved to: {OUT_SUMMARY}")
print("\n🎯 Done — governance scores generated successfully (ESG handled separately).")

