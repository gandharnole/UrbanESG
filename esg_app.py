import os
import io
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = "gemini-flash-latest"

GEMINI_OK = False
try:
    import google.generativeai as genai
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False


def call_gemini(prompt: str, api_key: str, model_name: str) -> str:
    """Calls Gemini and returns the clean response text."""
    if not GEMINI_OK:
        raise RuntimeError("google-generativeai not installed.")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing in .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    if hasattr(response, "text") and response.text:
        return response.text.strip()
    elif hasattr(response, "candidates") and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    else:
        return "⚠️ No text output from Gemini model."


# ---------- Streamlit UI Setup ----------
st.set_page_config(page_title="Buffalo ESG Dashboard", layout="wide")

st.markdown("""
<style>
.kpi-card {border:1px solid #eee; border-radius:14px; padding:14px 16px; background:#fff;}
.kpi-title {font-size:0.85rem; color:#666; margin-bottom:6px;}
.kpi-value {font-size:1.7rem; font-weight:700; margin-bottom:2px;}
.kpi-sub {font-size:0.8rem; color:#888;}
.section {margin-top: 1.0rem; margin-bottom: 0.4rem;}
hr {margin: 0.6rem 0 1.0rem 0;}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("📁 Data sources")
default_base = r"D:\Projects\AIforGood\data"
env_path = st.sidebar.text_input("Environment CSV", os.path.join(default_base, "environment_scores.csv"))
soc_path = st.sidebar.text_input("Social CSV", os.path.join(default_base, "social_scores.csv"))
gov_path = st.sidebar.text_input("Governance CSV", os.path.join(default_base, "governance_scores.csv"))
s_zip_path = st.sidebar.text_input("Social per-ZIP CSV (optional)", os.path.join(default_base, "social_disparity_scores.csv"))

st.sidebar.markdown("---")
st.sidebar.header("🧠 Gemini API")
if api_key:
    st.sidebar.success("✅ API key loaded securely from .env file")
else:
    st.sidebar.warning("⚠️ GOOGLE_API_KEY missing — heuristic mode will be used.")
st.sidebar.write(f"**Model:** {model_name}")

# ---------- Loaders ----------
@st.cache_data(show_spinner=True)
def normalize_date_column(df: pd.DataFrame, possible_names=None) -> pd.Series:
    if possible_names is None:
        possible_names = ['date', 'Date', 'timestamp', 'Timestamp', 'Created Date']
    for col in possible_names:
        if col in df.columns:
            return pd.to_datetime(df[col], errors='coerce').dt.date
    return pd.Series(pd.date_range(end=datetime.now(), periods=len(df)).date, index=df.index)

@st.cache_data(show_spinner=True)
def load_env(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df['date'] = normalize_date_column(df)
    if 'daily_EIS' not in df.columns and 'E_raw' in df.columns:
        df['daily_EIS'] = df['E_raw']
    elif 'daily_EIS' not in df.columns:
        raise ValueError("Environment file must contain 'daily_EIS' or 'E_raw'.")
    return df.dropna(subset=['date'])

@st.cache_data(show_spinner=True)
def load_soc(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df['date'] = normalize_date_column(df)
    if 'S_raw_0_100' not in df.columns:
        if {'S_weighted_component','S_weight'}.issubset(df.columns):
            df['S_raw_0_100'] = (df['S_weighted_component'] / df['S_weight']).clip(0, 100)
        else:
            raise ValueError("Social file missing 'S_raw_0_100'.")
    return df.dropna(subset=['date'])

@st.cache_data(show_spinner=True)
def load_gov(fp: str):
    df = pd.read_csv(fp)
    g_mean = float(pd.to_numeric(df['governance_score'], errors='coerce').dropna().mean())
    df['date'] = pd.date_range(end=datetime.now(), periods=len(df)).date
    return df, g_mean


# ---------- Read files ----------
with st.spinner("Loading data..."):
    try:
        env_df = load_env(env_path)
        soc_df = load_soc(soc_path)
        gov_zip_df, g_city_mean = load_gov(gov_path)
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()


# ---------- Show ESG Columns ----------
st.markdown("### 🧾 Columns Used for ESG Calculation")
col_e, col_s, col_g = st.columns(3)
with col_e:
    st.markdown("**Environment:**")
    st.write("`avg_aqi`, `CO2_emission_tons`, `NOx_emission_tons`, `SO2_emission_tons`, `green_cover_percent`, `tree_density_per_sqkm`, `avg_surface_temp_C`")
with col_s:
    st.markdown("**Social:**")
    st.write("`Health_Facility_Count`, `Crime_Incident_Count`, `Avg_Median_Income`")
with col_g:
    st.markdown("**Governance:**")
    st.write("`complaint_resolution_rate`, `avg_resolution_days`, `avg_budget_utilization`, `governance_score`")
st.markdown("---")


# ---------- Merge ESG ----------
merged = pd.merge(
    env_df[['date', 'daily_EIS']].rename(columns={'daily_EIS': 'E_raw'}),
    soc_df[['date', 'S_raw_0_100']].rename(columns={'S_raw_0_100': 'S_raw'}),
    on='date', how='inner'
)
merged['G_raw'] = g_city_mean
E_W, S_W, G_W = 0.40, 0.35, 0.25
merged['ESG'] = (merged['E_raw']*E_W + merged['S_raw']*S_W + merged['G_raw']*G_W).round(2)


# ---------- Date Filter ----------
min_d, max_d = merged['date'].min(), merged['date'].max()
colA, colB = st.columns(2)
with colA:
    start_date = st.date_input("Start", min_d, min_value=min_d, max_value=max_d)
with colB:
    end_date = st.date_input("End", max_d, min_value=min_d, max_value=max_d)

view = merged[(merged['date'] >= start_date) & (merged['date'] <= end_date)]
if view.empty:
    st.warning("No rows found in selected date range.")
    st.stop()


# ---------- KPI Cards ----------
latest_row = view.sort_values('date').iloc[-1]
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Environment (E)</div><div class='kpi-value'>{latest_row.E_raw:.1f}</div><div class='kpi-sub'>Weight 0.40</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Social (S)</div><div class='kpi-value'>{latest_row.S_raw:.1f}</div><div class='kpi-sub'>Weight 0.35</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Governance (G)</div><div class='kpi-value'>{latest_row.G_raw:.1f}</div><div class='kpi-sub'>Weight 0.25</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>ESG Score</div><div class='kpi-value'>{latest_row.ESG:.1f}</div><div class='kpi-sub'>{latest_row.date}</div></div>", unsafe_allow_html=True)


# ---------- ESG Trends ----------
st.subheader("📈 ESG Trends")
base = alt.Chart(view).encode(x='date:T')
chart = (
    base.mark_line(color='#1f77b4').encode(y='E_raw:Q') +
    base.mark_line(color='#2ca02c').encode(y='S_raw:Q') +
    base.mark_line(color='#9467bd').encode(y='G_raw:Q') +
    base.mark_line(color='#ff7f0e').encode(y='ESG:Q')
)
st.altair_chart(chart.properties(height=320), use_container_width=True)


# ---------- Focus Area Overview ----------
st.markdown("---")
st.subheader("🧭 ESG Focus Areas")

means = {
    'Environment (E)': view['E_raw'].mean(),
    'Social (S)': view['S_raw'].mean(),
    'Governance (G)': view['G_raw'].mean()
}

def category(score):
    if score >= 70:
        return "🟢 Strong"
    elif score >= 50:
        return "🟡 Moderate"
    else:
        return "🔴 Needs Attention"

cols = st.columns(3)
for (pillar, score), col in zip(means.items(), cols):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{pillar}</div>
            <div class="kpi-value">{score:.2f}</div>
            <div class="kpi-sub">{category(score)}</div>
        </div>
        """, unsafe_allow_html=True)

# Identify pillars that need attention or are moderate
focus_pillars = [p for p, s in means.items() if s < 70]


# ---------- Gemini Prompt Builder ----------
def build_prompt(pillar, esg_df):
    period = f"{start_date} to {end_date}"
    data_preview = esg_df[['date','E_raw','S_raw','G_raw','ESG']].tail(14).to_csv(index=False)
    return f"""
You are an ESG policy advisor for Buffalo, NY.
Focus Pillar: {pillar}
Time Period: {period}

Recent ESG snapshot:
{data_preview}

Task:
Provide a concise 3–4 paragraph recommendation plan to improve the {pillar} pillar.
Clearly mention:
1. Key issues or patterns observed.
2. Top 3 targeted actions Buffalo can take.
3. Expected impact on the {pillar} score.
Keep it direct, data-linked, and actionable.
"""


# ---------- AI-Powered Recommendations ----------
st.markdown("---")
st.subheader("🤖 AI-Powered Recommendations")

if st.button("Generate Recommendations", use_container_width=True):
    if not focus_pillars:
        st.info("All pillars are strong! No improvement suggestions needed.")
    else:
        for pillar in focus_pillars:
            st.markdown(f"### 🔹 Recommendations for {pillar}")
            try:
                prompt = build_prompt(pillar, view)
                with st.expander(f"Prompt for {pillar}", expanded=False):
                    st.code(prompt, language="markdown")

                if GEMINI_OK and api_key:
                    recs = call_gemini(prompt, api_key, model_name)
                    st.success(f"✅ Insights for {pillar}")
                    st.markdown(recs)
                else:
                    st.info("Gemini key missing — showing heuristic fallback.")
                    if "Environment" in pillar:
                        st.markdown("- Expand urban tree cover & add EV charging hubs in industrial corridors.")
                    elif "Social" in pillar:
                        st.markdown("- Improve public safety lighting, healthcare access, and local job fairs.")
                    else:
                        st.markdown("- Reduce 311 delays, automate ticket routing, and publish governance dashboards.")
            except Exception as e:
                st.error(f"Gemini error for {pillar}: {e}")


# ---------- Download ----------
st.markdown("---")
csv_buf = io.StringIO()
view.to_csv(csv_buf, index=False)
st.download_button("📥 Download ESG Slice (CSV)", csv_buf.getvalue(), file_name="esg_slice.csv", mime="text/csv")
