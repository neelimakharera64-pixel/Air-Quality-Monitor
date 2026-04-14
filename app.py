# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent))

from utils.preprocessor import build_features, get_feature_names
from utils.predictor import load_model, predict_co

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Monitor",
    page_icon="🌿",
    layout="centered",
)

# ── Minimal CSS — completely different look from loan app ──────
st.markdown("""
<style>
    /* Clean white background with green accent */
    .stApp { background-color: #f0f4f0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1b4332;
    }
    [data-testid="stSidebar"] * { color: #d8f3dc !important; }

    /* Cards */
    .info-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
        border-left: 5px solid #40916c;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    /* Result box */
    .result-good     { background:#d8f3dc; border:2px solid #40916c;
                       border-radius:12px; padding:20px; text-align:center; }
    .result-moderate { background:#fff3cd; border:2px solid #e9c46a;
                       border-radius:12px; padding:20px; text-align:center; }
    .result-poor     { background:#ffe0e0; border:2px solid #e63946;
                       border-radius:12px; padding:20px; text-align:center; }
    .result-hazardous{ background:#f4a261; border:2px solid #9d0208;
                       border-radius:12px; padding:20px; text-align:center; }

    /* Section title */
    .sec-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1b4332;
        border-bottom: 2px solid #40916c;
        padding-bottom: 4px;
        margin: 18px 0 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached model load ──────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 AirQuality AI")
    st.markdown("---")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "📡 Predict CO"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset:** UCI Air Quality")
    st.markdown("**Target:** CO concentration (mg/m³)")
    st.markdown("**Best Model:** LightGBM / XGBoost")
    st.caption("Hourly sensor readings · Italy · 2004-2005")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
def show_overview():
    st.markdown("# 🌿 Air Quality CO Prediction")
    st.markdown(
        "This tool uses machine learning to estimate **CO concentration** "
        "(carbon monoxide in mg/m³) from metal-oxide sensor readings "
        "and environmental conditions."
    )

    st.markdown("---")

    # Dataset stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Records",  "~9,357 hourly")
    col2.metric("Features", "23 engineered")
    col3.metric("Location", "Italian city")

    st.markdown("---")

    # What is CO
    st.markdown(
        '<div class="sec-title">What is CO?</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    Carbon monoxide (CO) is a colourless, odourless gas produced by
    incomplete combustion. Major sources include:
    - 🚗 Road traffic exhaust
    - 🏭 Industrial emissions
    - 🔥 Heating systems
    """)

    # Air quality scale
    st.markdown(
        '<div class="sec-title">CO Air Quality Scale</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success("🟢 Good\n\n< 2 mg/m³")
    with c2:
        st.warning("🟡 Moderate\n\n2 – 4 mg/m³")
    with c3:
        st.error("🔴 Poor\n\n4 – 7 mg/m³")
    with c4:
        st.markdown(
            '<div style="background:#f4a261;padding:12px;'
            'border-radius:8px;text-align:center;">'
            '⛔ Hazardous<br><small>> 7 mg/m³</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Feature overview
    st.markdown(
        '<div class="sec-title">Features Used</div>',
        unsafe_allow_html=True,
    )

    feat_df = pd.DataFrame({
        "Feature"     : ["PT08_S1_CO", "PT08_S2_NMHC", "PT08_S3_NOx",
                         "PT08_S4_NO2", "PT08_S5_O3", "T / RH / AH",
                         "Hour_sin/cos", "CO_lag_1h / 2h"],
        "Type"        : ["Sensor", "Sensor", "Sensor",
                         "Sensor", "Sensor", "Environmental",
                         "Cyclic Time", "Lag Feature"],
        "Description" : [
            "Tin oxide — CO proxy",
            "Titania — NMHC proxy",
            "Tungsten oxide — NOx proxy",
            "Tungsten oxide — NO2 proxy",
            "Indium oxide — O3 proxy",
            "Temperature / Humidity",
            "Circular hour encoding",
            "Previous hour CO readings",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Model summary
    st.markdown(
        '<div class="sec-title">Model Performance</div>',
        unsafe_allow_html=True,
    )
    perf_df = pd.DataFrame({
        "Model"   : ["XGBoost", "LightGBM", "CatBoost"],
        "RMSE"    : ["~0.35", "~0.33", "~0.38"],
        "R²"      : ["~0.93", "~0.94", "~0.91"],
        "MAPE %"  : ["~8.5", "~7.9", "~9.2"],
    })
    st.table(perf_df)
    st.caption(
        "Update the values above with your actual training results. "
        "Best model selected by composite RMSE + MAE + R² + MAPE score."
    )


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════
def co_gauge(value: float) -> go.Figure:
    """Needle gauge for CO concentration."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        number= {"suffix": " mg/m³", "font": {"size": 32}},
        title = {"text": "Predicted CO Concentration", "font": {"size": 15}},
        gauge = {
            "axis" : {"range": [0, 10], "tickwidth": 1},
            "bar"  : {"color": "#40916c"},
            "steps": [
                {"range": [0,   2], "color": "#d8f3dc"},
                {"range": [2,   4], "color": "#fff3cd"},
                {"range": [4,   7], "color": "#ffe0e0"},
                {"range": [7,  10], "color": "#f4a261"},
            ],
            "threshold": {
                "line"     : {"color": "#1b4332", "width": 4},
                "thickness": 0.85,
                "value"    : value,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def show_prediction():
    st.markdown("# Predict CO Concentration")
    st.markdown(
        "Enter current sensor readings and conditions. "
        "The model will estimate the CO level in mg/m³."
    )
    st.markdown("---")

    model = get_model()

    with st.form("co_form"):

        # ── Time Context ───────────────────────────────────────
        st.markdown(
            '<div class="sec-title">Time Context</div>',
            unsafe_allow_html=True,
        )
        t1, t2, t3 = st.columns(3)
        with t1:
            hour = st.slider("Hour of Day", 0, 23, 8)
        with t2:
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda m: [
                    "Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"
                ][m-1],
                index=2,
            )
        with t3:
            is_weekend = st.selectbox(
                "Day Type", ["Weekday", "Weekend"]
            )
            season = st.selectbox(
                "Season",
                ["Spring (0)", "Summer (1)", "Autumn (2)", "Winter (3)"],
            )

        # ── Sensor Readings ────────────────────────────────────
        st.markdown(
            '<div class="sec-title">🔬 Metal-Oxide Sensor Readings</div>',
            unsafe_allow_html=True,
        )
        s1, s2 = st.columns(2)
        with s1:
            PT08_S1_CO    = st.number_input("PT08_S1_CO (CO sensor)",
                min_value=500.0,  max_value=2500.0, value=1100.0, step=10.0)
            PT08_S2_NMHC  = st.number_input("PT08_S2_NMHC (NMHC sensor)",
                min_value=500.0,  max_value=2500.0, value=900.0,  step=10.0)
            PT08_S3_NOx   = st.number_input("PT08_S3_NOx (NOx sensor)",
                min_value=200.0,  max_value=2500.0, value=1000.0, step=10.0)
        with s2:
            PT08_S4_NO2   = st.number_input("PT08_S4_NO2 (NO2 sensor)",
                min_value=500.0,  max_value=2500.0, value=1400.0, step=10.0)
            PT08_S5_O3    = st.number_input("PT08_S5_O3 (O3 sensor)",
                min_value=200.0,  max_value=2500.0, value=1000.0, step=10.0)
            C6H6_GT       = st.number_input("Benzene C6H6 (µg/m³)",
                min_value=0.0,    max_value=60.0,   value=5.0,    step=0.1)

        # ── Ground-Truth Pollutants ────────────────────────────
        st.markdown(
            '<div class="sec-title">Pollutant Ground Truth (if known)</div>',
            unsafe_allow_html=True,
        )
        p1, p2 = st.columns(2)
        with p1:
            NOx_GT  = st.number_input("NOx (µg/m³)",
                min_value=0.0, max_value=1200.0, value=200.0, step=5.0)
            NO2_GT  = st.number_input("NO2 (µg/m³)",
                min_value=0.0, max_value=500.0,  value=100.0, step=5.0)
        with p2:
            T  = st.number_input("Temperature (°C)",
                min_value=-10.0, max_value=45.0, value=18.0, step=0.5)
            RH = st.number_input("Relative Humidity (%)",
                min_value=0.0,   max_value=100.0, value=50.0, step=1.0)
            AH = st.number_input("Absolute Humidity",
                min_value=0.0,   max_value=2.5,   value=0.7,  step=0.01)

        # ── Recent CO Readings ─────────────────────────────────
        st.markdown(
            '<div class="sec-title">Recent CO Readings (lag features)</div>',
            unsafe_allow_html=True,
        )
        l1, l2 = st.columns(2)
        with l1:
            co_lag1 = st.number_input(
                "CO 1 hour ago (mg/m³)",
                min_value=0.0, max_value=15.0, value=2.0, step=0.1,
                help="Enter the CO reading from 1 hour ago",
            )
        with l2:
            co_lag2 = st.number_input(
                "CO 2 hours ago (mg/m³)",
                min_value=0.0, max_value=15.0, value=2.0, step=0.1,
                help="Enter the CO reading from 2 hours ago",
            )

        st.markdown("")
        submitted = st.form_submit_button(
            "Estimate CO Level",
            use_container_width=True,
            type="primary",
        )

    # ── Results ────────────────────────────────────────────────
    if submitted:
        season_map = {
            "Spring (0)": 0, "Summer (1)": 1,
            "Autumn (2)": 2, "Winter (3)": 3,
        }
        raw = {
            "PT08_S1_CO"   : PT08_S1_CO,
            "PT08_S2_NMHC" : PT08_S2_NMHC,
            "PT08_S3_NOx"  : PT08_S3_NOx,
            "PT08_S4_NO2"  : PT08_S4_NO2,
            "PT08_S5_O3"   : PT08_S5_O3,
            "C6H6_GT"      : C6H6_GT,
            "NOx_GT"       : NOx_GT,
            "NO2_GT"       : NO2_GT,
            "T"            : T,
            "RH"           : RH,
            "AH"           : AH,
            "hour"         : hour,
            "month"        : month,
            "is_weekend"   : 1 if is_weekend == "Weekend" else 0,
            "season"       : season_map[season],
            "co_lag1"      : co_lag1,
            "co_lag2"      : co_lag2,
        }

        with st.spinner("Running model inference…"):
            df_proc = build_features(raw)
            result  = predict_co(model, df_proc)

        st.markdown("---")
        st.markdown("## Result")

        # ── Gauge ──────────────────────────────────────────────
        st.plotly_chart(co_gauge(result["co_pred"]),
                        use_container_width=True)

        # ── Level badge ────────────────────────────────────────
        level_to_css = {
            "Good"     : "result-good",
            "Moderate" : "result-moderate",
            "Poor"     : "result-poor",
            "Hazardous": "result-hazardous",
        }
        css = level_to_css[result["level"]]
        st.markdown(
            f'<div class="{css}">'
            f'<h2>{result["icon"]}  {result["level"]}</h2>'
            f'<p style="font-size:1.1rem;margin:0;">'
            f'Estimated CO: <strong>{result["co_pred"]} mg/m³</strong></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ── Advice ─────────────────────────────────────────────
        if result["level"] == "Good":
            st.success(f"{result['advice']}")
        elif result["level"] == "Moderate":
            st.warning(f"{result['advice']}")
        else:
            st.error(f"{result['advice']}")

        # ── Input summary ──────────────────────────────────────
        with st.expander("Show Input Summary"):
            summary = pd.DataFrame([{
                "Hour"      : hour,
                "Month"     : month,
                "PT08_S1_CO": PT08_S1_CO,
                "T (°C)"    : T,
                "RH (%)"    : RH,
                "CO_lag_1h" : co_lag1,
                "CO_lag_2h" : co_lag2,
                "Predicted CO (mg/m³)": result["co_pred"],
                "Air Quality Level"   : result["level"],
            }])
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # CORRECT - label is the first argument
            st.download_button(
                label="⬇Download Result",
                data=summary.to_csv(index=False),
                file_name="co_prediction.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    show_overview()
else:
    show_prediction()