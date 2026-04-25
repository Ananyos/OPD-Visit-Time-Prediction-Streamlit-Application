"""
=============================================================
  Hospital A · OPD Visit Time Prediction — Streamlit App
  UI/UX Update — Fix Summary:
    [FIX 1] Removed empty white box below header
    [FIX 2] Fixed blank selectbox text in dark/light mode
    [FIX 3] Matched result card style with input boxes
    [FIX 4] Added distance filter with slider
    [FIX 5] Centered the predict button horizontally
=============================================================
  Run with:  streamlit run streamlit_app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import random
from datetime import date

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital A · OPD Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS  (unchanged)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "opd_model.pkl"
DATA_PATH  = "opd_mock_data.csv"

TIME_SLOTS = [
    "08:00 - 09:00", "09:00 - 10:00", "10:00 - 11:00",
    "11:00 - 12:00", "12:00 - 13:00", "13:00 - 14:00",
    "14:00 - 15:00", "15:00 - 16:00", "16:00 - 17:00",
]

SERVICE_TYPES = [
    "General Consultation", "Lab Services", "Minor Procedures",
    "Cardiology", "Orthopedics", "Dermatology",
    "Pediatrics", "Obstetrics & Gynecology",
]

BRANCHES = ["Branch 1", "Branch 2", "Branch 3"]

BRANCH_BASE_TIMES = {
    "Branch 1": 90,
    "Branch 2": 60,
    "Branch 3": 120,
}

DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
#  [FIX 1] header uses display:none + height:0 so no blank space
#  [FIX 1] block-container padding-top set to 0
#  [FIX 2] Selectbox: only border-radius/border overridden,
#           NOT background/color, so Streamlit theme controls
#           text visibility in both light and dark mode
#  [FIX 3] branch-card uses rgba transparent bg that adapts
#           to both light and dark themes (matches input style)
#  [FIX 5] Button width:100% inside a narrow center column
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* [FIX 1] Remove all reserved space for Streamlit chrome */
#MainMenu  { visibility: hidden; height: 0; overflow: hidden; }
footer     { visibility: hidden; height: 0; overflow: hidden; }
header     { visibility: hidden; height: 0 !important; min-height: 0 !important; }

/* [FIX 1] Zero out the top padding that Streamlit adds below
   the (now-hidden) header — this was causing the blank box */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 440px !important;
    margin: 0 auto !important;
}

/* [FIX 1] Also clear any top margin on the root app element */
.stApp > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* ── App bar ── */
.app-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px 14px;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 4px;
}
.hamburger  { font-size: 22px; cursor: pointer; }
.logo-circle {
    width: 40px; height: 40px;
    background: #e8f5e9;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    border: 2px solid #1DB954;
    flex-shrink: 0;
}
.logo-cross { font-size: 20px; color: #e53935; font-weight: 700; line-height: 1; }
.app-title  { font-size: 17px; font-weight: 600; }

/* ── Heading ── */
.card-heading {
    font-size: 18px;
    font-weight: 700;
    color: #1DB954;
    text-align: center;
    margin: 18px 0 4px;
    line-height: 1.35;
}
.card-sub {
    font-size: 12px;
    color: #888;
    text-align: center;
    margin-bottom: 12px;
}

/* ── Field labels ── */
.field-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #888;
    text-transform: uppercase;
    margin: 8px 0 2px;
}

/* [FIX 2] Date input — shape only, no color override */
.stDateInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid rgba(128,128,128,0.25) !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* [FIX 2] Selectbox outer wrapper — shape only */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid rgba(128,128,128,0.25) !important;
}
/* [FIX 2] Inner BaseWeb select — remove its own border so only
   our wrapper border shows, preventing a double-border effect */
.stSelectbox [data-baseweb="select"] > div:first-child {
    border-radius: 10px !important;
    border: none !important;
    background: transparent !important;
}
/* [FIX 2] Force the selected value text to be visible.
   opacity:1 cancels any ghost/dim state Streamlit applies. */
.stSelectbox [data-baseweb="select"] [data-baseweb="value"],
.stSelectbox [data-baseweb="select"] [data-baseweb="placeholder"],
.stSelectbox [data-baseweb="select"] span {
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    opacity: 1 !important;
}

/* [FIX 5] Predict button — full width within its (narrow) column */
.stButton > button {
    background: #1DB954 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 25px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 10px 28px !important;
    width: 100% !important;
    margin-top: 10px !important;
    text-transform: uppercase;
    transition: background 0.2s;
}
.stButton > button:hover  { background: #17a349 !important; }
.stButton > button:active { background: #128a3a !important; }

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.15);
    margin: 14px 0 12px;
}

/* ── Result header ── */
.result-title { font-size: 17px; font-weight: 700; color: #1DB954; margin-bottom: 8px; }

/* [FIX 3] Branch card — transparent background adapts to both
   light and dark themes, matching how Streamlit input boxes look */
.branch-card {
    border-radius: 12px;
    padding: 13px 16px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1.5px solid rgba(128,128,128,0.2);
    background: rgba(128,128,128,0.07);
}
.branch-card.out-of-range {
    opacity: 0.4;
    border-style: dashed;
    border-color: rgba(128,128,128,0.15);
}
.branch-name {
    font-size: 14px;
    font-weight: 600;
    color: #1DB954;
}
.branch-info {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 5px;
}
.time-row, .dist-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
}
.icon-clock { opacity: 0.7; }
.icon-pin   { color: #e53935; }

/* ── Badges ── */
.badge-fastest {
    background: #e8f5e9; color: #1a7a38;
    font-size: 10px; font-weight: 600;
    padding: 2px 7px; border-radius: 20px;
    letter-spacing: 0.04em; margin-left: 6px;
}
.badge-closest {
    background: #e3f0fb; color: #185FA5;
    font-size: 10px; font-weight: 600;
    padding: 2px 7px; border-radius: 20px;
    letter-spacing: 0.04em; margin-left: 6px;
}
.badge-filtered {
    background: rgba(128,128,128,0.12); color: #888;
    font-size: 10px; font-weight: 500;
    padding: 2px 7px; border-radius: 20px;
    margin-left: 6px;
}

/* ── Empty state ── */
.result-placeholder {
    font-size: 13px;
    color: #aaa;
    text-align: center;
    padding: 24px 0 16px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MODEL LOADER  (backend unchanged)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on first run…")
def get_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            p = pickle.load(f)
        return p["model"], p["le_service"], p["le_branch"]

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)
    DOW_MAP = {d: i for i, d in enumerate(DOW_ORDER)}
    df["day_of_week_num"] = df["Day_of_Week"].map(DOW_MAP)
    df["arrival_hour"]    = df["Arrival_Time"].str.split(":").str[0].astype(int)

    le_service = LabelEncoder().fit(SERVICE_TYPES)
    le_branch  = LabelEncoder().fit(BRANCHES)
    df["service_encoded"] = le_service.transform(df["Service_Type"])
    df["branch_encoded"]  = le_branch.transform(df["Branch"])

    features = ["day_of_week_num","arrival_hour","service_encoded","branch_encoded","Queue_Size"]
    X, y = df[features], df["Total_OPD_Time_Mins"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "le_service": le_service,
                     "le_branch": le_branch, "features": features}, f)
    return model, le_service, le_branch


def predict_for_branch(model, le_service, le_branch, dow, hour, service, branch, queue):
    """Prediction logic unchanged."""
    FEATURE_COLS = ["day_of_week_num","arrival_hour","service_encoded","branch_encoded","Queue_Size"]
    X = pd.DataFrame([[
        DOW_ORDER.index(dow), hour,
        le_service.transform([service])[0],
        le_branch.transform([branch])[0],
        queue,
    ]], columns=FEATURE_COLS)
    return int(round(model.predict(X)[0]))


def fmt_time(mins):
    """Time formatting unchanged."""
    h, m = divmod(mins, 60)
    return f"{h} hr {m:02d} mins" if h > 0 else f"{m} mins"


# ─────────────────────────────────────────────────────────────
#  SESSION STATE — keeps results alive when slider reruns page
# ─────────────────────────────────────────────────────────────
if "results"    not in st.session_state: st.session_state.results    = []
if "queue_size" not in st.session_state: st.session_state.queue_size = None

model, le_service, le_branch = get_model()


# ═════════════════════════════════════════════════════════════
#  APP BAR
#  [FIX 1] No st.container() — direct markdown, no wrapper box
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-bar">
    <span class="hamburger">☰</span>
    <div class="logo-circle"><span class="logo-cross">✚</span></div>
    <span class="app-title">Hospital A</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE HEADING
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="card-heading">Total OPD Visit Time<br>Prediction</div>
<div class="card-sub">Expected Arrival Time</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  INPUT FIELDS
# ═════════════════════════════════════════════════════════════

# Row 1: Date + Time side by side
col_date, col_time = st.columns(2)

with col_date:
    st.markdown('<p class="field-label">Date</p>', unsafe_allow_html=True)
    selected_date = st.date_input(
        label="Date",
        value=date.today(),
        label_visibility="collapsed",
        key="date_input",
    )

with col_time:
    # [FIX 2] No background/color forced — Streamlit theme handles it
    st.markdown('<p class="field-label">Time</p>', unsafe_allow_html=True)
    selected_time = st.selectbox(
        label="Time",
        options=TIME_SLOTS,
        index=0,
        label_visibility="collapsed",
        key="time_input",
    )

# Row 2: Service full width
# [FIX 2] Same — shape-only CSS override
st.markdown('<p class="field-label">Service</p>', unsafe_allow_html=True)
selected_service = st.selectbox(
    label="Service",
    options=SERVICE_TYPES,
    index=0,
    label_visibility="collapsed",
    key="service_input",
)

# [FIX 5] Center the button: narrow center column flanked by spacers
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("SEE PREDICT TIME", key="predict_btn")


# ═════════════════════════════════════════════════════════════
#  PREDICTION  (backend logic unchanged)
# ═════════════════════════════════════════════════════════════
if predict_clicked:
    dow        = selected_date.strftime("%A")
    hour       = int(selected_time.split(":")[0])
    queue_size = random.randint(1, 10)

    results = []
    for branch in BRANCHES:
        pred_mins = predict_for_branch(
            model, le_service, le_branch,
            dow, hour, selected_service, branch, queue_size,
        )
        dist_km = round(random.uniform(5.0, 20.0), 2)
        results.append({
            "branch":  branch,
            "mins":    pred_mins,
            "display": fmt_time(pred_mins),
            "dist_km": dist_km,
        })

    results.sort(key=lambda x: x["mins"])     # default: fastest first
    st.session_state.results    = results
    st.session_state.queue_size = queue_size


# ═════════════════════════════════════════════════════════════
#  RESULTS SECTION
# ═════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

if not st.session_state.results:
    st.markdown("""
    <div class="result-placeholder">
        Fill in the form above and tap<br>
        <strong>SEE PREDICT TIME</strong>
    </div>
    """, unsafe_allow_html=True)

else:
    results    = st.session_state.results
    queue_size = st.session_state.queue_size

    st.markdown('<div class="result-title">Result</div>', unsafe_allow_html=True)

    # ── [FIX 4] FILTER CONTROLS ──────────────────────────────
    # Sort toggles in a two-column row (clean, compact)
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_by_time = st.checkbox("⏱ Sort by time",     value=True,  key="sort_time")
    with sort_col2:
        sort_by_dist = st.checkbox("📍 Sort by distance", value=False, key="sort_dist")

    # [FIX 4] Distance range slider — filters out-of-range branches
    all_distances = [r["dist_km"] for r in results]
    slider_min = max(0.0, float(int(min(all_distances))))
    slider_max = float(int(max(all_distances)) + 1)

    max_dist_filter = st.slider(
        label="Max distance",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_max,       # default: all branches visible
        step=0.5,
        format="%.1f KM",
        key="dist_slider",
    )

    # Apply sort (time takes precedence if both checked)
    sorted_results = list(results)
    if sort_by_dist and not sort_by_time:
        sorted_results.sort(key=lambda x: x["dist_km"])
    elif sort_by_time:
        sorted_results.sort(key=lambda x: x["mins"])

    # Identify top picks within the distance limit
    in_range       = [r for r in sorted_results if r["dist_km"] <= max_dist_filter]
    fastest_branch = in_range[0]["branch"]                                   if in_range else None
    closest_branch = min(in_range, key=lambda x: x["dist_km"])["branch"]    if in_range else None

    # ── [FIX 3] Render cards with theme-adaptive styling ─────
    cards_html = ""
    for r in sorted_results:
        within_range = r["dist_km"] <= max_dist_filter

        # Badge logic
        badges = ""
        if within_range:
            if sort_by_dist and not sort_by_time and r["branch"] == closest_branch:
                badges += '<span class="badge-closest">Closest</span>'
            elif r["branch"] == fastest_branch:
                badges += '<span class="badge-fastest">Fastest</span>'
        else:
            badges += '<span class="badge-filtered">Out of range</span>'

        card_class = "branch-card" + (" out-of-range" if not within_range else "")

        cards_html += f"""
        <div class="{card_class}">
            <div class="branch-name">{r['branch']}{badges}</div>
            <div class="branch-info">
                <div class="time-row">
                    <span class="icon-clock">⏱</span>
                    <span>{r['display']}</span>
                </div>
                <div class="dist-row">
                    <span class="icon-pin">📍</span>
                    <span>{r['dist_km']} KM</span>
                </div>
            </div>
        </div>
        """

    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Prediction details expander (logic unchanged) ────────
    dow = selected_date.strftime("%A")
    with st.expander("ℹ️ Prediction details", expanded=False):
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | Date | {selected_date.strftime('%d/%m/%Y')} ({dow}) |
        | Time slot | {selected_time} |
        | Service | {selected_service} |
        | Simulated queue | {queue_size} patients |
        """)
        st.caption("Distance is simulated. Queue size is randomised for demo purposes.")