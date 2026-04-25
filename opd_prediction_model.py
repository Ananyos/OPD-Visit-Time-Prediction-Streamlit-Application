"""
=============================================================
  OPD Visit Time Prediction — Machine Learning Model
  Hospital A · Streamlit-ready prediction script
=============================================================
  Model  : Random Forest Regressor
  Target : Total OPD Visit Time (minutes)
  Author : Generated for Hospital A OPD System
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

DATA_PATH   = "opd_mock_data.csv"   # ← update path as needed
MODEL_PATH  = "opd_model.pkl"       # saved model output

# Day-of-week order for consistent encoding
DOW_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# All valid service types used in the app
SERVICE_TYPES = [
    "General Consultation",
    "Lab Services",
    "Minor Procedures",
    "Cardiology",
    "Orthopedics",
    "Dermatology",
    "Pediatrics",
    "Obstetrics & Gynecology",
]

# All valid branches
BRANCHES = ["Branch 1", "Branch 2", "Branch 3"]

# Branch base times (minutes) — used for reference / validation
BRANCH_BASE_TIMES = {
    "Branch 1": 90,   # maps to "Branch A" logic
    "Branch 2": 60,   # maps to "Branch B" logic
    "Branch 3": 120,  # maps to "Branch C" logic
}


# ─────────────────────────────────────────────────────────────
#  STEP 1 · LOAD & INSPECT DATA
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the OPD CSV and show a quick summary."""
    df = pd.read_csv(path)

    print("=" * 55)
    print("  Dataset loaded successfully")
    print("=" * 55)
    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {list(df.columns)}")
    print(f"  Branches  : {sorted(df['Branch'].unique())}")
    print(f"  Services  : {len(df['Service_Type'].unique())} types")
    print(f"  Target avg: {df['Total_OPD_Time_Mins'].mean():.1f} mins")
    print("=" * 55)

    return df


# ─────────────────────────────────────────────────────────────
#  STEP 2 · FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create and encode all model features:
      - day_of_week_num  : Monday=0 … Sunday=6
      - arrival_hour     : integer start hour (e.g. "08:00 - 09:00" → 8)
      - service_encoded  : label-encoded service type
      - branch_encoded   : label-encoded branch name
      - Queue_Size       : already numeric, used as-is
    """
    df = df.copy()

    # Day of week → ordered integer
    df["day_of_week_num"] = df["Day_of_Week"].apply(
        lambda x: DOW_ORDER.index(x) if x in DOW_ORDER else -1
    )

    # Arrival time slot → start hour integer
    # e.g. "08:00 - 09:00" → 8
    df["arrival_hour"] = (
        df["Arrival_Time"]
        .str.split(":")
        .str[0]
        .astype(int)
    )

    # Encode categorical columns
    le_service = LabelEncoder().fit(SERVICE_TYPES)
    le_branch  = LabelEncoder().fit(BRANCHES)

    df["service_encoded"] = le_service.transform(df["Service_Type"])
    df["branch_encoded"]  = le_branch.transform(df["Branch"])

    return df, le_service, le_branch


# ─────────────────────────────────────────────────────────────
#  STEP 3 · TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "day_of_week_num",   # Mon–Sun (0–6)
    "arrival_hour",      # 8, 9, 10 … 16
    "service_encoded",   # 0–7
    "branch_encoded",    # 0–2
    "Queue_Size",        # 1–15+
]

def split_data(df: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test."""
    X = df[FEATURE_COLS]
    y = df["Total_OPD_Time_Mins"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ─────────────────────────────────────────────────────────────
#  STEP 4 · TRAIN MODEL & EVALUATE
# ─────────────────────────────────────────────────────────────

def train_model(X_train, y_train) -> RandomForestRegressor:
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=100,   # 100 decision trees
        max_depth=None,     # let trees grow fully
        random_state=42,    # reproducible results
        n_jobs=-1,          # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print MAE and RMSE, and show feature importances."""
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print("\n  ── Model Performance ───────────────────────────")
    print(f"  MAE  (Mean Absolute Error) : {mae:.2f}  minutes")
    print(f"  RMSE (Root Mean Sq Error)  : {rmse:.2f}  minutes")
    print(f"  Interpretation: predictions are off by ~{mae:.0f} mins on average")

    print("\n  ── Feature Importances ─────────────────────────")
    for feat, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<20} {imp:.3f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────
#  STEP 5 · SAVE MODEL TO DISK (for Streamlit integration)
# ─────────────────────────────────────────────────────────────

def save_model(model, le_service, le_branch, path: str = MODEL_PATH):
    """Pickle the model + encoders so Streamlit can load them."""
    payload = {
        "model":      model,
        "le_service": le_service,
        "le_branch":  le_branch,
        "features":   FEATURE_COLS,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  ✅  Model saved to: {path}")


def load_model(path: str = MODEL_PATH):
    """Load saved model + encoders from disk."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["le_service"], payload["le_branch"]


# ─────────────────────────────────────────────────────────────
#  PREDICTION FUNCTION  ← call this from Streamlit
# ─────────────────────────────────────────────────────────────

def predict_opd_time(
    day_of_week: str,
    arrival_time_slot: str,
    service_type: str,
    branch: str,
    queue_size: int,
    model=None,
    le_service=None,
    le_branch=None,
) -> dict:
    """
    Predict the total OPD visit time for a single patient.

    Parameters
    ----------
    day_of_week        : e.g. "Monday", "Tuesday" … "Sunday"
    arrival_time_slot  : e.g. "08:00 - 09:00"
    service_type       : one of SERVICE_TYPES list
    branch             : "Branch 1", "Branch 2", or "Branch 3"
    queue_size         : integer (expected queue length at arrival)
    model / le_*       : pass pre-loaded objects, or leave None to
                         auto-load from MODEL_PATH

    Returns
    -------
    dict with:
      predicted_minutes  (int)
      predicted_display  (str)  e.g. "2 hr 15 mins"
      branch_base_time   (int)  base time for that branch
      queue_contribution (int)  estimated extra time from queue
      input_summary      (dict) echo of inputs for UI display
    """

    # ── Auto-load model if not passed in ──────────────────
    if model is None:
        model, le_service, le_branch = load_model()

    # ── Validate inputs ───────────────────────────────────
    assert day_of_week in DOW_ORDER,      f"Invalid day: {day_of_week}"
    assert service_type in SERVICE_TYPES, f"Invalid service: {service_type}"
    assert branch in BRANCHES,            f"Invalid branch: {branch}"
    assert 0 <= queue_size <= 50,         f"Queue size out of range: {queue_size}"

    # ── Build feature row ─────────────────────────────────
    day_num      = DOW_ORDER.index(day_of_week)
    arrival_hour = int(arrival_time_slot.split(":")[0])
    svc_enc      = le_service.transform([service_type])[0]
    branch_enc   = le_branch.transform([branch])[0]

    X_input = pd.DataFrame([[
        day_num,
        arrival_hour,
        svc_enc,
        branch_enc,
        queue_size,
    ]], columns=FEATURE_COLS)

    # ── Predict ───────────────────────────────────────────
    predicted_mins = int(round(model.predict(X_input)[0]))

    # ── Format for display ────────────────────────────────
    hours   = predicted_mins // 60
    minutes = predicted_mins  % 60
    display = f"{hours} hr {minutes:02d} mins" if hours > 0 else f"{minutes} mins"

    base        = BRANCH_BASE_TIMES[branch]
    queue_extra = max(0, predicted_mins - base)

    return {
        "predicted_minutes":  predicted_mins,
        "predicted_display":  display,
        "branch_base_time":   base,
        "queue_contribution": queue_extra,
        "input_summary": {
            "day_of_week":       day_of_week,
            "arrival_time_slot": arrival_time_slot,
            "service_type":      service_type,
            "branch":            branch,
            "queue_size":        queue_size,
        },
    }


# ─────────────────────────────────────────────────────────────
#  MAIN — run this script directly to train & save the model
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  OPD Visit Time Prediction — Training Pipeline")
    print("=" * 55 + "\n")

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Feature engineering
    df, le_service, le_branch = engineer_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # 4. Train
    print("\n  Training Random Forest … ", end="", flush=True)
    model = train_model(X_train, y_train)
    print("done ✓")

    # 5. Evaluate
    evaluate_model(model, X_test, y_test)

    # 6. Save
    save_model(model, le_service, le_branch)

    # ── Demo: run a sample prediction ─────────────────────
    print("\n  ── Sample Prediction ───────────────────────────")
    result = predict_opd_time(
        day_of_week       = "Wednesday",
        arrival_time_slot = "08:00 - 09:00",
        service_type      = "Cardiology",
        branch            = "Branch 1",
        queue_size        = 5,
        model             = model,
        le_service        = le_service,
        le_branch         = le_branch,
    )
    print(f"  Input   : {result['input_summary']}")
    print(f"  Output  : {result['predicted_display']}  ({result['predicted_minutes']} mins)")
    print(f"  Base time (Branch 1) : {result['branch_base_time']} mins")
    print(f"  Queue adds approx    : {result['queue_contribution']} mins")
    print("\n  ✅  All done. Ready for Streamlit integration.\n")