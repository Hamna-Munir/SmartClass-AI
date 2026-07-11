import pandas as pd
import os
from datetime import datetime

LOG_PATH = "data/sessions.csv"
COLUMNS  = [
    "timestamp","student_id","emotion","confidence",
    "engagement_score","state","alert","note"
]

def init_log():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(LOG_PATH, index=False)

def log_entry(student_id, emotion, confidence,
              engagement_score, state, alert, note=""):
    try:
        row = pd.DataFrame([{
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id":       str(student_id),
            "emotion":          str(emotion),
            "confidence":       round(float(confidence), 2),
            "engagement_score": int(engagement_score),
            "state":            str(state),
            "alert":            bool(alert),
            "note":             str(note)
        }])
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    except Exception as e:
        print(f"Log error: {e}")

def read_log():
    try:
        if not os.path.exists(LOG_PATH):
            return pd.DataFrame(columns=COLUMNS)
        df = pd.read_csv(LOG_PATH)
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        # Fix column types
        df["engagement_score"] = pd.to_numeric(
            df["engagement_score"], errors="coerce").fillna(0).astype(int)
        df["confidence"]       = pd.to_numeric(
            df["confidence"], errors="coerce").fillna(0.0)
        df["alert"]            = df["alert"].astype(bool)
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

def get_session_summary(df):
    if df is None or df.empty:
        return {
            "total_readings":   0,
            "avg_engagement":   0,
            "dominant_emotion": "N/A",
            "dominant_state":   "N/A",
            "alert_count":      0,
            "duration_min":     0,
        }
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        duration = round(
            (df["timestamp"].max()-df["timestamp"].min()).seconds/60, 1)
    except Exception:
        duration = 0

    return {
        "total_readings":   len(df),
        "avg_engagement":   round(df["engagement_score"].mean(), 1),
        "dominant_emotion": df["emotion"].mode()[0]
                           if not df["emotion"].empty else "N/A",
        "dominant_state":   df["state"].mode()[0]
                           if not df["state"].empty else "N/A",
        "alert_count":      int(df["alert"].sum()),
        "duration_min":     duration,
    }

def clear_log():
    """Clear all session data"""
    pd.DataFrame(columns=COLUMNS).to_csv(LOG_PATH, index=False)