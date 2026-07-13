"""Streamlit dashboard.

Reads the artifact written by src/pipeline/predict.py (data/live/latest_forecast.json)
and displays it. Recomputes nothing itself.

Run with: streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "live" / "latest_forecast.json"
LOCAL_TZ = "America/New_York"

STATION_LABELS = {
    "buford_dam": "Buford Dam",
    "medlock_bridge": "Medlock Bridge",
}

# (display label, format string) -- iterated in this order, skipping any
# station-specific column that isn't present (e.g. Buford has no turbidity).
SUMMARY_FIELDS = [
    ("water_temp_c", "Water temp", "{:.1f}°C"),
    ("discharge_cfs", "Discharge", "{:.0f} cfs"),
    ("dissolved_oxygen_mgl", "Dissolved O2", "{:.1f} mg/L"),
    ("turbidity_fnu", "Murkiness", "{:.1f} NTU"),
    ("cloud_cover_pct", "Cloud cover", "{:.0f}%"),
    ("precipitation_in", "Precipitation", "{:.2f} in"),
    ("conductance_uscm", "Conductance", "{:.0f} µS/cm"),
]

BADGE_STYLE = {
    "Good": ("#E8F5E9", "#1B5E20"),
    "Fair": ("#FFF3E0", "#8D5B00"),
    "Poor": ("#FDECEA", "#B71C1C"),
}

CONDITION_COLORS = {"Good": "#2E7D32", "Fair": "#F9A825", "Poor": "#C62828"}


def fmt_ampm(ts: pd.Timestamp) -> str:
    """12-hour clock without a leading zero, cross-platform (avoids the
    Unix-only '%-I' strftime flag, which raises on Windows)."""
    return ts.strftime("%I:%M %p").lstrip("0")


def load_forecast() -> dict | None:
    if not DATA_PATH.exists():
        return None
    return json.loads(DATA_PATH.read_text())


def render_badge(condition: str | None) -> None:
    bg, fg = BADGE_STYLE.get(condition, ("#eee", "#333"))
    label = f"{condition} today" if condition else "No forecast"
    st.markdown(
        f'<span style="background:{bg};color:{fg};font-size:14px;'
        f'font-weight:600;padding:4px 14px;border-radius:999px;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_summary(summary: dict) -> None:
    st.caption("Right now")
    fields = [(label, fmt.format(summary[key])) for key, label, fmt in SUMMARY_FIELDS if key in summary]
    cols = st.columns(3)
    for i, (label, value) in enumerate(fields):
        cols[i % 3].metric(label, value)


def render_timeline(timeline: list[dict]) -> None:
    st.caption("Next 48 hours")
    df = pd.DataFrame(timeline)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(LOCAL_TZ)

    chart = (
        alt.Chart(df)
        .mark_rect(height=28)
        .encode(
            x=alt.X(
                "timestamp_local:T",
                title=None,
                axis=alt.Axis(format="%-I%p", labelAngle=0, tickCount="hour"),
            ),
            color=alt.Color(
                "condition:N",
                scale=alt.Scale(domain=list(CONDITION_COLORS.keys()), range=list(CONDITION_COLORS.values())),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("timestamp_local:T", title="Time", format="%a %-I:%M %p"),
                alt.Tooltip("condition:N", title="Condition"),
            ],
        )
        .properties(height=40)
    )
    st.altair_chart(chart, use_container_width=True)


def render_station(station_key: str, station_data: dict) -> None:
    st.subheader(STATION_LABELS.get(station_key, station_key))
    render_badge(station_data.get("daily_badge"))
    render_summary(station_data["current_summary"])
    render_timeline(station_data["hourly_timeline"])


def main() -> None:
    st.set_page_config(page_title="Chattahoochee trout forecast", layout="centered")
    st.title("Chattahoochee trout forecast")

    data = load_forecast()
    if data is None:
        st.warning(
            "No forecast data yet. Run `python -m src.pipeline.predict` to generate "
            f"{DATA_PATH.relative_to(DATA_PATH.parents[2])}."
        )
        return

    generated = pd.to_datetime(data["generated_at_utc"], utc=True).tz_convert(LOCAL_TZ)
    st.caption(f"Updated {fmt_ampm(generated)}")

    for station_key, station_data in data["stations"].items():
        render_station(station_key, station_data)
        st.divider()

    st.caption("Good means safe, comfortable conditions to fish and wade in — not a guaranteed catch.")


if __name__ == "__main__":
    main()
