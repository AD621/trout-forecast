"""Streamlit dashboard.

Reads the artifact written by src/pipeline/predict.py (data/live/latest_forecast.json)
and displays it. Recomputes nothing itself.

Run with: streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import DAM_SCHEDULE_URL, STATIONS  # noqa: E402
from src.scoring.composite import WADEABLE_MAX_CFS  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "live" / "latest_forecast.json"
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


def render_summary(summary: dict, station_key: str) -> None:
    st.caption("Right now")
    fields = [(key, label, fmt.format(summary[key])) for key, label, fmt in SUMMARY_FIELDS if key in summary]
    cols = st.columns(3)
    wadeable_max = WADEABLE_MAX_CFS.get(station_key)
    for i, (key, label, value) in enumerate(fields):
        with cols[i % 3]:
            st.metric(label, value)
            if key == "discharge_cfs" and wadeable_max is not None and summary[key] > wadeable_max:
                st.markdown(
                    '<span style="color:#8D5B00;font-size:12px;">'
                    "&#9888;&nbsp;High water level</span>",
                    unsafe_allow_html=True,
                )


def render_timeline(timeline: list[dict]) -> None:
    """Plain HTML/CSS colored strip -- deliberately not an Altair chart.

    An Altair/Vega-Lite version of this hit a serialization issue specific
    to how Streamlit passes chart data to the browser (a "Infinite extent"
    error on the temporal axis, reproducible even with a minimal DataFrame),
    which plain markdown sidesteps entirely.
    """
    st.caption("Next 48 hours")
    times = [
        pd.Timestamp(row["timestamp_utc"]).tz_convert(LOCAL_TZ) for row in timeline
    ]
    conditions = [row["condition"] for row in timeline]
    n = len(times)

    blocks = "".join(
        f'<div style="flex:1;height:28px;background:{CONDITION_COLORS.get(c, "#999")};" '
        f'title="{t.strftime("%a")} {fmt_ampm(t)} -- {c}"></div>'
        for t, c in zip(times, conditions)
    )

    label_idx = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1])) if n else []
    labels = "".join(
        f'<span style="flex:1;text-align:center;">{fmt_ampm(times[i]) if i in label_idx else ""}</span>'
        for i in range(n)
    )

    legend = "".join(
        f'<span style="margin-right:14px;"><span style="display:inline-block;width:10px;'
        f'height:10px;background:{color};border-radius:2px;margin-right:4px;"></span>{label}</span>'
        for label, color in CONDITION_COLORS.items()
    )

    st.markdown(
        f'<div style="display:flex;gap:2px;border-radius:6px;overflow:hidden;">{blocks}</div>'
        f'<div style="display:flex;font-size:11px;color:#888;margin-top:4px;">{labels}</div>'
        f'<div style="font-size:12px;margin-top:8px;">{legend}</div>',
        unsafe_allow_html=True,
    )


def render_links(station_key: str) -> None:
    links = [f"[View live USGS data]({STATIONS[station_key]['usgs_url']})"]
    if station_key == "buford_dam":
        links.append(f"[Click here for the dam release schedule]({DAM_SCHEDULE_URL})")
    st.markdown(" &nbsp;·&nbsp; ".join(links))


def render_station(station_key: str, station_data: dict) -> None:
    st.subheader(STATION_LABELS.get(station_key, station_key))
    render_badge(station_data.get("daily_badge"))
    render_summary(station_data["current_summary"], station_key)
    render_links(station_key)
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

    st.caption("Refreshes daily at 6:00 AM EST")
    generated = pd.to_datetime(data["generated_at_utc"], utc=True).tz_convert(LOCAL_TZ)
    st.caption(f"Updated {fmt_ampm(generated)}")

    for station_key, station_data in data["stations"].items():
        render_station(station_key, station_data)
        st.divider()

    st.caption("Good means safe, comfortable conditions to fish and wade in — not a guaranteed catch.")


if __name__ == "__main__":
    main()
