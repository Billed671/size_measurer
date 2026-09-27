import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logging_utils import read_measurements  # noqa: E402

st.set_page_config(page_title="Size Measurer", layout="wide")
st.title("Size Measurer — мониторинг")

rows = read_measurements()

if not rows:
    st.warning("Пока нет данных в logs/measurements.jsonl")
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
now = datetime.now(timezone.utc)
hour_ago = now - timedelta(hours=1)
df = df[df["ts"] >= hour_ago].sort_values("ts", ascending=False)

st.caption(f"Записей за последний час: {len(df)}")

if df.empty:
    st.info("За последний час измерений нет.")
    st.stop()


def color_row(row):
    if row.get("status") == "recheck":
        return ["background-color: #ff6b6b; color: #111"] * len(row)
    if row.get("status") == "suspicious":
        return ["background-color: #ffe066; color: #111"] * len(row)
    return ["background-color: #e0e0e0; color: #111"] * len(row)


show = df[[
    "ts", "id", "length_mm", "width_mm", "height_mm",
    "confidence", "status", "duration_ms", "source",
]].copy()
show["ts"] = show["ts"].dt.strftime("%H:%M:%S")

styled = show.style.apply(color_row, axis=1)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.subheader("Динамика габаритов")
fig = px.line(
    df.sort_values("ts"),
    x="ts", y=["length_mm", "width_mm", "height_mm"],
    labels={"value": "мм", "ts": "время", "variable": "ось"},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Confidence")
fig2 = px.histogram(df, x="confidence", nbins=20)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Статусы")
st.bar_chart(df["status"].value_counts())
