"""Interactive Streamlit dashboard backed exclusively by PyDuckLake Silver & Gold tables.

Run with:
    streamlit run DataAnalysis/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataAnalysis.rain import network_viz as nv  # noqa: E402
from DataAnalysis.rain.ml_model import WITH_BASELINE_FEATURES, fit_scenario_predictor  # noqa: E402
from DataEngineering.lakehouse_pipeline import CATALOG_PATH, lake_has_table, query_lake  # noqa: E402

FIGURES = ROOT / "DataAnalysis" / "figures"

LABEL_TEXT = {
    "clears": "Clears (-ve)",
    "denser": "Denser (+ve)",
    "unaffected": "Unaffected",
    "too few wet hours": "Too few wet hrs (<3h)",
    "no matched rainy hours": "No matched rainy hrs",
}

LABEL_RGB = {
    "clears": [91, 211, 221, 225],
    "denser": [255, 120, 135, 240],
    "unaffected": [138, 212, 173, 210],
    "too few wet hours": [247, 170, 88, 200],
    "no matched rainy hours": [119, 139, 160, 180],
}

CLASS_NAMES = {
    "total": "All vehicles",
    "car": "Cars",
    "motorcycle": "Motorcycles",
    "bus": "Buses",
    "truck": "Trucks",
}

SEGMENT_NAMES = {
    "weekday": "Weekday (Mon–Fri)",
    "weekend": "Weekend (Sat–Sun)",
    "light_1_to_5_mm": "Light rain (1–5 mm/h)",
    "heavy_5plus_mm": "Heavy rain (≥5 mm/h)",
}

WINDOW_NAMES = {
    "all": "All hours (06–21)",
    "rush": "Rush hours (07–09, 17–20)",
    "other": "Off-peak hours",
}


@st.cache_data(show_spinner=False)
def gold_table(name: str) -> pd.DataFrame:
    """Read a Gold table directly from PyDuckLake."""
    tbl = name if name.startswith("ml_") else f"rain_{name}"
    if not lake_has_table("gold", tbl):
        return pd.DataFrame()
    return query_lake(f"SELECT * FROM lake.gold.{tbl}")


@st.cache_data(show_spinner=False)
def load_cameras_silver() -> pd.DataFrame:
    if not lake_has_table("silver", "cameras"):
        return pd.DataFrame(columns=["camera_id", "location_th", "location_en"])
    df = query_lake("""
        SELECT TRY_CAST(camera_id AS INTEGER) AS camera_id, location_th, location_en
        FROM lake.silver.cameras
        WHERE TRY_CAST(camera_id AS INTEGER) IS NOT NULL
    """)
    return df.drop_duplicates("camera_id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_frame_quality() -> pd.DataFrame:
    """Detect blank CCTV feed outage hours where average frame size is < 3 KB."""
    if not lake_has_table("silver", "cctv_frames"):
        return pd.DataFrame(columns=["camera_id", "hour_ts", "avg_bytes", "min_bytes"])
    df = query_lake("""
        SELECT TRY_CAST(camera_id AS INTEGER) AS camera_id,
               date_trunc('hour', CAST(captured_at AS TIMESTAMP)) AS hour_ts,
               AVG(file_size_bytes) AS avg_bytes,
               MIN(file_size_bytes) AS min_bytes
        FROM lake.silver.cctv_frames
        GROUP BY 1, 2
    """)
    df["hour_ts"] = pd.to_datetime(df["hour_ts"]).dt.tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def load_traffy() -> pd.DataFrame:
    if not lake_has_table("silver", "traffy_complaints"):
        return pd.DataFrame()
    df = query_lake("""
        SELECT ticket_id, timestamp, district, latitude, longitude,
               is_flood, is_road, is_traffic, is_sidewalk
        FROM lake.silver.traffy_complaints
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """)
    for col in ("is_flood", "is_road", "is_traffic", "is_sidewalk"):
        df[col] = df[col].fillna(0).astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_fm91() -> pd.DataFrame:
    if not lake_has_table("silver", "fm91_news"):
        return pd.DataFrame()
    df = query_lake("""
        SELECT article_id, datetime, category, title,
               mentions_rain, mentions_flood, mentions_congestion, mentions_accident
        FROM lake.silver.fm91_news
    """)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"] = df["datetime"].dt.date
    for col in ("mentions_rain", "mentions_flood", "mentions_congestion", "mentions_accident"):
        df[col] = df[col].fillna(0).astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_rail_stations() -> pd.DataFrame:
    if not lake_has_table("silver", "osm_poi"):
        return pd.DataFrame(columns=["name", "name_en", "network", "latitude", "longitude", "station_label"])
    df = query_lake("""
        SELECT name, name_en, network, latitude, longitude
        FROM lake.silver.osm_poi
        WHERE category = 'rail_station' AND latitude IS NOT NULL AND longitude IS NOT NULL
    """)
    df["station_label"] = df["name"].fillna(df["name_en"]).fillna("Rail Station")
    return df


@st.cache_data(show_spinner=False)
def load_districts_geojson() -> dict | None:
    if not lake_has_table("silver", "district_boundaries"):
        return None
    df = query_lake("SELECT DISTRICT_NAME_T, DISTRICT_NAME_E, geometry FROM lake.silver.district_boundaries")
    features = []
    for r in df.itertuples(index=False):
        if not r.geometry:
            continue
        features.append({
            "type": "Feature",
            "properties": {"DISTRICT_NAME_T": r.DISTRICT_NAME_T, "DISTRICT_NAME_E": r.DISTRICT_NAME_E},
            "geometry": json.loads(r.geometry),
        })
    return {"type": "FeatureCollection", "features": features}


@st.cache_data(show_spinner=False)
def get_lake_counts_total() -> tuple[str, int]:
    if lake_has_table("silver", "counts_v2"):
        n = int(query_lake("SELECT count(*) AS n FROM lake.silver.counts_v2")["n"].iloc[0])
        return "silver.counts_v2", n
    if lake_has_table("silver", "vehicle_counts"):
        n = int(query_lake("SELECT count(*) AS n FROM lake.silver.vehicle_counts")["n"].iloc[0])
        return "silver.vehicle_counts", n
    return "unavailable", 0


@st.cache_resource(show_spinner="Fitting interactive ML scenario model from PyDuckLake...")
def load_scenario_models():
    return fit_scenario_predictor()


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1f}%"


def plot_ci_forest(df: pd.DataFrame, group_col: str, group_labels: dict[str, str],
                   title: str, x_title: str = "Change when raining (%)") -> go.Figure:
    """Grouped horizontal 95% CI forest plot with generous non-overlapping margins."""
    fig = go.Figure()
    color_map = {"total": "#8ad4ad", "car": "#5bd3dd", "motorcycle": "#f7aa58"}
    groups = list(dict.fromkeys(df[group_col]))
    val_col = "effect_pct" if "effect_pct" in df.columns else "effect_pct_points"

    for cls in [c for c in ("total", "car", "motorcycle") if c in set(df["cls"])]:
        sub = df[df["cls"] == cls].copy()
        sub["group_label"] = sub[group_col].map(lambda x: group_labels.get(x, str(x)))
        custom = []
        for row in sub.itertuples(index=False):
            n_wet = getattr(row, "n_wet_hours", getattr(row, "n_hours", "—"))
            n_clk = getattr(row, "n_clock_hours", None)
            extra = f" | Clock hrs: {n_clk}" if n_clk is not None else ""
            custom.append([row.ci_low, row.ci_high, n_wet, extra])
        fig.add_trace(go.Scatter(
            x=sub[val_col],
            y=sub["group_label"],
            error_x=dict(
                type="data",
                symmetric=False,
                array=sub["ci_high"] - sub[val_col],
                arrayminus=sub[val_col] - sub["ci_low"],
                thickness=2,
                width=5,
            ),
            mode="markers",
            name=CLASS_NAMES.get(cls, cls),
            marker=dict(size=10, color=color_map.get(cls, "#a9bed0")),
            customdata=custom,
            hovertemplate=(
                "<b>%{y} — " + CLASS_NAMES.get(cls, cls) + "</b><br>"
                "Estimate: %{x:+.1f}%<br>"
                "95% CI: [%{customdata[0]:+.1f}%, %{customdata[1]:+.1f}%]<br>"
                "Obs hours: %{customdata[2]}%{customdata[3]}<extra></extra>"
            ),
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="#7993a9")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[group_labels.get(g, str(g)) for g in reversed(groups)],
        automargin=True,
    )
    fig.update_xaxes(automargin=True)
    fig.update_layout(
        title=dict(text=title, y=0.96),
        xaxis_title=x_title,
        yaxis_title="",
        height=max(320, 115 * len(groups)),
        margin=dict(l=20, r=20, t=78, b=64),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_rain_timing(timing_df: pd.DataFrame) -> go.Figure:
    view = timing_df.copy()
    view["day_label"] = view["day_type"].map({"weekday": "Weekday", "weekend": "Weekend"})
    view["hour_label"] = view["hour"].map(lambda h: f"{int(h):02d}:00")
    fig = px.bar(
        view,
        x="hour_label",
        y="n_camera_hours",
        color="day_label",
        barmode="group",
        color_discrete_map={"Weekday": "#5bd3dd", "Weekend": "#f7aa58"},
        custom_data=["n_clock_hours", "motorcycle_effect_pct"],
        labels={"hour_label": "Clock hour", "n_camera_hours": "Wet camera-hours", "day_label": "Day type"},
        title="Weekday vs Weekend Rain Exposure by Clock Hour (Overlap only at 16:00–17:00)",
    )
    fig.update_traces(hovertemplate=(
        "<b>%{x} (%{fullData.name})</b><br>"
        "Camera-hours: %{y}<br>"
        "Distinct clock hours: %{customdata[0]}<br>"
        "Motorcycle change: %{customdata[1]:+.1f}%<extra></extra>"
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=68, b=56),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_hourly_dual_axis(trend_df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {"total": "#8ad4ad", "car": "#5bd3dd", "motorcycle": "#f7aa58", "bus": "#c792ea", "truck": "#ff7887"}

    fig.add_trace(
        go.Bar(
            x=trend_df["hour_ts"],
            y=trend_df["mean_rain_mm"],
            name="Mean Rain (mm/h)",
            marker_color="rgba(91, 211, 221, 0.28)",
            hovertemplate="%{x|%d %b %H:%M}<br>Mean rain: %{y:.2f} mm/h<extra></extra>",
        ),
        secondary_y=True,
    )
    for cls in metrics:
        fig.add_trace(
            go.Scatter(
                x=trend_df["hour_ts"],
                y=trend_df[cls],
                mode="lines+markers",
                name=CLASS_NAMES.get(cls, cls),
                line=dict(width=2.5, color=colors.get(cls, "#ffffff")),
                marker=dict(size=5),
                customdata=np.column_stack([trend_df["cameras"], trend_df["wet_cameras"]]),
                hovertemplate=(
                    "<b>" + CLASS_NAMES.get(cls, cls) + "</b><br>"
                    "%{x|%d %b %H:%M}<br>"
                    "Mean density: %{y:.2f} veh/frame<br>"
                    "Active cameras: %{customdata[0]} (Wet: %{customdata[1]})<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    fig.update_yaxes(title_text="Mean vehicles / frame", secondary_y=False, automargin=True)
    fig.update_yaxes(title_text="Mean gauge rain (mm/h)", secondary_y=True, showgrid=False, automargin=True)
    fig.update_layout(
        title="Hourly Traffic Density vs Mean Rainfall",
        height=440,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=68, b=68),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_rebound(rebound_df: pd.DataFrame) -> go.Figure:
    view = rebound_df.copy()
    view["cls_name"] = view["cls"].map(lambda c: CLASS_NAMES.get(c, c))
    fig = px.line(
        view,
        x="minute",
        y="ratio",
        color="cls_name",
        markers=True,
        color_discrete_map={"Cars": "#5bd3dd", "Motorcycles": "#f7aa58"},
        custom_data=["n_events"],
        labels={"minute": "Minutes relative to first dry hour after rain (t = 0)",
                "ratio": "Ratio to matched dry baseline (1.0 = normal)", "cls_name": "Vehicle class"},
        title="Traffic Recovery Before and After Rain Stops (-60 to +180 min)",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#7993a9")
    fig.add_vline(x=0, line_dash="dot", line_color="#ff7887")
    fig.update_traces(hovertemplate="%{x:+d} min · Ratio %{y:.3f}x (events: %{customdata[0]})<extra></extra>")
    fig.update_layout(
        height=370,
        margin=dict(l=20, r=20, t=68, b=64),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_pydeck_map(cameras_df: pd.DataFrame, rail_df: pd.DataFrame,
                     traffy_df: pd.DataFrame, traffy_layer_col: str | None,
                     show_rail: bool, show_districts: bool, map_height: int) -> pdk.Deck:
    layers = []
    if show_districts:
        geojson = load_districts_geojson()
        if geojson is not None:
            layers.append(pdk.Layer(
                "GeoJsonLayer",
                data=geojson,
                stroked=True,
                filled=True,
                get_fill_color=[30, 58, 95, 35],
                get_line_color=[120, 160, 200, 120],
                line_width_min_pixels=1,
            ))

    if traffy_layer_col and not traffy_df.empty:
        heat = traffy_df[traffy_df[traffy_layer_col]][["longitude", "latitude"]]
        if not heat.empty:
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=heat,
                get_position=["longitude", "latitude"],
                opacity=0.6,
                radius_pixels=38,
            ))

    if show_rail and not rail_df.empty:
        rail_view = rail_df[["longitude", "latitude", "station_label", "network"]].copy()
        rail_view["tooltip_title"] = "Station: " + rail_view["station_label"].astype(str)
        rail_view["tooltip_body"] = rail_view["network"].fillna("BTS / MRT").astype(str)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=rail_view,
            get_position=["longitude", "latitude"],
            get_fill_color=[255, 209, 102, 190],
            get_radius=120,
            radius_min_pixels=3,
            radius_max_pixels=7,
            pickable=True,
        ))

    cam_view = cameras_df.copy()
    cam_view["fill_color"] = cam_view["label"].map(lambda x: LABEL_RGB.get(x, [119, 139, 160, 180]))
    cam_view["radius"] = np.where(cam_view["label"] == "denser", 260, 175)
    cam_view["tooltip_title"] = cam_view.apply(
        lambda r: f"Cam #{int(r['camera_id'])}" + (f" · {r['location_th']}" if r.get("location_th") else ""),
        axis=1,
    )
    cam_view["tooltip_body"] = cam_view.apply(
        lambda r: (
            f"Status: {LABEL_TEXT.get(r.get('label'), r.get('label'))} | "
            f"Effect: {pct(r.get('effect_pct'))} | "
            f"Wet hrs: {int(r.get('n_wet_hours', 0))} | "
            f"Rail: {int(r.get('rail_dist_m', 0))} m"
        ),
        axis=1,
    )
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=cam_view,
        get_position=["longitude", "latitude"],
        get_fill_color="fill_color",
        get_line_color=[10, 20, 34, 230],
        line_width_min_pixels=1,
        stroked=True,
        get_radius="radius",
        radius_min_pixels=5,
        radius_max_pixels=14,
        pickable=True,
    ))

    center_lat = float(cam_view["latitude"].mean()) if not cam_view.empty else 13.7563
    center_lon = float(cam_view["longitude"].mean()) if not cam_view.empty else 100.5018
    return pdk.Deck(
        layers=layers,
        height=map_height,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11.3, pitch=0),
        tooltip={"html": "<b>{tooltip_title}</b><br/>{tooltip_body}", "style": {"backgroundColor": "#122237", "color": "#eef5fa"}},
    )


def render_dashboard() -> None:
    st.set_page_config(page_title="Bangkok Rain & Traffic Analytics (PyDuckLake)", layout="wide")
    st.title("Bangkok Rain, Urban Mobility & Traffic Analytics")
    st.caption("100 BMA CCTV Cameras · 25–31 Aug 2026 · Powered by PyDuckLake Silver & Gold Lakehouse Tables")

    if not CATALOG_PATH.exists():
        st.error(f"PyDuckLake catalog not found at `{CATALOG_PATH}`. Run `python DataEngineering/lakehouse_pipeline.py` first.")
        st.stop()

    required = ["camera", "camera_hour", "city_effect", "camera_effect", "natural_experiment"]
    missing = [name for name in required if gold_table(name).empty]
    if missing:
        st.error("Missing Gold tables in PyDuckLake: " + ", ".join(missing) + ". Run `python -m DataAnalysis.rain.run_all`.")
        st.stop()

    city = gold_table("city_effect")
    stratified = gold_table("stratified_effects")
    timing = gold_table("rain_timing")
    overall = city[city["window"] == "all"].set_index("cls")
    hours = gold_table("camera_hour")
    hours["hour_ts"] = pd.to_datetime(hours["hour_ts"]).dt.tz_localize(None)
    hours["date"] = hours["hour_ts"].dt.date

    quality = load_frame_quality()
    if not quality.empty:
        hours = hours.merge(quality[["camera_id", "hour_ts", "avg_bytes"]], on=["camera_id", "hour_ts"], how="left")
    else:
        hours["avg_bytes"] = 20000.0
    hours["avg_bytes"] = hours["avg_bytes"].fillna(20000.0)

    cam_meta = load_cameras_silver()
    cameras = gold_table("camera").merge(gold_table("camera_effect"), on="camera_id", how="left").merge(
        cam_meta, on="camera_id", how="left",
    )
    cameras["label"] = cameras["label"].fillna("no matched rainy hours")
    cameras["n_wet_hours"] = cameras["n_wet_hours"].fillna(0).astype(int)
    cameras["location_th"] = cameras["location_th"].fillna("")
    cameras["location_en"] = cameras["location_en"].fillna("")

    # ---------------- Sidebar Parameter Sliders ----------------
    with st.sidebar:
        st.header("Dashboard Parameters")
        with st.expander("1. Time Series & Data Quality", expanded=True):
            all_dates = sorted(hours["date"].unique())
            chosen_dates = st.multiselect("Dates", all_dates, default=all_dates)
            chosen_classes = st.multiselect(
                "Vehicle Classes",
                ["motorcycle", "car", "total", "bus", "truck"],
                default=["motorcycle", "car"],
                format_func=lambda x: CLASS_NAMES.get(x, x),
            )
            exclude_blank_hours = st.checkbox(
                "Exclude blank-feed placeholder hours (< 3 KB)",
                value=True,
                help="BMA snapshots around 07:21–07:26 (e.g., Aug 27 07:00) contain 1.4 KB solid white frames.",
            )

        with st.expander("2. Spatial Map (PyDeck)", expanded=True):
            map_height = st.slider("Map Panel Height (px)", min_value=520, max_value=960, value=760, step=20)
            all_labels = sorted(cameras["label"].unique())
            selected_labels = st.multiselect(
                "Camera Response Filter",
                all_labels,
                default=all_labels,
                format_func=lambda x: LABEL_TEXT.get(x, x),
            )
            min_wet_slider = st.slider("Minimum Wet Hours", 0, int(cameras["n_wet_hours"].max()), 0)
            heat_options = {
                "None": None,
                "Traffy: Flood Reports": "is_flood",
                "Traffy: Traffic Jams": "is_traffic",
                "Traffy: Road Issues": "is_road",
                "Traffy: Sidewalk Obstructions": "is_sidewalk",
            }
            heat_choice = st.selectbox("Traffy Heatmap Overlay", list(heat_options.keys()))
            show_rail = st.checkbox("Show BTS / MRT Stations", value=True)
            show_districts = st.checkbox("Show 50 Bangkok Districts", value=False)

        with st.expander("3. Network Graph", expanded=False):
            rail_radius = st.slider("Camera ↔ Rail Radius (m)", 300, 1500, 800, step=100)
            cam_radius = st.slider("Camera ↔ Camera Corridor (m)", 400, 2000, 1200, step=100)
            net_communities = st.checkbox("Detect Communities (Greedy Modularity)", value=True)
            net_layout = st.selectbox("Layout Algorithm", nv.LAYOUTS, index=0)
            net_size_by = st.selectbox("Node Size By", ["degree", "betweenness", "closeness", "pagerank", "single size"], index=0)
            net_color_by = st.selectbox("Node Color By", (["community"] if net_communities else []) + ["rain_effect", "degree", "betweenness", "closeness", "pagerank"])
            net_scale = st.slider("Graph Spread Scale", 500, 1800, 950, step=50)
            net_size_range = st.slider("Node Size Range", 6, 60, (10, 36), step=2)
            net_noverlap = st.checkbox("Prevent Node Overlap (Noverlap)", value=True)
            net_labels_n = st.slider("Label Top N Nodes", 0, 120, 25, step=5)
            net_font_size = st.slider("Label Font Size", 8, 28, 13, step=1)
            net_min_comp = st.slider("Hide Components < Nodes", 1, 20, 1, step=1)
            net_show_edges = st.checkbox("Show Network Edges", value=True)

        with st.expander("4. Supervised ML What-If Simulator", expanded=False):
            sim_rain = st.slider("Simulated Rain (mm/hr)", 0.0, 35.0, 5.0, step=0.5)
            sim_hod = st.slider("Hour of Day", 6, 21, 17, step=1)
            sim_weekend = st.checkbox("Weekend Day", value=False)
            sim_rail_dist = st.slider("Distance to Nearest BTS/MRT (m)", 50, 3000, 450, step=50)
            sim_near_shelter = st.checkbox("Near Footbridge / Expressway Shelter", value=True)
            sim_base_total = st.slider("Typical Dry Baseline Vehicles / Frame", 3.0, 35.0, 14.0, step=0.5)
            sim_temp = st.slider("Ambient Temperature (°C)", 24.0, 36.0, 29.0, step=0.5)
            sim_vis = st.slider("METAR Visibility (km)", 2.0, 12.0, 8.0, step=0.5)

    # ---------------- Executive KPI Strip ----------------
    st.info(
        "**Key Finding:** Motorcycles visible per frame drop **-21.0%** (95% CI -27.4% to -14.4%, rush hour **-30.5%**) "
        "during rain compared to dry baselines at the exact same camera, clock hour, and weekday/weekend type. "
        "Cars overall stay near zero (-2.2%, CI -6.4% to +2.2%), pulling overall frame counts down by -5.4%."
    )
    k1, k2, k3, k4 = st.columns(4)
    for col, cls, title in zip((k1, k2, k3), ("total", "car", "motorcycle"), ("All Vehicles (Wet)", "Cars (Wet)", "Motorcycles (Wet)")):
        r = overall.loc[cls]
        col.metric(title, pct(r["effect_pct"]), help=f"95% CI {pct(r['ci_low'])} to {pct(r['ci_high'])} across {int(r['n_wet_hours'])} wet camera-hours")
    wd = set(timing[timing["day_type"] == "weekday"]["hour"]) if not timing.empty else set()
    we = set(timing[timing["day_type"] == "weekend"]["hour"]) if not timing.empty else set()
    k4.metric("Weekday–Weekend Rain Overlap", f"{len(wd & we)} hrs", help="Rain fell weekdays 16:00–20:00 (8 clock hrs, 69 cam-hrs) vs weekends 12:00–17:00 (8 clock hrs, 224 cam-hrs)")

    tabs = st.tabs([
        "Overview & Timing",
        "Time Series & Anomalies",
        "Spatial Map (PyDeck)",
        "Network Analysis (PyVis)",
        "Supervised ML & Correlations",
        "Hypotheses, Flood & News",
        "Data Quality & Causal Audit",
    ])

    # ---------------- TAB 1: OVERVIEW ----------------
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_ci_forest(city, "window", WINDOW_NAMES, "Rain Effect by Time Window (95% CI)"), width="stretch")
        with c2:
            if not stratified.empty:
                st.plotly_chart(plot_ci_forest(stratified, "segment", SEGMENT_NAMES, "Subgroup Splits: Weekday/Weekend & Rain Intensity"), width="stretch")
        if not timing.empty:
            st.plotly_chart(plot_rain_timing(timing), width="stretch")
        st.warning(
            "**Why '69 camera-hours (8 clock hours, 16:00–20:00)' matters:** Across Mon–Fri, rain >= 1 mm only occurred during 8 distinct clock-hour timestamps, hitting multiple cameras at once (summing to 69 camera-hour pairs). "
            "Weekend rain fell almost exclusively earlier in the day (12:00–17:00), leaving only **16:00 and 17:00** to compare directly."
        )
        with st.expander("Why do still-frame counts DECREASE when it rains instead of increasing congestion?"):
            st.markdown(
                "- **Frames measure spatial density, not travel speed:** A 10-minute still photo counts how many vehicles are inside the frame at that instant—it cannot measure travel time (km/h) or throughput (`flow = density × speed`).\n"
                "- **Motorcycles leave open roadways (-21.0% citywide, -30.5% rush hour):** Two-wheelers visibly disappear during rainfall (canceling trips, delaying departures until rain clears as seen in `Rebound`, or stopping under off-road cover). Because motorcycles make up ~20.5% of daytime counts (`113,310 / 551,501`), losing them pulls total vehicle counts down (`-5.4%`).\n"
                "- **Gridlock limits inflow:** Even when slow traffic builds up upstream, blocked corridors can feed *fewer* passing cars into open main-road frames. Only in our **15-hour same-hour neighbor comparison** (`natural_experiment`: rained-on cameras vs dry neighbors <= 3 km away) do cars read **+12.6 percentage points higher** (95% CI +2.2 to +20.7)."
            )

    # ---------------- TAB 2: TIME SERIES & ANOMALIES ----------------
    with tabs[1]:
        subset = hours[hours["date"].isin(chosen_dates)].copy()
        if exclude_blank_hours:
            subset = subset[subset["avg_bytes"] >= 3000]
        if not subset.empty and chosen_classes:
            trend = subset.groupby("hour_ts", as_index=False).agg(
                total=("total", "mean"),
                car=("car", "mean"),
                motorcycle=("motorcycle", "mean"),
                bus=("bus", "mean"),
                truck=("truck", "mean"),
                mean_rain_mm=("rain_mm", "mean"),
                cameras=("camera_id", "nunique"),
                wet_cameras=("wet", "sum"),
                frames=("n_frames", "sum"),
            )
            st.plotly_chart(plot_hourly_dual_axis(trend, chosen_classes), width="stretch")
        else:
            st.info("Select at least one date and one vehicle class in the sidebar.")

        rebound = gold_table("rebound")
        if not rebound.empty:
            st.plotly_chart(plot_rebound(rebound), width="stretch")

        st.info(
            "**Aug 27 07:00 (and 07:00–08:00) Drop Explanation:** Querying `lake.silver.cctv_frames` shows that on **Aug 27 at 07:21–07:26**, **71 of 73 captured frames** (across cameras like `#11, #18, #26`) have file sizes of **1,402–1,404 bytes**—completely solid-white BMA server outage placeholder images (normal frames are 25–35 KB). "
            "Thus neither YOLO detector failed; the source images were blank. Same 07:00–08:00 outage placeholders occurred across subsets of cameras on Aug 26, 28, 29, 30, and 31. Toggle `Exclude blank-feed placeholder hours` in the sidebar to inspect before vs after."
        )

    # ---------------- TAB 3: TALL PYDECK SPATIAL MAP ----------------
    with tabs[2]:
        rail_df = load_rail_stations()
        traffy_df = load_traffy()
        filtered = cameras[cameras["label"].isin(selected_labels) & (cameras["n_wet_hours"] >= min_wet_slider)].copy()
        st.caption(
            f"Showing **{len(filtered)} / {len(cameras)} cameras** · Height: `{map_height}px` · "
            "Cyan = Clears, Red = Denser, Green = Unaffected, Orange = <3 wet hrs, Slate = No matched rain"
        )
        if not filtered.empty:
            deck = build_pydeck_map(
                filtered, rail_df, traffy_df, heat_options[heat_choice], show_rail, show_districts, map_height,
            )
            st.pydeck_chart(deck, width="stretch", height=map_height)
            cols_show = ["camera_id", "location_th", "label", "effect_pct", "ci_low", "ci_high",
                         "n_wet_hours", "gauge", "gauge_dist_m", "rail_dist_m", "near_rail", "near_shelter"]
            avail = [c for c in cols_show if c in filtered.columns]
            st.dataframe(filtered[avail].sort_values("effect_pct", na_position="last"), width="stretch", hide_index=True)
        else:
            st.info("No cameras match the current sidebar filter.")

    # ---------------- TAB 4: INTERACTIVE PYVIS + NETWORKX ----------------
    with tabs[3]:
        rail_df = load_rail_stations()
        st.subheader("Bangkok CCTV & Rapid Transit Corridor Network")
        G = nv.build_bangkok_network(cameras, rail_df, rail_radius_m=rail_radius, cam_radius_m=cam_radius)
        cent_df = nv.compute_centralities(G)
        communities = nv.detect_communities(G) if net_communities else None
        if communities:
            cent_df.insert(1, "Community", cent_df["Node"].map(communities))

        stats = nv.compute_basic_stats(G)
        s_cols = st.columns(6)
        s_cols[0].metric("Nodes", stats["Nodes"])
        s_cols[1].metric("Edges", stats["Edges"])
        s_cols[2].metric("Density", f"{stats['Density']:.3f}")
        s_cols[3].metric("Diameter", stats["Diameter"])
        s_cols[4].metric("Avg Clustering", f"{stats['Avg Clustering']:.3f}")
        s_cols[5].metric("Avg Path Length", f"{stats['Avg Path Length']:.2f}")
        if not stats["Connected"]:
            st.caption(f"Graph has disconnected corridor clusters; diameter and path length reflect the largest connected component ({stats['LCC Size']} of {stats['Nodes']} nodes). Adjust edge parameters in sidebar Section 3.")

        html_content = nv.render_pyvis_html(
            G,
            cent_df,
            communities=communities,
            layout=net_layout,
            size_by=net_size_by,
            color_by=net_color_by,
            node_size_range=net_size_range,
            scale_factor=float(net_scale),
            no_overlap=net_noverlap,
            font_size=net_font_size,
            label_top_n=net_labels_n,
            min_component_size=net_min_comp,
            show_edges=net_show_edges,
        )
        st.iframe(html_content, height=760)

        dl_col, rand_col = st.columns([1, 2])
        with dl_col:
            st.download_button(
                "Download Interactive Graph (HTML)",
                data=html_content,
                file_name="bangkok_cctv_network.html",
                mime="text/html",
            )
        with rand_col:
            if st.checkbox("Compare with Random Graph (Erdős–Rényi, same nodes & edges)"):
                st.dataframe(nv.compare_with_random(G).round(3), width="stretch")

        left_n, right_n = st.columns(2)
        with left_n:
            st.subheader("Degree Distribution")
            scale_mode = st.radio("Scale", ["Histogram", "Log-log (CCDF)"], horizontal=True)
            fig_deg = nv.plot_degree_distribution(G, log_log=(scale_mode == "Log-log (CCDF)"))
            st.pyplot(fig_deg, width="stretch")
        with right_n:
            st.subheader("Centrality & Community Table")
            st.dataframe(cent_df.sort_values("Betweenness", ascending=False), width="stretch", height=400, hide_index=True)

    # ---------------- TAB 5: SUPERVISED ML & CORRELATIONS ----------------
    with tabs[4]:
        st.subheader("Supervised Traffic Prediction Models (Trained from PyDuckLake)")
        ml_metrics = gold_table("ml_metrics")
        ml_imp = gold_table("ml_feature_importance")
        ml_preds = gold_table("ml_predictions")
        corr_df = gold_table("rain_correlations")

        if ml_metrics.empty:
            st.warning("ML tables not found in PyDuckLake. Run `python DataAnalysis/rain/ml_model.py`.")
        else:
            st.markdown(
                "Supervised **Regression** (`HistGradientBoostingRegressor`, `RandomForestRegressor`, `Ridge`) predicts hourly vehicle counts per camera, while "
                "**Classification** (`HistGradientBoostingClassifier`) flags `+25%` traffic surges vs matched dry baselines. Evaluated on **Chronological Split** (Train Aug 25–29 $\\to$ Test Aug 30–31) and **5-Fold Spatial GroupKFold** (unseen cameras)."
            )
            st.dataframe(ml_metrics, width="stretch", hide_index=True)

            m_left, m_right = st.columns(2)
            with m_left:
                if not ml_imp.empty:
                    imp_target = st.selectbox("Permutation Importance Target", sorted(ml_imp["target"].unique()))
                    sub_imp = ml_imp[ml_imp["target"] == imp_target].head(10).sort_values("importance_mean")
                    fig_imp = px.bar(
                        sub_imp, x="importance_mean", y="feature", orientation="h",
                        error_x="importance_std",
                        labels={"importance_mean": "Mean Permutation Importance (R² drop)", "feature": "Feature"},
                        title=f"Top Drivers: {imp_target}",
                    )
                    fig_imp.update_layout(height=380, margin=dict(l=15, r=20, t=58, b=36),
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_imp, width="stretch")
            with m_right:
                if not ml_preds.empty:
                    ml_preds["weather"] = np.where(ml_preds["rain_mm"] >= 1.0, "Rain (>=1 mm)", "Dry (<1 mm)")
                    fig_pred = px.scatter(
                        ml_preds, x="actual", y="predicted", color="weather", opacity=0.65,
                        color_discrete_map={"Dry (<1 mm)": "#5bd3dd", "Rain (>=1 mm)": "#f7aa58"},
                        hover_data=["camera_id", "hod", "rain_mm"],
                        labels={"actual": "Actual Vehicles / Frame", "predicted": "Predicted Vehicles / Frame"},
                        title="Held-Out Test Predictions (Aug 30–31): Actual vs Predicted",
                    )
                    max_v = float(max(ml_preds["actual"].max(), ml_preds["predicted"].max(), 1.0))
                    fig_pred.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dash", color="#7993a9"))
                    fig_pred.update_layout(height=380, margin=dict(l=15, r=20, t=58, b=48),
                                           legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pred, width="stretch")

            st.subheader("Interactive What-If Scenario Simulator (Sidebar Section 4)")
            feat_df, sim_models = load_scenario_models()
            ratio_m = float(feat_df["motorcycle_base"].sum() / max(feat_df["total_base"].sum(), 1e-6))
            ratio_c = float(feat_df["car_base"].sum() / max(feat_df["total_base"].sum(), 1e-6))
            sim_row = pd.DataFrame([{
                "hod": sim_hod,
                "weekend": int(sim_weekend),
                "is_rush": int(sim_hod in {7, 8, 9, 17, 18, 19, 20}),
                "rain_mm": sim_rain,
                "wet": int(sim_rain >= 1.0),
                "heavy": int(sim_rain >= 5.0),
                "temperature_2m": sim_temp,
                "relative_humidity_2m": float(feat_df["relative_humidity_2m"].median()),
                "cloud_cover": 90.0 if sim_rain >= 1.0 else 45.0,
                "visibility_km": sim_vis,
                "wind_kt": float(feat_df["wind_kt"].median()),
                "latitude": float(feat_df["latitude"].median()),
                "longitude": float(feat_df["longitude"].median()),
                "rail_dist_m": sim_rail_dist,
                "gauge_dist_m": float(feat_df["gauge_dist_m"].median()),
                "near_rail": int(sim_rail_dist <= 500),
                "near_shelter": int(sim_near_shelter),
                "in_flood_area": 0,
                "max_flood_cm": 5.0 if sim_rain >= 10.0 else 0.0,
                "fm91_rain": 2.0 if sim_rain >= 1.0 else 0.0,
                "fm91_congestion": 2.0 if sim_rain >= 5.0 else 0.0,
                "total_base": sim_base_total,
                "car_base": sim_base_total * ratio_c,
                "motorcycle_base": sim_base_total * ratio_m,
            }])[WITH_BASELINE_FEATURES]
            dry_row = sim_row.copy()
            dry_row["rain_mm"], dry_row["wet"], dry_row["heavy"], dry_row["max_flood_cm"] = 0.0, 0, 0, 0.0
            sc1, sc2, sc3 = st.columns(3)
            for col, tgt, label in zip((sc1, sc2, sc3), ("total", "car", "motorcycle"), ("Predicted Total", "Predicted Cars", "Predicted Motorcycles")):
                pw = float(sim_models[tgt].predict(sim_row)[0])
                pdry = float(sim_models[tgt].predict(dry_row)[0])
                delta = 100.0 * (pw / max(pdry, 0.01) - 1.0)
                col.metric(label, f"{pw:.2f} veh/frame", f"{delta:+.1f}% vs 0 mm scenario")

        if not corr_df.empty:
            st.subheader("Empirical Pearson Correlations Across Lakehouse Features")
            scope_choice = st.radio("Correlation Scope", ["hourly_camera", "camera_effect"], horizontal=True,
                                    format_func=lambda s: "Hourly Observations (n ≈ 9,996)" if s == "hourly_camera" else "Camera Rain Effect (n = 94)")
            sub_c = corr_df[corr_df["scope"] == scope_choice].copy()
            fig_corr = px.bar(
                sub_c, x="pearson_r", y="feature_label", color="target", barmode="group", orientation="h",
                labels={"pearson_r": "Pearson Correlation (r)", "feature_label": "Variable", "target": "Target"},
                title="Measured Linear Correlations Across Traffic & Weather Variables",
            )
            fig_corr.add_vline(x=0, line_dash="dash", line_color="#7993a9")
            fig_corr.update_layout(height=450, margin=dict(l=20, r=20, t=58, b=48),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, width="stretch")

    # ---------------- TAB 6: HYPOTHESES, FLOOD & NEWS ----------------
    with tabs[5]:
        nat = gold_table("natural_experiment")
        if not nat.empty:
            st.subheader("1) Same-Hour Natural Experiment: Rained-on vs Dry Neighbor Cameras (≤ 3 km)")
            st.plotly_chart(
                plot_ci_forest(nat, "cls", CLASS_NAMES, "Natural Experiment (15 Clock Hours · 172 Camera Pairs)", "Difference in % vs baseline (Percentage Points)"),
                width="stretch",
            )
            st.caption("Cars read +12.59 pts higher (CI +2.2 to +20.7) at wet cameras vs dry neighbors within 3 km in the exact same hour, though observed in only 15 hours without taxi/private splits.")

        h1, h2 = st.columns(2)
        with h1:
            st.subheader("2) Shelter & Transit Overlap vs Rain Density")
            denser_share = gold_table("denser_share")
            if not denser_share.empty:
                m = denser_share.melt(id_vars="feature", value_vars=["denser_share", "other_share"], var_name="group", value_name="share")
                m["share_pct"] = m["share"] * 100.0
                m["feature_label"] = m["feature"].map({
                    "near_rail": "Near BTS/MRT (<=500 m)",
                    "near_shelter": "Near Shelter (<=150 m)",
                    "in_flood_area": "In Flood Cluster (<=100 m)",
                }).fillna(m["feature"])
                m["group_label"] = m["group"].map({"denser_share": "5 Denser Cameras", "other_share": "71 Other Cameras"})
                fig_d = px.bar(m, x="feature_label", y="share_pct", color="group_label", barmode="group",
                               color_discrete_map={"5 Denser Cameras": "#ff7887", "71 Other Cameras": "#5bd3dd"},
                               labels={"feature_label": "", "share_pct": "% of Cameras", "group_label": ""})
                fig_d.update_layout(height=340, margin=dict(l=15, r=15, t=24, b=64),
                                    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_d, width="stretch")

        with h2:
            st.subheader("3) DDS Flood Sensor ≥ 5 cm Matched to Phetchaburi Cameras")
            flood = gold_table("flood_sensor_hours")
            if not flood.empty:
                flood_cp = flood.copy()
                flood_cp["hour_ts"] = pd.to_datetime(flood_cp["hour_ts"]).dt.tz_localize(None)
                phet = hours[hours["camera_id"].isin([250, 256])].merge(flood_cp[["hour_ts", "flood_cm"]], on="hour_ts", how="inner")
                if not phet.empty:
                    comp = phet.groupby("camera_id", as_index=False)[["car", "car_base", "motorcycle", "motorcycle_base"]].mean()
                    cm = comp.melt(id_vars="camera_id", var_name="series", value_name="avg_vehicles")
                    cm["camera_label"] = cm["camera_id"].map({250: "Cam 250 (Phetchaburi 38)", 256: "Cam 256 (Phetchaburi 27)"})
                    cm["series_name"] = cm["series"].map({
                        "car": "Cars (Flood >= 5cm)", "car_base": "Cars (Dry Base)",
                        "motorcycle": "Motorcycles (Flood >= 5cm)", "motorcycle_base": "Motorcycles (Dry Base)",
                    })
                    fig_p = px.bar(cm, x="camera_label", y="avg_vehicles", color="series_name", barmode="group",
                                   labels={"camera_label": "", "avg_vehicles": "Mean vehicles/frame", "series_name": ""})
                    fig_p.update_layout(height=340, margin=dict(l=15, r=15, t=24, b=64),
                                        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_p, width="stretch")
                st.caption("`FL.RTW.02 Thanon Phetchaburi` recorded all 13 flood readings >= 5 cm (5.0–5.1 cm). Matched road cameras 250 & 256 did not show higher standing density.")

        f1, f2 = st.columns(2)
        with f1:
            st.subheader("4) Traffy Fondue Complaint Breakdown")
            traffy_df = load_traffy()
            if not traffy_df.empty:
                t_counts = pd.DataFrame({
                    "Category": ["Road (is_road)", "Traffic (is_traffic)", "Flood (is_flood)", "Sidewalk (is_sidewalk)"],
                    "Count": [int(traffy_df[c].sum()) for c in ("is_road", "is_traffic", "is_flood", "is_sidewalk")],
                })
                fig_t = px.bar(t_counts, x="Count", y="Category", orientation="h", color="Category", text_auto=True)
                fig_t.update_layout(height=300, showlegend=False, margin=dict(l=15, r=25, t=20, b=30),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_t, width="stretch")
        with f2:
            st.subheader("5) Daily FM91 Traffic News Mentions (Aug 25–31)")
            fm_df = load_fm91()
            if not fm_df.empty:
                daily = fm_df.groupby("date", as_index=False)[["mentions_rain", "mentions_flood", "mentions_congestion", "mentions_accident"]].sum()
                dm = daily.melt(id_vars="date", var_name="topic", value_name="articles")
                dm["Topic"] = dm["topic"].map({
                    "mentions_rain": "Rain", "mentions_flood": "Flood",
                    "mentions_congestion": "Congestion", "mentions_accident": "Accident",
                })
                fig_fm = px.line(dm, x="date", y="articles", color="Topic", markers=True,
                                 labels={"date": "Date", "articles": "News Articles", "Topic": ""})
                fig_fm.update_layout(height=300, margin=dict(l=15, r=15, t=20, b=56),
                                     legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fm, width="stretch")

    # ---------------- TAB 7: DATA QUALITY & CAUSAL AUDIT ----------------
    with tabs[6]:
        src_name, n_imgs = get_lake_counts_total()
        c_cols = st.columns(3)
        c_cols[0].metric("DuckLake Frame Rows", f"{n_imgs:,}", help=f"Source: lake.{src_name}")
        c_cols[1].metric("Analyzed Cameras", f"{int(overall.loc['total', 'n_cameras']):,} / 100")
        c_cols[2].metric("Wet Camera-Hours", f"{int(overall.loc['total', 'n_wet_hours']):,}")

        st.subheader("Conceptual Causal Model: Evidence & Data Coverage Audit")
        scorecard = pd.DataFrame([
            {"Causal Node": "Rain Telemetry (>= 1 mm)", "Data Status": "Present (DDS Rain, 129 gauges)", "Observed Evidence & Caveats": "293 wet camera-hours matched; motorcycles drop -21.0%."},
            {"Causal Node": "Motorcycles Sheltering / Curb Block", "Data Status": "Tested (background-subtraction)", "Observed Evidence & Caveats": "Weak difference (2.5% dry vs 2.8% wet); mostly sidewalk parking, not 1–2 travel lanes blocked."},
            {"Causal Node": "DDS Flood >= 5 cm", "Data Status": "No GPS coordinates in DDS table", "Observed Evidence & Caveats": "All 13 >=5 cm readings at FL.RTW.02 Thanon Phetchaburi; matched road cams 250/256 showed no volume rise."},
            {"Causal Node": "Traffy Fondue (Flood / Road / Traffic)", "Data Status": "Present (7,247 geolocated tickets)", "Observed Evidence & Caveats": "Mostly inside soi/alleys (only 2 cameras within 100 m of DBSCAN flood clusters)."},
            {"Causal Node": "Transit Shift & +25% Ride Surge", "Data Status": "Not directly measured", "Observed Evidence & Caveats": "Natural experiment shows +12.6 pts cars in 15 hrs, but no BTS/MRT ridership, taxi, or surge-pricing feeds."},
            {"Causal Node": "Capacity Drop -> Road Gridlock", "Data Status": "Density only (no speed/flow)", "Observed Evidence & Caveats": "10-min still snapshots measure vehicle count in view, not km/h travel speed or queue delay."},
        ])
        st.dataframe(scorecard, width="stretch", hide_index=True)

        with st.expander("View Original Static Verification Plots (DataAnalysis/figures/*.png)"):
            g1, g2 = st.columns(2)
            for idx, fname in enumerate(("city_effect", "camera_map", "rebound", "natural_experiment", "denser_share", "mealtime")):
                fp = FIGURES / f"{fname}.png"
                if fp.is_file():
                    (g1 if idx % 2 == 0 else g2).image(str(fp), caption=fp.name, width="stretch")

        st.caption("All dashboard queries execute directly against PyDuckLake (`DataEngineering/lakehouse/bkk_cctv.ducklake`). Raw & Gold CSV files are only seed/export artifacts.")


if __name__ == "__main__":
    render_dashboard()
