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

from DataAnalysis.i18n import (  # noqa: E402
    CLASS_NAMES_BY_LANG,
    DAY_TYPE_BY_LANG,
    HEAT_OPTIONS_BY_LANG,
    LABEL_TEXT_BY_LANG,
    SEGMENT_NAMES_BY_LANG,
    UI,
    WINDOW_NAMES_BY_LANG,
    localize_network_tables,
    tr,
)
from DataAnalysis.rain import network_viz as nv  # noqa: E402
from DataAnalysis.rain.ml_model import WITH_BASELINE_FEATURES, fit_scenario_predictor  # noqa: E402
from DataEngineering.lakehouse_pipeline import CATALOG_PATH, lake_has_table, query_lake  # noqa: E402

FIGURES = ROOT / "DataAnalysis" / "figures"

LABEL_TEXT = LABEL_TEXT_BY_LANG["en"]
CLASS_NAMES = CLASS_NAMES_BY_LANG["en"]
SEGMENT_NAMES = SEGMENT_NAMES_BY_LANG["en"]
WINDOW_NAMES = WINDOW_NAMES_BY_LANG["en"]

LABEL_RGB = {
    "clears": [91, 211, 221, 225],
    "denser": [255, 120, 135, 240],
    "unaffected": [138, 212, 173, 210],
    "too few wet hours": [247, 170, 88, 200],
    "no matched rainy hours": [119, 139, 160, 180],
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
    if lake_has_table("silver", "transit_stations"):
        df = query_lake("""
            SELECT name_th AS name, name_en, primary_system AS network, latitude, longitude
            FROM lake.silver.transit_stations
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """)
    elif lake_has_table("silver", "osm_poi"):
        df = query_lake("""
            SELECT name, name_en, network, latitude, longitude
            FROM lake.silver.osm_poi
            WHERE category = 'rail_station' AND latitude IS NOT NULL AND longitude IS NOT NULL
        """)
    else:
        return pd.DataFrame(columns=["name", "name_en", "network", "latitude", "longitude", "station_label"])
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


def _toggle_lang() -> None:
    st.session_state["lang"] = "th" if st.session_state.get("lang", "en") == "en" else "en"


def plot_ci_forest(df: pd.DataFrame, group_col: str, group_labels: dict[str, str],
                   title: str, x_title: str = "Change when raining (%)", lang: str = "en") -> go.Figure:
    """Grouped horizontal 95% CI forest plot with generous non-overlapping margins."""
    fig = go.Figure()
    color_map = {"total": "#8ad4ad", "car": "#5bd3dd", "motorcycle": "#f7aa58"}
    cls_names = CLASS_NAMES_BY_LANG.get(lang, CLASS_NAMES)
    groups = list(dict.fromkeys(df[group_col]))
    val_col = "effect_pct" if "effect_pct" in df.columns else "effect_pct_points"

    for cls in [c for c in ("total", "car", "motorcycle") if c in set(df["cls"])]:
        sub = df[df["cls"] == cls].copy()
        sub["group_label"] = sub[group_col].map(lambda x: group_labels.get(x, str(x)))
        custom = []
        for row in sub.itertuples(index=False):
            n_wet = getattr(row, "n_wet_hours", getattr(row, "n_hours", "—"))
            n_clk = getattr(row, "n_clock_hours", None)
            clk_lbl = "ชม.นาฬิกา" if lang == "th" else "Clock hrs"
            extra = f" | {clk_lbl}: {n_clk}" if n_clk is not None else ""
            custom.append([row.ci_low, row.ci_high, n_wet, extra])
        cname = cls_names.get(cls, cls)
        est_lbl = "ค่าประมาณ" if lang == "th" else "Estimate"
        obs_lbl = "ชม.สังเกต" if lang == "th" else "Obs hours"
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
            name=cname,
            marker=dict(size=10, color=color_map.get(cls, "#a9bed0")),
            customdata=custom,
            hovertemplate=(
                f"<b>%{{y}} — {cname}</b><br>"
                f"{est_lbl}: %{{x:+.1f}}%<br>"
                "95% CI: [%{customdata[0]:+.1f}%, %{customdata[1]:+.1f}%]<br>"
                f"{obs_lbl}: %{{customdata[2]}}%{{customdata[3]}}<extra></extra>"
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


def plot_rain_timing(timing_df: pd.DataFrame, lang: str = "en") -> go.Figure:
    day_map = DAY_TYPE_BY_LANG.get(lang, DAY_TYPE_BY_LANG["en"])
    view = timing_df.copy()
    view["day_label"] = view["day_type"].map(day_map)
    view["hour_label"] = view["hour"].map(lambda h: f"{int(h):02d}:00")
    x_lbl = "ชั่วโมงนาฬิกา" if lang == "th" else "Clock hour"
    y_lbl = "ชั่วโมงกล้องที่มีฝน" if lang == "th" else "Wet camera-hours"
    d_lbl = "ประเภทวัน" if lang == "th" else "Day type"
    fig = px.bar(
        view,
        x="hour_label",
        y="n_camera_hours",
        color="day_label",
        barmode="group",
        color_discrete_map={day_map["weekday"]: "#5bd3dd", day_map["weekend"]: "#f7aa58"},
        custom_data=["n_clock_hours", "motorcycle_effect_pct"],
        labels={"hour_label": x_lbl, "n_camera_hours": y_lbl, "day_label": d_lbl},
        title=tr("timing_title", lang),
    )
    ch_lbl = "ชม.กล้อง" if lang == "th" else "Camera-hours"
    dc_lbl = "จำนวนชั่วโมงเวลาจริง" if lang == "th" else "Distinct clock hours"
    mc_lbl = "การเปลี่ยนมอเตอร์ไซค์" if lang == "th" else "Motorcycle change"
    fig.update_traces(hovertemplate=(
        "<b>%{x} (%{fullData.name})</b><br>"
        f"{ch_lbl}: %{{y}}<br>"
        f"{dc_lbl}: %{{customdata[0]}}<br>"
        f"{mc_lbl}: %{{customdata[1]:+.1f}}%<extra></extra>"
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=68, b=56),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_hourly_dual_axis(trend_df: pd.DataFrame, metrics: list[str], lang: str = "en") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {"total": "#8ad4ad", "car": "#5bd3dd", "motorcycle": "#f7aa58", "bus": "#c792ea", "truck": "#ff7887"}
    cls_names = CLASS_NAMES_BY_LANG.get(lang, CLASS_NAMES)

    rain_hover = "ฝนเฉลี่ย: %{y:.2f} มม./ชม." if lang == "th" else "Mean rain: %{y:.2f} mm/h"
    fig.add_trace(
        go.Bar(
            x=trend_df["hour_ts"],
            y=trend_df["mean_rain_mm"],
            name=tr("ts_rain_bar", lang),
            marker_color="rgba(91, 211, 221, 0.28)",
            hovertemplate=f"%{{x|%d %b %H:%M}}<br>{rain_hover}<extra></extra>",
        ),
        secondary_y=True,
    )
    dens_lbl = "ความหนาแน่นเฉลี่ย" if lang == "th" else "Mean density"
    unit_lbl = "คัน/เฟรม" if lang == "th" else "veh/frame"
    cam_lbl = "กล้องทำงาน" if lang == "th" else "Active cameras"
    wet_lbl = "ฝนตก" if lang == "th" else "Wet"
    for cls in metrics:
        cname = cls_names.get(cls, cls)
        fig.add_trace(
            go.Scatter(
                x=trend_df["hour_ts"],
                y=trend_df[cls],
                mode="lines+markers",
                name=cname,
                line=dict(width=2.5, color=colors.get(cls, "#ffffff")),
                marker=dict(size=5),
                customdata=np.column_stack([trend_df["cameras"], trend_df["wet_cameras"]]),
                hovertemplate=(
                    f"<b>{cname}</b><br>"
                    "%{x|%d %b %H:%M}<br>"
                    f"{dens_lbl}: %{{y:.2f}} {unit_lbl}<br>"
                    f"{cam_lbl}: %{{customdata[0]}} ({wet_lbl}: %{{customdata[1]}})<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    fig.update_yaxes(title_text=tr("ts_y1", lang), secondary_y=False, automargin=True)
    fig.update_yaxes(title_text=tr("ts_y2", lang), secondary_y=True, showgrid=False, automargin=True)
    fig.update_layout(
        title=tr("ts_title", lang),
        height=440,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=68, b=68),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_rebound(rebound_df: pd.DataFrame, lang: str = "en") -> go.Figure:
    cls_names = CLASS_NAMES_BY_LANG.get(lang, CLASS_NAMES)
    view = rebound_df.copy()
    view["cls_name"] = view["cls"].map(lambda c: cls_names.get(c, c))
    fig = px.line(
        view,
        x="minute",
        y="ratio",
        color="cls_name",
        markers=True,
        color_discrete_map={cls_names["car"]: "#5bd3dd", cls_names["motorcycle"]: "#f7aa58"},
        custom_data=["n_events"],
        labels={"minute": tr("rebound_x", lang), "ratio": tr("rebound_y", lang), "cls_name": tr("rebound_cls", lang)},
        title=tr("rebound_title", lang),
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#7993a9")
    fig.add_vline(x=0, line_dash="dot", line_color="#ff7887")
    min_u = "นาที" if lang == "th" else "min"
    ratio_u = "อัตราส่วน" if lang == "th" else "Ratio"
    ev_u = "เหตุการณ์" if lang == "th" else "events"
    fig.update_traces(hovertemplate=f"%{{x:+d}} {min_u} · {ratio_u} %{{y:.3f}}x ({ev_u}: %{{customdata[0]}})<extra></extra>")
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
                     show_rail: bool, show_districts: bool, map_height: int, lang: str = "en") -> pdk.Deck:
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
        st_prefix = "สถานี: " if lang == "th" else "Station: "
        rail_view["tooltip_title"] = st_prefix + rail_view["station_label"].astype(str)
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

    lbl_map = LABEL_TEXT_BY_LANG.get(lang, LABEL_TEXT)
    cam_view = cameras_df.copy()
    cam_view["fill_color"] = cam_view["label"].map(lambda x: LABEL_RGB.get(x, [119, 139, 160, 180]))
    cam_view["radius"] = np.where(cam_view["label"] == "denser", 260, 175)
    cam_prefix = "กล้อง #" if lang == "th" else "Cam #"
    cam_view["tooltip_title"] = cam_view.apply(
        lambda r: f"{cam_prefix}{int(r['camera_id'])}" + (
            f" · {r['location_th'] or r.get('location_en', '')}"
            if lang == "th"
            else f" · {r.get('location_en') or r.get('location_th', '')}"
        ),
        axis=1,
    )
    if lang == "th":
        cam_view["tooltip_body"] = cam_view.apply(
            lambda r: (
                f"สถานะ: {lbl_map.get(r.get('label'), r.get('label'))} | "
                f"ผลกระทบ: {pct(r.get('effect_pct'))} | "
                f"ชม.ฝน: {int(r.get('n_wet_hours', 0))} | "
                f"รถไฟฟ้า: {int(r.get('rail_dist_m', 0))} ม."
            ),
            axis=1,
        )
    else:
        cam_view["tooltip_body"] = cam_view.apply(
            lambda r: (
                f"Status: {lbl_map.get(r.get('label'), r.get('label'))} | "
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
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    lang = st.session_state["lang"]
    cls_names = CLASS_NAMES_BY_LANG.get(lang, CLASS_NAMES)
    lbl_map = LABEL_TEXT_BY_LANG.get(lang, LABEL_TEXT)
    seg_map = SEGMENT_NAMES_BY_LANG.get(lang, SEGMENT_NAMES)
    win_map = WINDOW_NAMES_BY_LANG.get(lang, WINDOW_NAMES)
    heat_options = HEAT_OPTIONS_BY_LANG.get(lang, HEAT_OPTIONS_BY_LANG["en"])

    st.set_page_config(page_title=tr("page_title", lang), layout="wide")
    hdr_col, lang_col = st.columns([6, 1])
    with hdr_col:
        st.title(tr("main_title", lang))
        st.caption(tr("subtitle", lang))
    with lang_col:
        st.write("")
        st.button(
            tr("toggle_btn", lang),
            key="lang_toggle_btn",
            on_click=_toggle_lang,
            help=tr("toggle_help", lang),
            width="stretch",
        )

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
    cameras = gold_table("camera").merge(gold_table("camera_effect"), on="camera_id", how="left")
    if "location_th" not in cameras.columns:
        cameras = cameras.merge(cam_meta, on="camera_id", how="left")
    elif not cam_meta.empty:
        meta_idx = cam_meta.set_index("camera_id")
        for col in ("location_th", "location_en"):
            cameras[col] = cameras[col].fillna(cameras["camera_id"].map(meta_idx[col]))
    cameras["label"] = cameras["label"].fillna("no matched rainy hours")
    cameras["n_wet_hours"] = cameras["n_wet_hours"].fillna(0).astype(int)
    for col in ("location_th", "location_en"):
        cameras[col] = cameras[col].fillna("") if col in cameras.columns else ""

    # ---------------- Sidebar Parameter Sliders ----------------
    with st.sidebar:
        st.header(tr("sidebar_header", lang))
        with st.expander(tr("exp_ts", lang), expanded=True):
            all_dates = sorted(hours["date"].unique())
            chosen_dates = st.multiselect(tr("dates", lang), all_dates, default=all_dates)
            chosen_classes = st.multiselect(
                tr("veh_classes", lang),
                ["motorcycle", "car", "total", "bus", "truck"],
                default=["motorcycle", "car"],
                format_func=lambda x: cls_names.get(x, x),
            )
            exclude_blank_hours = st.checkbox(
                tr("excl_blank", lang),
                value=True,
                help=tr("excl_blank_help", lang),
            )

        with st.expander(tr("exp_map", lang), expanded=True):
            map_height = st.slider(tr("map_height", lang), min_value=520, max_value=960, value=760, step=20)
            all_labels = sorted(cameras["label"].unique())
            selected_labels = st.multiselect(
                tr("cam_filter", lang),
                all_labels,
                default=all_labels,
                format_func=lambda x: lbl_map.get(x, x),
            )
            min_wet_slider = st.slider(tr("min_wet", lang), 0, int(cameras["n_wet_hours"].max()), 0)
            heat_choice = st.selectbox(tr("traffy_overlay", lang), list(heat_options.keys()))
            show_rail = st.checkbox(tr("show_rail", lang), value=True)
            show_districts = st.checkbox(tr("show_dist", lang), value=False)

        with st.expander(tr("exp_net", lang), expanded=False):
            rail_radius = st.slider(tr("rail_rad", lang), 300, 1500, 800, step=100)
            cam_radius = st.slider(tr("cam_rad", lang), 400, 2000, 900, step=100)
            k_rail = st.slider(tr("k_rail", lang), 1, 3, 1, step=1)
            k_cam = st.slider(tr("k_cam", lang), 0, 3, 2, step=1)
            net_communities = st.checkbox(tr("net_comm", lang), value=True)
            net_layout = st.selectbox(tr("net_layout", lang), nv.LAYOUTS, index=0)
            net_size_by = st.selectbox(tr("net_size", lang), ["degree", "betweenness", "closeness", "pagerank", "single size"], index=0)
            net_color_by = st.selectbox(tr("net_color", lang), (["community"] if net_communities else []) + ["rain_effect", "degree", "betweenness", "closeness", "pagerank"])
            net_scale = st.slider(tr("net_scale", lang), 600, 2200, 1350, step=50)
            net_size_range = st.slider(tr("net_range", lang), 6, 48, (10, 30), step=2)
            net_noverlap = st.checkbox(tr("net_noverlap", lang), value=False)
            net_labels_n = st.slider(tr("net_labels_n", lang), 0, 100, 35, step=5)
            net_font_size = st.slider(tr("net_font", lang), 8, 28, 13, step=1)
            net_min_comp = st.slider(tr("net_min_comp", lang), 1, 15, 2, step=1)
            net_show_edges = st.checkbox(tr("net_edges", lang), value=True)

        with st.expander(tr("exp_sim", lang), expanded=False):
            sim_rain = st.slider(tr("sim_rain", lang), 0.0, 35.0, 5.0, step=0.5)
            sim_hod = st.slider(tr("sim_hod", lang), 6, 21, 17, step=1)
            sim_weekend = st.checkbox(tr("sim_weekend", lang), value=False)
            sim_rail_dist = st.slider(tr("sim_rail_dist", lang), 50, 3000, 450, step=50)
            sim_near_shelter = st.checkbox(tr("sim_shelter", lang), value=True)
            sim_base_total = st.slider(tr("sim_base", lang), 3.0, 35.0, 14.0, step=0.5)
            sim_temp = st.slider(tr("sim_temp", lang), 24.0, 36.0, 29.0, step=0.5)
            sim_vis = st.slider(tr("sim_vis", lang), 2.0, 12.0, 8.0, step=0.5)

    # ---------------- Executive KPI Strip ----------------
    st.info(tr("kpi_banner", lang))
    k1, k2, k3, k4 = st.columns(4)
    kpi_titles = (tr("kpi_all", lang), tr("kpi_cars", lang), tr("kpi_moto", lang))
    for col, cls, title in zip((k1, k2, k3), ("total", "car", "motorcycle"), kpi_titles):
        r = overall.loc[cls]
        col.metric(
            title,
            pct(r["effect_pct"]),
            help=tr("kpi_help", lang, lo=pct(r["ci_low"]), hi=pct(r["ci_high"]), n=int(r["n_wet_hours"])),
        )
    wd = set(timing[timing["day_type"] == "weekday"]["hour"]) if not timing.empty else set()
    we = set(timing[timing["day_type"] == "weekend"]["hour"]) if not timing.empty else set()
    k4.metric(tr("kpi_overlap", lang), tr("kpi_overlap_hrs", lang, n=len(wd & we)), help=tr("kpi_overlap_help", lang))

    tabs = st.tabs(UI[lang]["tabs"])

    # ---------------- TAB 1: OVERVIEW ----------------
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_ci_forest(city, "window", win_map, tr("ci_title_win", lang), tr("ci_x", lang), lang=lang), width="stretch")
        with c2:
            if not stratified.empty:
                st.plotly_chart(plot_ci_forest(stratified, "segment", seg_map, tr("ci_title_seg", lang), tr("ci_x", lang), lang=lang), width="stretch")
        if not timing.empty:
            st.plotly_chart(plot_rain_timing(timing, lang=lang), width="stretch")
        st.warning(tr("timing_warn", lang))
        with st.expander(tr("why_dec_exp", lang)):
            st.markdown(tr("why_dec_md", lang))

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
            st.plotly_chart(plot_hourly_dual_axis(trend, chosen_classes, lang=lang), width="stretch")
        else:
            st.info(tr("ts_empty", lang))

        rebound = gold_table("rebound")
        if not rebound.empty:
            st.plotly_chart(plot_rebound(rebound, lang=lang), width="stretch")

        st.info(tr("outage_info", lang))

    # ---------------- TAB 3: TALL PYDECK SPATIAL MAP ----------------
    with tabs[2]:
        rail_df = load_rail_stations()
        traffy_df = load_traffy()
        filtered = cameras[cameras["label"].isin(selected_labels) & (cameras["n_wet_hours"] >= min_wet_slider)].copy()
        st.caption(tr("map_cap", lang, shown=len(filtered), total=len(cameras), h=map_height))
        if not filtered.empty:
            deck = build_pydeck_map(
                filtered, rail_df, traffy_df, heat_options[heat_choice], show_rail, show_districts, map_height, lang=lang,
            )
            st.pydeck_chart(deck, width="stretch", height=map_height)
            cols_show = ["camera_id", "location_th", "location_en", "label", "effect_pct", "ci_low", "ci_high",
                         "n_wet_hours", "gauge", "gauge_dist_m", "rail_dist_m", "near_rail", "near_shelter"]
            avail = [c for c in cols_show if c in filtered.columns]
            st.dataframe(filtered[avail].sort_values("effect_pct", na_position="last"), width="stretch", hide_index=True)
        else:
            st.info(tr("map_empty", lang))

    # ---------------- TAB 4: INTERACTIVE PYVIS + NETWORKX ----------------
    with tabs[3]:
        rail_df = load_rail_stations()
        st.subheader(tr("net_subhdr", lang))
        G = nv.build_bangkok_network(
            cameras, rail_df, rail_radius_m=rail_radius, cam_radius_m=cam_radius, k_rail=k_rail, k_cam=k_cam,
        )
        cent_df = nv.compute_centralities(G)
        communities = nv.detect_communities(G) if net_communities else None
        if communities:
            cent_df.insert(1, "Community", cent_df["Node"].map(communities))
        cam_links_df, comm_summary_df = nv.connection_summary_tables(G, communities)

        stats = nv.compute_basic_stats(G)
        s_cols = st.columns(6)
        m_lbls = UI[lang]["net_metrics"]
        s_cols[0].metric(m_lbls[0], stats["Nodes"])
        s_cols[1].metric(m_lbls[1], stats["Edges"])
        s_cols[2].metric(m_lbls[2], len(set(communities.values())) if communities else 0)
        s_cols[3].metric(m_lbls[3], stats["Diameter"])
        s_cols[4].metric(m_lbls[4], f"{stats['Avg Clustering']:.3f}")
        s_cols[5].metric(m_lbls[5], f"{stats['Avg Path Length']:.2f}")

        selected_comm = None
        all_cl_label = tr("net_all_clusters", lang)
        if communities and not comm_summary_df.empty:
            multi_comms = comm_summary_df[comm_summary_df["Cameras"] + comm_summary_df["Stations"] >= 2]
            labels_map = {
                tr(
                    "net_comm_fmt",
                    lang,
                    cid=int(r["Community"]),
                    cams=int(r["Cameras"]),
                    st=int(r["Stations"]),
                    hubs=r["Station Hubs"] or tr("net_corr_only", lang),
                ): int(r["Community"])
                for _, r in multi_comms.iterrows()
            }
            pick = st.selectbox(tr("net_zoom", lang), [all_cl_label] + list(labels_map.keys()))
            if pick != all_cl_label:
                selected_comm = labels_map[pick]

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
            selected_community=selected_comm,
        )
        st.iframe(html_content, height=740)

        dl_col, rand_col = st.columns([1, 2])
        with dl_col:
            st.download_button(
                tr("net_dl", lang),
                data=html_content,
                file_name="bangkok_cctv_network.html",
                mime="text/html",
            )
        with rand_col:
            if st.checkbox(tr("net_rand", lang)):
                st.dataframe(nv.compare_with_random(G).round(3), width="stretch")

        raw_links = cam_links_df if selected_comm is None else cam_links_df[cam_links_df["Community"] == selected_comm]
        view_links, view_comms = localize_network_tables(raw_links, comm_summary_df, lang)

        st.subheader(tr("net_links_hdr", lang))
        st.dataframe(view_links, width="stretch", height=320, hide_index=True)

        if not comm_summary_df.empty:
            st.subheader(tr("net_comms_hdr", lang))
            st.dataframe(view_comms, width="stretch", height=260, hide_index=True)

        left_n, right_n = st.columns(2)
        with left_n:
            st.subheader(tr("net_deg_hdr", lang))
            scale_opts = UI[lang]["net_scale_opts"]
            scale_mode = st.radio(tr("net_scale_radio", lang), scale_opts, horizontal=True)
            fig_deg = nv.plot_degree_distribution(G, log_log=(scale_mode == scale_opts[1]))
            st.pyplot(fig_deg, width="stretch")
        with right_n:
            st.subheader(tr("net_cent_hdr", lang))
            st.dataframe(cent_df.sort_values("Betweenness", ascending=False), width="stretch", height=400, hide_index=True)

    # ---------------- TAB 5: SUPERVISED ML & CORRELATIONS ----------------
    with tabs[4]:
        st.subheader(tr("ml_subhdr", lang))
        ml_metrics = gold_table("ml_metrics")
        ml_imp = gold_table("ml_feature_importance")
        ml_preds = gold_table("ml_predictions")
        corr_df = gold_table("rain_correlations")

        if ml_metrics.empty:
            st.warning(tr("ml_missing", lang))
        else:
            st.markdown(tr("ml_desc", lang))
            st.dataframe(ml_metrics, width="stretch", hide_index=True)

            m_left, m_right = st.columns(2)
            with m_left:
                if not ml_imp.empty:
                    imp_target = st.selectbox(tr("ml_imp_sel", lang), sorted(ml_imp["target"].unique()))
                    sub_imp = ml_imp[ml_imp["target"] == imp_target].head(10).sort_values("importance_mean")
                    fig_imp = px.bar(
                        sub_imp, x="importance_mean", y="feature", orientation="h",
                        error_x="importance_std",
                        labels={"importance_mean": tr("ml_imp_x", lang), "feature": tr("ml_imp_y", lang)},
                        title=tr("ml_imp_title", lang, tgt=imp_target),
                    )
                    fig_imp.update_layout(height=380, margin=dict(l=15, r=20, t=58, b=36),
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_imp, width="stretch")
            with m_right:
                if not ml_preds.empty:
                    wet_tag = tr("ml_pred_wet", lang)
                    dry_tag = tr("ml_pred_dry", lang)
                    ml_preds["weather"] = np.where(ml_preds["rain_mm"] >= 1.0, wet_tag, dry_tag)
                    fig_pred = px.scatter(
                        ml_preds, x="actual", y="predicted", color="weather", opacity=0.65,
                        color_discrete_map={dry_tag: "#5bd3dd", wet_tag: "#f7aa58"},
                        hover_data=["camera_id", "hod", "rain_mm"],
                        labels={"actual": tr("ml_pred_x", lang), "predicted": tr("ml_pred_y", lang)},
                        title=tr("ml_pred_title", lang),
                    )
                    max_v = float(max(ml_preds["actual"].max(), ml_preds["predicted"].max(), 1.0))
                    fig_pred.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dash", color="#7993a9"))
                    fig_pred.update_layout(height=380, margin=dict(l=15, r=20, t=58, b=48),
                                           legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pred, width="stretch")

            st.subheader(tr("sim_subhdr", lang))
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
            for col, tgt, label in zip((sc1, sc2, sc3), ("total", "car", "motorcycle"), UI[lang]["sim_cards"]):
                pw = float(sim_models[tgt].predict(sim_row)[0])
                pdry = float(sim_models[tgt].predict(dry_row)[0])
                delta = 100.0 * (pw / max(pdry, 0.01) - 1.0)
                col.metric(label, tr("sim_unit", lang, val=pw), tr("sim_delta", lang, delta=delta))

        if not corr_df.empty:
            st.subheader(tr("corr_subhdr", lang))
            scope_choice = st.radio(
                tr("corr_scope", lang),
                ["hourly_camera", "camera_effect"],
                horizontal=True,
                format_func=lambda s: tr("corr_hourly", lang) if s == "hourly_camera" else tr("corr_cam", lang),
            )
            sub_c = corr_df[corr_df["scope"] == scope_choice].copy()
            fig_corr = px.bar(
                sub_c, x="pearson_r", y="feature_label", color="target", barmode="group", orientation="h",
                labels={"pearson_r": tr("corr_x", lang), "feature_label": tr("corr_y", lang), "target": tr("corr_t", lang)},
                title=tr("corr_title", lang),
            )
            fig_corr.add_vline(x=0, line_dash="dash", line_color="#7993a9")
            fig_corr.update_layout(height=450, margin=dict(l=20, r=20, t=58, b=48),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, width="stretch")

    # ---------------- TAB 6: HYPOTHESES, FLOOD & NEWS ----------------
    with tabs[5]:
        nat = gold_table("natural_experiment")
        if not nat.empty:
            st.subheader(tr("hyp_1", lang))
            st.plotly_chart(
                plot_ci_forest(nat, "cls", cls_names, tr("hyp_1_title", lang), tr("hyp_1_x", lang), lang=lang),
                width="stretch",
            )
            st.caption(tr("hyp_1_cap", lang))

        h1, h2 = st.columns(2)
        with h1:
            st.subheader(tr("hyp_2", lang))
            denser_share = gold_table("denser_share")
            if not denser_share.empty:
                m = denser_share.melt(id_vars="feature", value_vars=["denser_share", "other_share"], var_name="group", value_name="share")
                m["share_pct"] = m["share"] * 100.0
                feat_lbls = (
                    {"near_rail": "ใกล้ BTS/MRT (<=500 ม.)", "near_shelter": "ใกล้จุดหลบฝน (<=150 ม.)", "in_flood_area": "ในกลุ่มจุดน้ำท่วม (<=100 ม.)"}
                    if lang == "th"
                    else {"near_rail": "Near BTS/MRT (<=500 m)", "near_shelter": "Near Shelter (<=150 m)", "in_flood_area": "In Flood Cluster (<=100 m)"}
                )
                grp_lbls = (
                    {"denser_share": "5 กล้องที่รถแน่นขึ้น", "other_share": "71 กล้องที่เหลือ"}
                    if lang == "th"
                    else {"denser_share": "5 Denser Cameras", "other_share": "71 Other Cameras"}
                )
                m["feature_label"] = m["feature"].map(feat_lbls).fillna(m["feature"])
                m["group_label"] = m["group"].map(grp_lbls)
                y_pct_lbl = "% ของกล้อง" if lang == "th" else "% of Cameras"
                fig_d = px.bar(m, x="feature_label", y="share_pct", color="group_label", barmode="group",
                               color_discrete_map={grp_lbls["denser_share"]: "#ff7887", grp_lbls["other_share"]: "#5bd3dd"},
                               labels={"feature_label": "", "share_pct": y_pct_lbl, "group_label": ""})
                fig_d.update_layout(height=340, margin=dict(l=15, r=15, t=24, b=64),
                                    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_d, width="stretch")

        with h2:
            st.subheader(tr("hyp_3", lang))
            flood = gold_table("flood_sensor_hours")
            if not flood.empty:
                flood_cp = flood.copy()
                flood_cp["hour_ts"] = pd.to_datetime(flood_cp["hour_ts"]).dt.tz_localize(None)
                phet = hours[hours["camera_id"].isin([250, 256])].merge(flood_cp[["hour_ts", "flood_cm"]], on="hour_ts", how="inner")
                if not phet.empty:
                    comp = phet.groupby("camera_id", as_index=False)[["car", "car_base", "motorcycle", "motorcycle_base"]].mean()
                    cm = comp.melt(id_vars="camera_id", var_name="series", value_name="avg_vehicles")
                    cm["camera_label"] = cm["camera_id"].map(
                        {250: "กล้อง 250 (เพชรบุรี 38)", 256: "กล้อง 256 (เพชรบุรี 27)"}
                        if lang == "th"
                        else {250: "Cam 250 (Phetchaburi 38)", 256: "Cam 256 (Phetchaburi 27)"}
                    )
                    cm["series_name"] = cm["series"].map(
                        {"car": "รถยนต์ (น้ำท่วม >=5 ซม.)", "car_base": "รถยนต์ (ค่าฐานแห้ง)",
                         "motorcycle": "มอเตอร์ไซค์ (น้ำท่วม >=5 ซม.)", "motorcycle_base": "มอเตอร์ไซค์ (ค่าฐานแห้ง)"}
                        if lang == "th"
                        else {"car": "Cars (Flood >= 5cm)", "car_base": "Cars (Dry Base)",
                              "motorcycle": "Motorcycles (Flood >= 5cm)", "motorcycle_base": "Motorcycles (Dry Base)"}
                    )
                    y_avg_lbl = "รถเฉลี่ย / เฟรม" if lang == "th" else "Mean vehicles/frame"
                    fig_p = px.bar(cm, x="camera_label", y="avg_vehicles", color="series_name", barmode="group",
                                   labels={"camera_label": "", "avg_vehicles": y_avg_lbl, "series_name": ""})
                    fig_p.update_layout(height=340, margin=dict(l=15, r=15, t=24, b=64),
                                        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_p, width="stretch")
                st.caption(tr("hyp_3_cap", lang))

        f1, f2 = st.columns(2)
        with f1:
            st.subheader(tr("hyp_4", lang))
            traffy_df = load_traffy()
            if not traffy_df.empty:
                cats = (
                    ["ถนน (is_road)", "รถติด (is_traffic)", "น้ำท่วม (is_flood)", "ทางเท้า (is_sidewalk)"]
                    if lang == "th"
                    else ["Road (is_road)", "Traffic (is_traffic)", "Flood (is_flood)", "Sidewalk (is_sidewalk)"]
                )
                cat_lbl = "หมวดหมู่" if lang == "th" else "Category"
                cnt_lbl = "จำนวนเรื่อง" if lang == "th" else "Count"
                t_counts = pd.DataFrame({
                    cat_lbl: cats,
                    cnt_lbl: [int(traffy_df[c].sum()) for c in ("is_road", "is_traffic", "is_flood", "is_sidewalk")],
                })
                fig_t = px.bar(t_counts, x=cnt_lbl, y=cat_lbl, orientation="h", color=cat_lbl, text_auto=True)
                fig_t.update_layout(height=300, showlegend=False, margin=dict(l=15, r=25, t=20, b=30),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_t, width="stretch")
        with f2:
            st.subheader(tr("hyp_5", lang))
            fm_df = load_fm91()
            if not fm_df.empty:
                daily = fm_df.groupby("date", as_index=False)[["mentions_rain", "mentions_flood", "mentions_congestion", "mentions_accident"]].sum()
                dm = daily.melt(id_vars="date", var_name="topic", value_name="articles")
                topic_lbls = (
                    {"mentions_rain": "ฝนตก", "mentions_flood": "น้ำท่วม", "mentions_congestion": "รถติด", "mentions_accident": "อุบัติเหตุ"}
                    if lang == "th"
                    else {"mentions_rain": "Rain", "mentions_flood": "Flood", "mentions_congestion": "Congestion", "mentions_accident": "Accident"}
                )
                dm["Topic"] = dm["topic"].map(topic_lbls)
                fig_fm = px.line(
                    dm, x="date", y="articles", color="Topic", markers=True,
                    labels={"date": "วันที่" if lang == "th" else "Date", "articles": "จำนวนข่าว" if lang == "th" else "News Articles", "Topic": ""},
                )
                fig_fm.update_layout(height=300, margin=dict(l=15, r=15, t=20, b=56),
                                     legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fm, width="stretch")

    # ---------------- TAB 7: DATA QUALITY & CAUSAL AUDIT ----------------
    with tabs[6]:
        src_name, n_imgs = get_lake_counts_total()
        c_cols = st.columns(3)
        c_cols[0].metric(tr("audit_rows", lang), f"{n_imgs:,}", help=f"Source: lake.{src_name}")
        c_cols[1].metric(tr("audit_cams", lang), f"{int(overall.loc['total', 'n_cameras']):,} / 100")
        c_cols[2].metric(tr("audit_wet", lang), f"{int(overall.loc['total', 'n_wet_hours']):,}")

        st.subheader(tr("audit_hdr", lang))
        st.dataframe(pd.DataFrame(UI[lang]["scorecard"]), width="stretch", hide_index=True)

        with st.expander(tr("audit_exp", lang)):
            g1, g2 = st.columns(2)
            for idx, fname in enumerate(("city_effect", "camera_map", "rebound", "natural_experiment", "denser_share", "mealtime")):
                fp = FIGURES / f"{fname}.png"
                if fp.is_file():
                    (g1 if idx % 2 == 0 else g2).image(str(fp), caption=fp.name, width="stretch")

        st.caption(tr("audit_footer", lang))


if __name__ == "__main__":
    render_dashboard()
