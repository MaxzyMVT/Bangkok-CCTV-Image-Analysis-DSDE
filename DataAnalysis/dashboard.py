"""Interactive Streamlit dashboard for Bangkok rain and traffic analysis.

Run with:
    streamlit run DataAnalysis/dashboard.py
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "DataEngineering" / "WebScraping" / "raw"
OUTPUT = ROOT / "DataAnalysis" / "output"
FIGURES = ROOT / "DataAnalysis" / "figures"

LABEL_TH = {
    "clears": "รถลดชัดเจน",
    "denser": "รถเพิ่มชัดเจน",
    "unaffected": "ยังไม่ชัดเจน",
    "too few wet hours": "ฝนน้อย (< 3 ชม.)",
    "no matched rainy hours": "ไม่มีชั่วโมงฝนที่จับคู่ได้",
}

LABEL_RGB = {
    "clears": [91, 211, 221, 220],
    "denser": [255, 120, 135, 240],
    "unaffected": [138, 212, 173, 210],
    "too few wet hours": [247, 170, 88, 200],
    "no matched rainy hours": [119, 139, 160, 180],
}

LABEL_HEX = {
    "clears": "#5bd3dd",
    "denser": "#ff7887",
    "unaffected": "#8ad4ad",
    "too few wet hours": "#f7aa58",
    "no matched rainy hours": "#778ba0",
}

CLASS_TH = {
    "total": "รถทั้งหมด",
    "car": "รถยนต์",
    "motorcycle": "จักรยานยนต์",
    "bus": "รถบัส",
    "truck": "รถบรรทุก",
}

SEGMENT_TH = {
    "weekday": "วันธรรมดา (Weekday)",
    "weekend": "สุดสัปดาห์ (Weekend)",
    "light_1_to_5_mm": "ฝนเบา 1–5 มม./ชม.",
    "heavy_5plus_mm": "ฝนหนัก ≥ 5 มม./ชม.",
}

WINDOW_TH = {
    "all": "ทุกช่วงเวลา (All)",
    "rush": "ชั่วโมงเร่งด่วน (Rush)",
    "other": "นอกเวลาเร่งด่วน (Other)",
}


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    """Load an analysis output CSV if present."""
    path = OUTPUT / f"{name}.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_camera_locations() -> pd.DataFrame:
    """Load unique camera Thai/English names from CCTV metadata."""
    path = ROOT / "DataEngineering" / "cctv_images_metadata.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["camera_id", "location_th", "location_en"])
    df = pd.read_csv(path, usecols=["camera_id", "location_th", "location_en"])
    return df.drop_duplicates("camera_id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_traffy() -> pd.DataFrame:
    """Load Traffy Fondue complaints with category flags and coordinates."""
    path = RAW / "traffy_fondue" / "traffy_fondue.csv"
    if not path.is_file():
        return pd.DataFrame()
    cols = ["ticket_id", "timestamp", "district", "latitude", "longitude",
            "is_flood", "is_road", "is_traffic", "is_sidewalk"]
    df = pd.read_csv(path, usecols=cols)
    for col in ("is_flood", "is_road", "is_traffic", "is_sidewalk"):
        df[col] = df[col].fillna(False).astype(bool)
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_fm91() -> pd.DataFrame:
    """Load FM91 traffic news articles with parsed timestamps."""
    path = RAW / "fm91_news" / "fm91_news.csv"
    if not path.is_file():
        return pd.DataFrame()
    cols = ["article_id", "datetime", "category", "title",
            "mentions_rain", "mentions_flood", "mentions_congestion", "mentions_accident"]
    df = pd.read_csv(path, usecols=cols)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"] = df["datetime"].dt.date
    for col in ("mentions_rain", "mentions_flood", "mentions_congestion", "mentions_accident"):
        df[col] = df[col].fillna(False).astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_rail_stations() -> pd.DataFrame:
    """Load BTS/MRT/ARL rail stations from OpenStreetMap POIs."""
    path = RAW / "openstreetmap" / "osm_poi.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["name", "name_en", "network", "latitude", "longitude"])
    df = pd.read_csv(path, usecols=["category", "name", "name_en", "network", "latitude", "longitude"])
    rail = df[df["category"] == "rail_station"].dropna(subset=["latitude", "longitude"]).copy()
    rail["station_label"] = rail["name"].fillna(rail["name_en"]).fillna("Rail station")
    return rail.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_districts_geojson() -> dict | None:
    """Load Bangkok 50 district polygons when requested on the spatial map."""
    path = RAW / "bma_open_data" / "bma_districts.geojson"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    """Format signed percentage value or em-dash when missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1f}%"


def show_figure(name: str, caption: str) -> None:
    """Render committed Matplotlib verification figure when present."""
    path = FIGURES / f"{name}.png"
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")


def haversine_matrix_m(lat1: np.ndarray, lon1: np.ndarray,
                       lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compute pairwise Haversine distances in meters between two coordinate arrays."""
    r = 6_371_000.0
    p1 = np.radians(lat1)[:, None]
    p2 = np.radians(lat2)[None, :]
    dp = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def plot_ci_forest(df: pd.DataFrame, group_col: str, group_labels: dict[str, str],
                   title: str, x_title: str = "การเปลี่ยนแปลงเมื่อฝนตก (%)") -> go.Figure:
    """Build grouped horizontal 95% confidence-interval forest plot with Plotly."""
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
            extra = f" | ชม.นาฬิกา: {n_clk}" if n_clk is not None else ""
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
            mode="markers+text",
            text=[pct(v) for v in sub[val_col]],
            textposition="top center",
            name=CLASS_TH.get(cls, cls),
            marker=dict(size=10, color=color_map.get(cls, "#a9bed0")),
            customdata=custom,
            hovertemplate=(
                "<b>%{y} — " + CLASS_TH.get(cls, cls) + "</b><br>"
                "ค่าประมาณ: %{x:+.1f}%<br>"
                "95% CI: [%{customdata[0]:+.1f}%, %{customdata[1]:+.1f}%]<br>"
                "ชั่วโมง: %{customdata[2]}%{customdata[3]}<extra></extra>"
            ),
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="#7993a9")
    fig.update_yaxes(categoryorder="array", categoryarray=[group_labels.get(g, str(g)) for g in reversed(groups)])
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="",
        height=max(260, 110 * len(groups)),
        margin=dict(l=20, r=20, t=48, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_rain_timing(timing_df: pd.DataFrame) -> go.Figure:
    """Build grouped weekday vs weekend wet camera-hour chart with effect hover info."""
    view = timing_df.copy()
    view["day_label"] = view["day_type"].map({"weekday": "วันธรรมดา", "weekend": "สุดสัปดาห์"})
    view["hour_label"] = view["hour"].map(lambda h: f"{int(h):02d}:00")
    fig = px.bar(
        view,
        x="hour_label",
        y="n_camera_hours",
        color="day_label",
        barmode="group",
        color_discrete_map={"วันธรรมดา": "#5bd3dd", "สุดสัปดาห์": "#f7aa58"},
        custom_data=["n_clock_hours", "motorcycle_effect_pct"],
        labels={"hour_label": "เวลา (ชั่วโมงของวัน)", "n_camera_hours": "จำนวนชั่วโมงกล้องที่ฝนตก", "day_label": "ประเภทวัน"},
        title="ชั่วโมงที่ฝนตกแทบไม่ซ้อนทับกันระหว่างวันธรรมดา vs สุดสัปดาห์",
    )
    fig.update_traces(hovertemplate=(
        "<b>%{x} (%{fullData.name})</b><br>"
        "ชั่วโมงกล้อง: %{y}<br>"
        "ชั่วโมงนาฬิกา: %{customdata[0]}<br>"
        "ผลต่อจักรยานยนต์: %{customdata[1]:+.1f}%<extra></extra>"
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=48, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_hourly_dual_axis(trend_df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Build dual-axis chart with selected vehicle classes (lines) and rainfall (bars)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {"total": "#8ad4ad", "car": "#5bd3dd", "motorcycle": "#f7aa58", "bus": "#c792ea", "truck": "#ff7887"}

    fig.add_trace(
        go.Bar(
            x=trend_df["hour_ts"],
            y=trend_df["mean_rain_mm"],
            name="ฝนเฉลี่ย (มม./ชม.)",
            marker_color="rgba(91, 211, 221, 0.28)",
            hovertemplate="%{x|%d %b %H:%M}<br>ฝนเฉลี่ย: %{y:.2f} มม./ชม.<extra></extra>",
        ),
        secondary_y=True,
    )
    for cls in metrics:
        fig.add_trace(
            go.Scatter(
                x=trend_df["hour_ts"],
                y=trend_df[cls],
                mode="lines+markers",
                name=CLASS_TH.get(cls, cls),
                line=dict(width=2.5, color=colors.get(cls, "#ffffff")),
                marker=dict(size=5),
                customdata=np.column_stack([trend_df["cameras"], trend_df["wet_cameras"]]),
                hovertemplate=(
                    "<b>" + CLASS_TH.get(cls, cls) + "</b><br>"
                    "%{x|%d %b %H:%M}<br>"
                    "เฉลี่ย: %{y:.2f} คัน/ภาพ<br>"
                    "กล้องทำงาน: %{customdata[0]} (ฝนตก %{customdata[1]})<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    fig.update_yaxes(title_text="จำนวนรถเฉลี่ย (คัน / ภาพ)", secondary_y=False)
    fig.update_yaxes(title_text="ฝนเฉลี่ยใกล้กล้อง (มม. / ชม.)", secondary_y=True, showgrid=False)
    fig.update_layout(
        title="ปริมาณรถเฉลี่ยต่อภาพเทียบกับปริมาณน้ำฝนรายชั่วโมง",
        height=430,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=48, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_rebound(rebound_df: pd.DataFrame) -> go.Figure:
    """Plot post-rain recovery curve relative to matched dry baseline."""
    view = rebound_df.copy()
    view["cls_th"] = view["cls"].map(lambda c: CLASS_TH.get(c, c))
    fig = px.line(
        view,
        x="minute",
        y="ratio",
        color="cls_th",
        markers=True,
        color_discrete_map={"รถยนต์": "#5bd3dd", "จักรยานยนต์": "#f7aa58"},
        custom_data=["n_events"],
        labels={"minute": "นาทีเทียบกับเวลาเริ่มฝนตก (t = 0)", "ratio": "อัตราส่วนเทียบระดับปกติ (1.0 = ฐานแห้ง)", "cls_th": "ประเภทรถ"},
        title="การฟื้นตัวของรถหลังเริ่มฝนตก (-60 ถึง +180 นาที)",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#7993a9")
    fig.add_vline(x=0, line_dash="dot", line_color="#ff7887")
    fig.update_traces(hovertemplate="%{x:+d} นาที · อัตราส่วน %{y:.3f}x (เหตุการณ์: %{customdata[0]})<extra></extra>")
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=48, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_camera_rail_network(cameras_df: pd.DataFrame, rail_df: pd.DataFrame,
                              rail_radius_m: int, cam_radius_m: int) -> tuple[go.Figure, go.Figure, pd.DataFrame]:
    """Construct spatial multimodal camera-rail NetworkX graph and compute centrality metrics."""
    G = nx.Graph()
    cams = cameras_df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    for row in cams.itertuples(index=False):
        nid = f"CAM_{int(row.camera_id)}"
        label_name = row.location_th if getattr(row, "location_th", "") else f"กล้อง {int(row.camera_id)}"
        G.add_node(
            nid,
            kind="camera",
            camera_id=int(row.camera_id),
            name=f"#{int(row.camera_id)} {label_name}",
            lat=float(row.latitude),
            lon=float(row.longitude),
            effect_pct=None if pd.isna(row.effect_pct) else float(row.effect_pct),
            effect_label=row.label,
        )

    cam_coords = cams[["latitude", "longitude"]].to_numpy()
    if len(cams) > 1:
        cc_dist = haversine_matrix_m(cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 0], cam_coords[:, 1])
        i_idx, j_idx = np.where((cc_dist > 0) & (cc_dist <= cam_radius_m))
        for i, j in zip(i_idx, j_idx):
            if i < j:
                u = f"CAM_{int(cams.iloc[i]['camera_id'])}"
                v = f"CAM_{int(cams.iloc[j]['camera_id'])}"
                G.add_edge(u, v, edge_type="corridor", dist_m=float(cc_dist[i, j]))

    if not rail_df.empty and not cams.empty:
        cr_dist = haversine_matrix_m(
            cam_coords[:, 0], cam_coords[:, 1],
            rail_df["latitude"].to_numpy(), rail_df["longitude"].to_numpy(),
        )
        linked_rail_idx = np.unique(np.where(cr_dist <= rail_radius_m)[1])
        for r_idx in linked_rail_idx:
            r_row = rail_df.iloc[r_idx]
            rid = f"RAIL_{r_idx}"
            G.add_node(
                rid,
                kind="rail",
                name=str(r_row["station_label"]),
                lat=float(r_row["latitude"]),
                lon=float(r_row["longitude"]),
                effect_pct=None,
                effect_label="rail",
            )
        ci_idx, ri_idx = np.where(cr_dist <= rail_radius_m)
        for ci, ri in zip(ci_idx, ri_idx):
            G.add_edge(
                f"CAM_{int(cams.iloc[ci]['camera_id'])}",
                f"RAIL_{ri}",
                edge_type="transit",
                dist_m=float(cr_dist[ci, ri]),
            )

    deg = nx.degree_centrality(G) if G.number_of_nodes() else {}
    btw = nx.betweenness_centrality(G) if G.number_of_nodes() else {}
    comm_map: dict[str, int] = {}
    if G.number_of_edges() > 0:
        for cid, comp in enumerate(nx.community.greedy_modularity_communities(G)):
            for n in comp:
                comm_map[n] = cid + 1
    else:
        for i, n in enumerate(G.nodes()):
            comm_map[n] = i + 1

    rows = []
    for n, attr in G.nodes(data=True):
        rows.append({
            "node_id": n,
            "kind": attr["kind"],
            "camera_id": attr.get("camera_id"),
            "name": attr["name"],
            "lat": attr["lat"],
            "lon": attr["lon"],
            "degree": G.degree(n),
            "degree_centrality": round(deg.get(n, 0.0), 4),
            "betweenness_centrality": round(btw.get(n, 0.0), 4),
            "community": comm_map.get(n, 0),
            "effect_pct": attr.get("effect_pct"),
            "effect_label": attr.get("effect_label"),
        })
    metrics_df = pd.DataFrame(rows)

    fig_net = go.Figure()
    for etype, color, dash in (("corridor", "rgba(169,190,208,0.28)", "solid"),
                               ("transit", "rgba(247,170,88,0.45)", "dot")):
        ex, ey = [], []
        for u, v, edata in G.edges(data=True):
            if edata.get("edge_type") != etype:
                continue
            ex.extend([G.nodes[u]["lon"], G.nodes[v]["lon"], None])
            ey.extend([G.nodes[u]["lat"], G.nodes[v]["lat"], None])
        if ex:
            fig_net.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(width=1.2, color=color, dash=dash),
                hoverinfo="none",
                name="เชื่อมสายทางกล้อง" if etype == "corridor" else "เชื่อมสถานีรถไฟฟ้า",
            ))

    cam_nodes = metrics_df[metrics_df["kind"] == "camera"].copy()
    cam_nodes["label_th"] = cam_nodes["effect_label"].map(lambda x: LABEL_TH.get(x, x))
    for lbl in cam_nodes["effect_label"].unique():
        sub = cam_nodes[cam_nodes["effect_label"] == lbl]
        fig_net.add_trace(go.Scatter(
            x=sub["lon"], y=sub["lat"],
            mode="markers",
            name=LABEL_TH.get(lbl, lbl),
            marker=dict(
                size=7 + sub["degree"] * 1.6,
                color=LABEL_HEX.get(lbl, "#778ba0"),
                line=dict(width=1, color="#0a1422"),
            ),
            customdata=np.column_stack([
                sub["name"], sub["degree"], sub["betweenness_centrality"],
                sub["community"], sub["effect_pct"].fillna(np.nan),
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "ระดับเชื่อมต่อ (Degree): %{customdata[1]}<br>"
                "Betweenness: %{customdata[2]:.4f}<br>"
                "กลุ่มพื้นที่ (Community): #%{customdata[3]}<br>"
                "ผลของฝน: %{customdata[4]:+.1f}%<extra></extra>"
            ),
        ))

    rail_nodes = metrics_df[metrics_df["kind"] == "rail"]
    if not rail_nodes.empty:
        fig_net.add_trace(go.Scatter(
            x=rail_nodes["lon"], y=rail_nodes["lat"],
            mode="markers",
            name="สถานี BTS/MRT (ในรัศมี)",
            marker=dict(symbol="diamond", size=9, color="#ffd166", line=dict(width=1, color="#0a1422")),
            customdata=np.column_stack([rail_nodes["name"], rail_nodes["degree"]]),
            hovertemplate="<b>สถานี %{customdata[0]}</b><br>กล้องใกล้สถานี: %{customdata[1]}<extra></extra>",
        ))

    fig_net.update_layout(
        title="โครงข่ายเชื่อมโยงกล้อง CCTV และสถานีรถไฟฟ้าตามพิกัดจริง (ขนาดจุด = Degree)",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=500,
        margin=dict(l=20, r=20, t=48, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_net.update_yaxes(scaleanchor="x", scaleratio=1)

    tested_cams = cam_nodes.dropna(subset=["effect_pct"]).copy()
    fig_scatter = px.scatter(
        tested_cams,
        x="degree",
        y="effect_pct",
        color="label_th",
        size="betweenness_centrality",
        size_max=22,
        color_discrete_map={LABEL_TH[k]: v for k, v in LABEL_HEX.items()},
        hover_name="name",
        hover_data={"community": True, "betweenness_centrality": ":.4f", "effect_pct": ":+.1f"},
        labels={"degree": "จำนวนจุดเชื่อมในโครงข่าย (Degree)", "effect_pct": "ผลต่อปริมาณรถเมื่อฝนตก (%)", "label_th": "ผลของกล้อง"},
        title="จุดเชื่อมโยงสูง (Hub) ตอบสนองต่อฝนต่างจากจุดอื่นหรือไม่",
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#7993a9")
    fig_scatter.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=48, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig_net, fig_scatter, metrics_df


def build_pydeck_map(cameras_df: pd.DataFrame, rail_df: pd.DataFrame,
                     traffy_df: pd.DataFrame, traffy_layer_col: str | None,
                     show_rail: bool, show_districts: bool) -> pdk.Deck:
    """Build layered interactive PyDeck spatial map of Bangkok cameras and context layers."""
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
        rail_view["tooltip_title"] = "สถานี " + rail_view["station_label"].astype(str)
        rail_view["tooltip_body"] = rail_view["network"].fillna("Rail transit").astype(str)
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
        lambda r: f"กล้อง {int(r['camera_id'])}" + (f" · {r['location_th']}" if r.get("location_th") else ""),
        axis=1,
    )
    cam_view["tooltip_body"] = cam_view.apply(
        lambda r: (
            f"สถานะ: {LABEL_TH.get(r['label'], r['label'])} | "
            f"ผล: {pct(r['effect_pct'])} | "
            f"ฝน {int(r['n_wet_hours'])} ชม. | "
            f"รถไฟฟ้า {int(r['rail_dist_m'])} ม."
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
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11.0, pitch=0),
        tooltip={"html": "<b>{tooltip_title}</b><br/>{tooltip_body}", "style": {"backgroundColor": "#122237", "color": "#eef5fa"}},
    )


def render_dashboard() -> None:
    """Render the interactive Streamlit rain-and-traffic analytics dashboard."""
    st.set_page_config(page_title="Bangkok Rain & Traffic Insights", layout="wide")
    st.title("ฝนเปลี่ยนการจราจรกรุงเทพฯ อย่างไร")
    st.caption("100 กล้อง CCTV BMA · 25–31 ส.ค. 2026 · Interactive Plotly, PyDeck Spatial & NetworkX Analysis")

    required = ["camera", "camera_hour", "city_effect", "camera_effect", "natural_experiment"]
    missing = [name for name in required if table(name).empty]
    if missing:
        st.error("ไม่พบตารางผลวิเคราะห์: " + ", ".join(missing) + " — รัน python DataAnalysis/rain/run_all.py")
        st.stop()

    city = table("city_effect")
    stratified = table("stratified_effects")
    timing = table("rain_timing")
    overall = city[city["window"] == "all"].set_index("cls")
    hours = table("camera_hour")
    hours["hour_ts"] = pd.to_datetime(hours["hour_ts"])
    hours["date"] = hours["hour_ts"].dt.date

    cam_meta = load_camera_locations()
    cameras = table("camera").merge(table("camera_effect"), on="camera_id", how="left").merge(
        cam_meta, on="camera_id", how="left",
    )
    cameras["label"] = cameras["label"].fillna("no matched rainy hours")
    cameras["n_wet_hours"] = cameras["n_wet_hours"].fillna(0).astype(int)
    cameras["location_th"] = cameras["location_th"].fillna("")
    cameras["location_en"] = cameras["location_en"].fillna("")

    st.info(
        "ผลหลัก: ในชั่วโมงที่ฝนตก รถจักรยานยนต์ในภาพลดลงราว 20.98% (95% CI -27.4% ถึง -14.4%) "
        "เมื่อเทียบกล้องเดิม ชั่วโมงเดิม และประเภทวันเดียวกัน ส่วนรถยนต์โดยรวม (-2.24%, CI -6.4% ถึง +2.2%) ยังไม่เปลี่ยนชัด"
    )
    cols = st.columns(4)
    for col, cls, label in zip(cols[:3], ("total", "car", "motorcycle"), ("รถทั้งหมด", "รถยนต์", "จักรยานยนต์")):
        row = overall.loc[cls]
        col.metric(label, pct(row["effect_pct"]))
        col.caption(f"95% CI {pct(row['ci_low'])} ถึง {pct(row['ci_high'])} · {int(row['n_wet_hours'])} ชม.กล้อง")
    overlap_hours = 0
    if not timing.empty:
        wd = set(timing[timing["day_type"] == "weekday"]["hour"])
        we = set(timing[timing["day_type"] == "weekend"]["hour"])
        overlap_hours = len(wd & we)
    cols[3].metric("ชั่วโมงที่ฝนตกทั้งสองประเภทวัน", f"{overlap_hours} ชม.")
    cols[3].caption("ซ้อนกันเฉพาะ 16:00 และ 17:00 น. (94 กล้องมีคู่เทียบ)")

    tab_overview, tab_time, tab_map, tab_network, tab_hypotheses, tab_data = st.tabs([
        "ผลหลัก & การแบ่งกลุ่ม",
        "ช่วงเวลา & หลังฝนหยุด",
        "แผนที่เชิงพื้นที่ (PyDeck)",
        "เครือข่ายกล้อง–รถไฟฟ้า (NetworkX)",
        "สมมติฐาน, น้ำท่วม & ข่าว",
        "ข้อมูล, ผังแนวคิด & ข้อจำกัด",
    ])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_ci_forest(city, "window", WINDOW_TH, "ผลต่างปริมาณรถเมื่อฝนตก แยกตามช่วงเวลา"),
                width="stretch",
            )
        with c2:
            if not stratified.empty:
                st.plotly_chart(
                    plot_ci_forest(stratified, "segment", SEGMENT_TH, "การแบ่งกลุ่ม: วันธรรมดา/สุดสัปดาห์ & ความแรงฝน"),
                    width="stretch",
                )
        if not timing.empty:
            st.plotly_chart(plot_rain_timing(timing), width="stretch")
        st.warning(
            "ข้อควรระวังเรื่องเวลาตก: ฝนวันธรรมดาเกิด 69 ชม.กล้อง (8 ชม.นาฬิกา, ช่วง 16:00–20:00) "
            "ส่วนสุดสัปดาห์เกิด 224 ชม.กล้อง (8 ชม.นาฬิกา, ช่วง 12:00–17:00) จึงซ้อนกันเพียง 16:00–17:00 น."
        )
        with st.expander("ดูตารางผลสรุป (city_effect & stratified_effects)"):
            st.dataframe(city, width="stretch", hide_index=True)
            if not stratified.empty:
                st.dataframe(stratified, width="stretch", hide_index=True)

    with tab_time:
        dates = sorted(hours["date"].unique())
        f1, f2 = st.columns([2, 2])
        chosen = f1.multiselect("เลือกวันที่", dates, default=dates)
        metrics = f2.multiselect(
            "ประเภทรถที่แสดงบนกราฟ",
            ["motorcycle", "car", "total", "bus", "truck"],
            default=["motorcycle", "car"],
            format_func=lambda x: CLASS_TH.get(x, x),
        )
        subset = hours[hours["date"].isin(chosen)]
        if not subset.empty and metrics:
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
            st.plotly_chart(plot_hourly_dual_axis(trend, metrics), width="stretch")
            rebound = table("rebound")
            if not rebound.empty:
                st.plotly_chart(plot_rebound(rebound), width="stretch")
            with st.expander("ดูข้อมูลรายชั่วโมง (Hourly Aggregated Table)"):
                st.dataframe(trend, width="stretch", hide_index=True)
        else:
            st.info("กรุณาเลือกอย่างน้อย 1 วัน และ 1 ประเภทรถ")

    with tab_map:
        rail_df = load_rail_stations()
        traffy_df = load_traffy()
        all_labels = sorted(cameras["label"].unique())
        mc1, mc2, mc3, mc4 = st.columns([2, 1.2, 1.5, 1.3])
        selected_labels = mc1.multiselect(
            "กรองผลของกล้อง",
            all_labels,
            default=all_labels,
            format_func=lambda x: f"{LABEL_TH.get(x, x)} ({x})",
        )
        min_wet = mc2.slider("ชั่วโมงฝนขั้นต่ำ", 0, int(cameras["n_wet_hours"].max()), 0)
        heat_options = {"ไม่แสดง Heatmap": None, "จุดร้องเรียนน้ำท่วม (Flood)": "is_flood",
                        "จุดร้องเรียนรถติด (Traffic)": "is_traffic", "ปัญหาถนน (Road)": "is_road",
                        "ปัญหาทางเท้า (Sidewalk)": "is_sidewalk"}
        heat_choice = mc3.selectbox("Traffy Heatmap Overlay", list(heat_options.keys()))
        show_rail = mc4.checkbox("แสดงสถานี BTS/MRT", value=True)
        show_districts = mc4.checkbox("แสดงขอบเขต 50 เขต กทม.", value=False)

        filtered = cameras[cameras["label"].isin(selected_labels) & (cameras["n_wet_hours"] >= min_wet)].copy()
        st.caption(f"แสดง {len(filtered)} / {len(cameras)} กล้อง · ฟ้า=รถลดชัดเจน, แดง=รถเพิ่มชัดเจน, เขียว=ไม่ชัดเจน, ส้ม=ฝน<3ชม., เทา=ไม่มีฝนจับคู่")
        if not filtered.empty:
            st.pydeck_chart(
                build_pydeck_map(filtered, rail_df, traffy_df, heat_options[heat_choice], show_rail, show_districts),
                width="stretch",
            )
            display_cols = ["camera_id", "location_th", "label", "effect_pct", "ci_low", "ci_high",
                            "n_wet_hours", "gauge", "gauge_dist_m", "rail_dist_m", "near_rail", "near_shelter"]
            avail_cols = [c for c in display_cols if c in filtered.columns]
            st.dataframe(filtered[avail_cols].sort_values("effect_pct", na_position="last"), width="stretch", hide_index=True)
        else:
            st.info("ไม่มีกล้องที่ตรงกับเงื่อนไขตัวกรอง")

    with tab_network:
        rail_df = load_rail_stations()
        st.subheader("วิเคราะห์โครงข่ายกล้อง CCTV และสถานีรถไฟฟ้า (Week 14 Spatial-Network Analysis)")
        nc1, nc2 = st.columns(2)
        rail_radius = nc1.slider("รัศมีเชื่อม กล้อง ↔ สถานี BTS/MRT (เมตร)", 300, 1500, 800, step=100)
        cam_radius = nc2.slider("รัศมีเชื่อม กล้อง ↔ กล้องตามแนวถนน (เมตร)", 500, 2000, 1200, step=100)

        fig_net, fig_scatter, net_df = build_camera_rail_network(cameras, rail_df, rail_radius, cam_radius)
        cam_net = net_df[net_df["kind"] == "camera"].copy()
        cam_net["camera_id"] = cam_net["camera_id"].astype("Int64")
        rail_linked = net_df[net_df["kind"] == "rail"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("โหนดกล้อง", f"{len(cam_net)}")
        k2.metric("สถานีรถไฟฟ้าในรัศมี", f"{len(rail_linked)}")
        k3.metric("จำนวนกลุ่มพื้นที่ (Communities)", f"{cam_net['community'].nunique()}")
        k4.metric("จุดเชื่อมต่อสูงสุด (Max Degree)", f"{int(net_df['degree'].max()) if not net_df.empty else 0}")

        g1, g2 = st.columns([1.45, 1])
        with g1:
            st.plotly_chart(fig_net, width="stretch")
        with g2:
            st.plotly_chart(fig_scatter, width="stretch")
        st.write("Top 12 โหนดที่มีความเป็นศูนย์กลางสูงสุด (Degree & Betweenness Centrality)")
        top_hubs = cam_net.sort_values(["betweenness_centrality", "degree"], ascending=False).head(12)
        st.dataframe(
            top_hubs[["camera_id", "name", "degree", "betweenness_centrality", "community", "effect_pct", "effect_label"]],
            width="stretch",
            hide_index=True,
        )

    with tab_hypotheses:
        nat = table("natural_experiment")
        if not nat.empty:
            st.subheader("1) การเทียบกล้องฝนตก vs กล้องแห้งใกล้กัน (≤ 3 กม. ชั่วโมงเดียวกัน)")
            st.plotly_chart(
                plot_ci_forest(nat, "cls", CLASS_TH, "Natural Experiment (15 ชั่วโมง · 172 คู่กล้อง)", "ส่วนต่างสัมพัทธ์ (Percentage Points)"),
                width="stretch",
            )
            st.caption("รถยนต์ในฝั่งกล้องที่ฝนตกสูงกว่ากล้องแห้ง +12.59 pts (95% CI +2.2 ถึง +20.7) แต่พบเพียง 15 ชั่วโมงและแยกแท็กซี่กับรถส่วนตัวไม่ได้")

        h1, h2 = st.columns(2)
        with h1:
            st.subheader("2) สัดส่วนกล้องรถหนาแน่นใกล้รถไฟฟ้า/ที่หลบฝน")
            denser_share = table("denser_share")
            if not denser_share.empty:
                m = denser_share.melt(id_vars="feature", value_vars=["denser_share", "other_share"],
                                      var_name="group", value_name="share")
                m["share_pct"] = m["share"] * 100.0
                m["feature_th"] = m["feature"].map({
                    "near_rail": "ใกล้รถไฟฟ้า (≤500 ม.)",
                    "near_shelter": "ใกล้จุดหลบฝน",
                    "in_flood_area": "อยู่ในกลุ่มจุดน้ำท่วม",
                }).fillna(m["feature"])
                m["group_th"] = m["group"].map({"denser_share": "5 กล้องที่รถหนาแน่นขึ้น", "other_share": "71 กล้องอื่น"})
                fig_d = px.bar(m, x="feature_th", y="share_pct", color="group_th", barmode="group",
                               color_discrete_map={"5 กล้องที่รถหนาแน่นขึ้น": "#ff7887", "71 กล้องอื่น": "#5bd3dd"},
                               labels={"feature_th": "คุณลักษณะพื้นที่", "share_pct": "สัดส่วน (%)", "group_th": "กลุ่มกล้อง"})
                fig_d.update_layout(height=320, margin=dict(l=10, r=10, t=24, b=24),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_d, width="stretch")
            pol = table("policy_stations")
            if not pol.empty:
                st.caption(f"Policy stations (`policy_stations.csv`): ช่องนนทรี มีค่า `moto_lost_per_wet_hour = {pol['moto_lost_per_wet_hour'].iloc[0]:+.4f}` (ติดลบ = จักรยานยนต์เพิ่มขึ้นหลังรวมจุด)")

        with h2:
            st.subheader("3) เซนเซอร์น้ำท่วม ≥ 5 ซม. vs กล้องบนถนนเพชรบุรี")
            flood = table("flood_sensor_hours")
            if not flood.empty:
                phet = hours[hours["camera_id"].isin([250, 256])].merge(
                    flood[["hour_ts", "flood_cm"]].assign(hour_ts=pd.to_datetime(flood["hour_ts"])),
                    on="hour_ts",
                    how="inner",
                )
                if not phet.empty:
                    comp = phet.groupby("camera_id", as_index=False)[["car", "car_base", "motorcycle", "motorcycle_base"]].mean()
                    cm = comp.melt(id_vars="camera_id", var_name="series", value_name="avg_vehicles")
                    cm["camera_label"] = cm["camera_id"].map({250: "กล้อง 250 (ซ.เพชรบุรี 38)", 256: "กล้อง 256 (ซ.เพชรบุรี 27)"})
                    cm["series_th"] = cm["series"].map({
                        "car": "รถยนต์ (ช่วงน้ำท่วม ≥5 ซม.)",
                        "car_base": "รถยนต์ (ฐานแห้ง)",
                        "motorcycle": "มอเตอร์ไซค์ (ช่วงน้ำท่วม ≥5 ซม.)",
                        "motorcycle_base": "มอเตอร์ไซค์ (ฐานแห้ง)",
                    })
                    fig_p = px.bar(cm, x="camera_label", y="avg_vehicles", color="series_th", barmode="group",
                                   labels={"camera_label": "กล้องบนถนนเพชรบุรี", "avg_vehicles": "เฉลี่ย (คัน/ภาพ)", "series_th": "ตัววัด"})
                    fig_p.update_layout(height=320, margin=dict(l=10, r=10, t=24, b=24),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_p, width="stretch")
                st.caption(f"เซนเซอร์ `FL.RTW.02 ถนนเพชรบุรี (ดอนบอสโก)` มีน้ำ ≥5 ซม. {len(flood)} ชั่วโมง (30–31 ส.ค.) แม้ชั่วโมงนั้นเกจฝนใกล้กล้องเป็น 0 มม.")

        f1, f2 = st.columns(2)
        with f1:
            st.subheader("4) ข้อร้องเรียน Traffy Fondue แยกหมวด")
            traffy_df = load_traffy()
            if not traffy_df.empty:
                t_counts = pd.DataFrame({
                    "หมวดปัญหา": ["ถนน (is_road)", "จราจร (is_traffic)", "น้ำท่วม (is_flood)", "ทางเท้า (is_sidewalk)"],
                    "จำนวนเรื่อง": [int(traffy_df[c].sum()) for c in ("is_road", "is_traffic", "is_flood", "is_sidewalk")],
                })
                fig_t = px.bar(t_counts, x="จำนวนเรื่อง", y="หมวดปัญหา", orientation="h",
                               color="หมวดปัญหา", text_auto=True)
                fig_t.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=20, b=20),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_t, width="stretch")
        with f2:
            st.subheader("5) ข่าวจราจร FM91 รายวัน (25–31 ส.ค.)")
            fm_df = load_fm91()
            if not fm_df.empty:
                daily = fm_df.groupby("date", as_index=False)[
                    ["mentions_rain", "mentions_flood", "mentions_congestion", "mentions_accident"]
                ].sum()
                dm = daily.melt(id_vars="date", var_name="topic", value_name="articles")
                dm["topic_th"] = dm["topic"].map({
                    "mentions_rain": "พูดถึงฝน",
                    "mentions_flood": "พูดถึงน้ำท่วม",
                    "mentions_congestion": "พูดถึงรถติด",
                    "mentions_accident": "พูดถึงอุบัติเหตุ",
                })
                fig_fm = px.line(dm, x="date", y="articles", color="topic_th", markers=True,
                                 labels={"date": "วันที่", "articles": "จำนวนข่าว", "topic_th": "หัวข้อ"})
                fig_fm.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fm, width="stretch")

    with tab_data:
        counts_path = ROOT / "DataAnalysis" / "data" / "counts_v2.csv"
        c_cols = st.columns(3)
        n_imgs = len(pd.read_csv(counts_path, usecols=["image_id"])) if counts_path.is_file() else 0
        c_cols[0].metric("ภาพที่นับใน counts_v2.csv", f"{n_imgs:,}")
        c_cols[1].metric("กล้องที่มีผลจับคู่ฝน", f"{int(overall.loc['total', 'n_cameras']):,} / 100")
        c_cols[2].metric("ชั่วโมงกล้องที่ฝนตก", f"{int(overall.loc['total', 'n_wet_hours']):,}")

        st.subheader("สถานะข้อมูลเทียบกับผังแนวคิด Mermaid")
        scorecard = pd.DataFrame([
            {"โหนดในผังแนวคิด": "ฝนตก (Rain ≥ 1–2 mm)", "สถานะในข้อมูล": "มีข้อมูลจริง (DDS Rain 129 จุด)", "ข้อค้นพบ / ข้อจำกัด": "จับคู่ 293 ชม.กล้อง; รถจักรยานยนต์ลด -21%"},
            {"โหนดในผังแนวคิด": "มอเตอร์ไซค์จอดหลบใต้สะพาน/ทางด่วน", "สถานะในข้อมูล": "ทดสอบแล้ว (สัดส่วน stationary)", "ข้อค้นพบ / ข้อจำกัด": "ยังไม่พบหลักฐานชัด (2.5% แห้ง vs 2.8% ฝน; ส่วนใหญ่อยู่บนทางเท้า)"},
            {"โหนดในผังแนวคิด": "น้ำท่วมขัง ≥ 5 cm (DDS Flood)", "สถานะในข้อมูล": "ไม่มีพิกัด GPS โดยตรง", "ข้อค้นพบ / ข้อจำกัด": "พบ 13 ชม. ที่ FL.RTW.02 ถ.เพชรบุรี (5.0–5.1 ซม.); เทียบชื่อถนนกับกล้อง 250/256 ยังไม่พบปริมาณรถเพิ่ม"},
            {"โหนดในผังแนวคิด": "Traffy Fondue (ท่อตัน/จราจร/น้ำท่วม)", "สถานะในข้อมูล": "มี 7,247 เรื่อง + พิกัด", "ข้อค้นพบ / ข้อจำกัด": "ส่วนใหญ่เกิดในซอย (มีเพียง 2 กล้องใกล้จุดน้ำท่วม ≤100 ม.)"},
            {"โหนดในผังแนวคิด": "เปลี่ยนไปขึ้น BTS/MRT & รถรับส่ง (+25% Surge)", "สถานะในข้อมูล": "ยังไม่มีข้อมูลโดยตรง", "ข้อค้นพบ / ข้อจำกัด": "Natural experiment พบรถเพิ่ม +12.6 pts ใน 15 ชม. แต่ไม่มียอดผู้โดยสาร/ราคา surge/แท็กซี่"},
            {"โหนดในผังแนวคิด": "ความจุถนนหดตัว & รถติดวิกฤต (Gridlock)", "สถานะในข้อมูล": "วัดได้เฉพาะความหนาแน่นในภาพนิ่ง", "ข้อค้นพบ / ข้อจำกัด": "ภาพนิ่งไม่วัดความเร็ว (km/h) หรือปริมาณรถไหลผ่าน (veh/hr)"},
        ])
        st.dataframe(scorecard, width="stretch", hide_index=True)

        with st.expander("ดูกราฟ Matplotlib ต้นฉบับจาก DataAnalysis/figures/"):
            g1, g2 = st.columns(2)
            with g1:
                show_figure("city_effect", "city_effect.png")
                show_figure("rebound", "rebound.png")
                show_figure("denser_share", "denser_share.png")
            with g2:
                show_figure("camera_map", "camera_map.png")
                show_figure("natural_experiment", "natural_experiment.png")
                show_figure("mealtime", "mealtime.png")

        st.warning(
            "หมายเหตุความสอดคล้องของ Pipeline: `DataAnalysis/data/counts_v2.csv` มี 60,721 ภาพ (รวม 551,501 คัน จาก `YOLO.predict(conf=0.30)` ดิบ) "
            "ในขณะที่ `DataEngineering/vehicle_counts.csv` และ `silver.vehicle_counts` ใน DuckLakehouse ใช้ `YOLOVehicleDetector` (รวม 569,906 คัน). "
            "Power BI Desktop สามารถนำเข้า CSV ทั้งหมดใน `DataAnalysis/output/` ได้โดยตรง."
        )


if __name__ == "__main__":
    render_dashboard()
