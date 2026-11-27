"""NetworkX + PyVis interactive visualization helpers for Bangkok CCTV & Rail network.

Provides layout, centrality, community detection, CCDF degree distribution, and PyVis
rendering backed by PyDuckLake camera and rapid transit coordinates.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colormaps, colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from scipy.spatial import cKDTree
from scipy.stats import rankdata

MEASURES = ["degree", "betweenness", "closeness", "pagerank"]
MEASURE_COLUMN = {
    "degree": "Degree",
    "betweenness": "Betweenness",
    "closeness": "Closeness",
    "pagerank": "PageRank",
}
LAYOUTS = ["forceatlas2", "spring", "geo_coords", "kamada_kawai", "shell_by_degree", "circular"]
EFFECT_HEX = {
    "clears": "#5bd3dd",
    "denser": "#ff7887",
    "unaffected": "#8ad4ad",
    "too few wet hours": "#f7aa58",
    "no matched rainy hours": "#778ba0",
    "rail": "#ffd166",
}
COMMUNITY_PALETTE = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#14b8a6",
    "#6366f1", "#eab308", "#a855f7", "#22c55e", "#0ea5e9",
]

JPG_BUTTON_HTML = """
<button onclick="downloadJPG()" title="Save current graph view as JPG"
  style="position:absolute; top:10px; right:10px; z-index:6; padding:5px 10px; font:12px sans-serif;
         background:#ffffff; color:#111111; border:1px solid #bbbbbb; border-radius:4px; cursor:pointer">Download JPG</button>
<script>
function downloadJPG() {
  var src = document.querySelector('#mynetwork canvas');
  if (!src) return;
  var out = document.createElement('canvas');
  out.width = src.width; out.height = src.height;
  var ctx = out.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(src, 0, 0);
  var a = document.createElement('a');
  a.href = out.toDataURL('image/jpeg', 0.95);
  a.download = 'bangkok_cctv_network.jpg';
  document.body.appendChild(a); a.click(); a.remove();
}
</script>
"""


def _haversine_matrix_m(lat1: np.ndarray, lon1: np.ndarray,
                        lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6_371_000.0
    p1 = np.radians(lat1)[:, None]
    p2 = np.radians(lat2)[None, :]
    dp = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_bangkok_network(
    cameras_df: pd.DataFrame,
    rail_df: pd.DataFrame,
    rail_radius_m: int = 800,
    cam_radius_m: int = 900,
    k_rail: int = 1,
    k_cam: int = 2,
) -> nx.Graph:
    """Build sparse, interpretable CCTV + BTS/MRT corridor network using k-nearest spatial edges."""
    G = nx.Graph()
    cams = cameras_df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    for r in cams.itertuples(index=False):
        cid = int(r.camera_id)
        nid = f"Cam #{cid}"
        loc = getattr(r, "location_th", "") or nid
        eff = None if pd.isna(getattr(r, "effect_pct", np.nan)) else float(r.effect_pct)
        G.add_node(
            nid,
            kind="camera",
            camera_id=cid,
            location_th=loc,
            lat=float(r.latitude),
            lon=float(r.longitude),
            effect_pct=eff,
            effect_label=str(getattr(r, "label", "no matched rainy hours")),
            nearest_station="—",
            nearest_station_m=None,
            tooltip=f"{loc} | Rain effect: {f'{eff:+.1f}%' if eff is not None else 'n/a'}",
        )

    cam_coords = cams[["latitude", "longitude"]].to_numpy()
    if len(cams) > 1 and k_cam > 0:
        cc = _haversine_matrix_m(cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 0], cam_coords[:, 1])
        np.fill_diagonal(cc, np.inf)
        for i in range(len(cams)):
            for j in np.argsort(cc[i])[:k_cam]:
                if cc[i, j] <= cam_radius_m:
                    u = f"Cam #{int(cams.iloc[i]['camera_id'])}"
                    v = f"Cam #{int(cams.iloc[j]['camera_id'])}"
                    if not G.has_edge(u, v):
                        G.add_edge(
                            u, v,
                            edge_type="corridor",
                            weight=round(max(0.1, (cam_radius_m - float(cc[i, j])) / max(cam_radius_m, 1)), 3),
                            distance_m=round(float(cc[i, j])),
                        )

    if not rail_df.empty and not cams.empty and k_rail > 0:
        cr = _haversine_matrix_m(
            cam_coords[:, 0], cam_coords[:, 1],
            rail_df["latitude"].to_numpy(), rail_df["longitude"].to_numpy(),
        )
        rail_ids: dict[int, str] = {}
        edges_to_add: list[tuple[str, int, float]] = []
        for c_idx in range(len(cams)):
            cam_id = f"Cam #{int(cams.iloc[c_idx]['camera_id'])}"
            best_r = int(np.argmin(cr[c_idx]))
            G.nodes[cam_id]["nearest_station"] = str(rail_df.iloc[best_r]["station_label"])
            G.nodes[cam_id]["nearest_station_m"] = round(float(cr[c_idx, best_r]))
            for r_idx in np.argsort(cr[c_idx])[:k_rail]:
                dist_m = float(cr[c_idx, r_idx])
                if dist_m <= rail_radius_m:
                    edges_to_add.append((cam_id, int(r_idx), dist_m))
                    if int(r_idx) not in rail_ids:
                        row = rail_df.iloc[int(r_idx)]
                        rname = f"Rail: {row['station_label']}"
                        rail_ids[int(r_idx)] = rname
                        G.add_node(
                            rname,
                            kind="rail",
                            location_th=str(row["station_label"]),
                            lat=float(row["latitude"]),
                            lon=float(row["longitude"]),
                            effect_pct=None,
                            effect_label="rail",
                            nearest_station=str(row["station_label"]),
                            nearest_station_m=0,
                            tooltip=f"{row.get('network', 'BTS/MRT')} Station",
                        )
        for cam_id, r_idx, dist_m in edges_to_add:
            rname = rail_ids[r_idx]
            if not G.has_edge(cam_id, rname):
                G.add_edge(
                    cam_id, rname,
                    edge_type="transit",
                    weight=round(max(0.1, (rail_radius_m - dist_m) / max(rail_radius_m, 1)), 3),
                    distance_m=round(dist_m),
                )
    return G


def to_simple_undirected(G: nx.Graph) -> nx.Graph:
    U = nx.Graph(G)
    U.remove_edges_from(nx.selfloop_edges(U))
    return U


def largest_component(G: nx.Graph) -> nx.Graph:
    if len(G) == 0 or nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def path_stats(G: nx.Graph) -> tuple[float, int]:
    if len(G) <= 1:
        return 0.0, 0
    total, pairs, diameter = 0, 0, 0
    for s in G:
        dist = nx.single_source_shortest_path_length(G, s)
        total += sum(dist.values())
        pairs += len(dist) - 1
        diameter = max(diameter, max(dist.values()))
    return (total / pairs if pairs else 0.0), diameter


def compute_basic_stats(G: nx.Graph) -> Dict:
    U = to_simple_undirected(G)
    lcc = largest_component(U)
    L, diameter = path_stats(lcc)
    return {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": nx.density(G) if len(G) > 1 else 0.0,
        "Diameter": diameter,
        "Avg Clustering": nx.average_clustering(U) if len(U) else 0.0,
        "Avg Path Length": L,
        "Connected": len(U) == 0 or lcc.number_of_nodes() == U.number_of_nodes(),
        "LCC Size": lcc.number_of_nodes(),
    }


def compare_with_random(G: nx.Graph, n_samples: int = 5) -> pd.DataFrame:
    U = to_simple_undirected(G)
    n, m = U.number_of_nodes(), U.number_of_edges()

    def measure(H):
        return [nx.average_clustering(H), path_stats(largest_component(H))[0]]

    obs = np.array(measure(U))
    rand_mean = np.mean([measure(nx.gnm_random_graph(n, m, seed=s)) for s in range(n_samples)], axis=0)
    return pd.DataFrame(
        {"This network": obs, "Random graph": rand_mean, "Ratio": obs / np.where(rand_mean == 0, np.nan, rand_mean)},
        index=["Avg clustering C", "Avg path length L"],
    )


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    table = {
        "Degree": nx.degree_centrality(G) if len(G) else {},
        "Betweenness": nx.betweenness_centrality(G) if len(G) else {},
        "Closeness": nx.closeness_centrality(G) if len(G) else {},
        "PageRank": nx.pagerank(G) if G.number_of_edges() else {n: 1.0 / max(len(G), 1) for n in G},
    }
    df = pd.DataFrame(table)
    df.index.name = "Node"
    df = df.reset_index()
    df["Kind"] = [G.nodes[n].get("kind", "") for n in df["Node"]]
    df["Rain Effect (%)"] = [G.nodes[n].get("effect_pct") for n in df["Node"]]
    return df


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Detect spatial-corridor communities ordered by largest community size (0 = largest)."""
    U = to_simple_undirected(G)
    if U.number_of_edges() == 0:
        return {n: i for i, n in enumerate(U.nodes())}
    comms = sorted(nx.community.greedy_modularity_communities(U), key=len, reverse=True)
    mapping = {}
    for idx, comp in enumerate(comms):
        for n in comp:
            mapping[n] = idx
    return mapping


def connection_summary_tables(G: nx.Graph, communities: Optional[Dict[str, int]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (camera_connections_df, community_summary_df) explaining exact links & clusters."""
    cam_rows = []
    for n, attr in G.nodes(data=True):
        if attr.get("kind") != "camera":
            continue
        neighbors = list(G.neighbors(n))
        rail_links = [
            f"{nb.removeprefix('Rail: ')} ({G.edges[n, nb].get('distance_m', '?')}m)"
            for nb in neighbors if G.nodes[nb].get("kind") == "rail"
        ]
        cam_links = sorted(
            [(G.edges[n, nb].get("distance_m", 0), nb) for nb in neighbors if G.nodes[nb].get("kind") == "camera"]
        )
        cam_rows.append({
            "Camera": n,
            "Location (TH)": attr.get("location_th", ""),
            "Community": communities.get(n, -1) if communities else None,
            "Linked Stations": ", ".join(rail_links) if rail_links else f"— (closest: {attr.get('nearest_station')} {attr.get('nearest_station_m')}m)",
            "Neighbor Cameras": ", ".join(f"{nb} ({d}m)" for d, nb in cam_links) if cam_links else "Isolated",
            "Degree": G.degree(n),
            "Rain Effect (%)": attr.get("effect_pct"),
            "Status": attr.get("effect_label"),
        })
    cam_df = pd.DataFrame(cam_rows).sort_values(["Community", "Degree"], ascending=[True, False]).reset_index(drop=True)

    comm_rows = []
    if communities:
        by_c: dict[int, list[str]] = {}
        for n, cid in communities.items():
            by_c.setdefault(cid, []).append(n)
        for cid, nodes in sorted(by_c.items()):
            cams = [n for n in nodes if G.nodes[n].get("kind") == "camera"]
            rails = sorted(n.removeprefix("Rail: ") for n in nodes if G.nodes[n].get("kind") == "rail")
            effs = [G.nodes[n]["effect_pct"] for n in cams if G.nodes[n].get("effect_pct") is not None]
            comm_rows.append({
                "Community": cid,
                "Cameras": len(cams),
                "Stations": len(rails),
                "Station Hubs": ", ".join(rails[:5]) + ("..." if len(rails) > 5 else ""),
                "Sample Cameras": ", ".join(cams[:5]) + ("..." if len(cams) > 5 else ""),
                "Mean Rain Effect (%)": round(float(np.mean(effs)), 2) if effs else None,
            })
    comm_df = pd.DataFrame(comm_rows)
    return cam_df, comm_df


def _compute_layout(G: nx.Graph, layout_type: str, scale_factor: float = 1350.0) -> Dict[str, Tuple[float, float]]:
    if len(G) == 0:
        return {}
    if layout_type == "geo_coords":
        lons = np.array([G.nodes[n]["lon"] for n in G])
        lats = np.array([G.nodes[n]["lat"] for n in G])
        span = max(float(np.ptp(lons)), float(np.ptp(lats)), 1e-6)
        cx, cy = float(np.mean(lons)), float(np.mean(lats))
        return {
            n: (float((G.nodes[n]["lon"] - cx) / span * scale_factor * 2.0),
                float(-(G.nodes[n]["lat"] - cy) / span * scale_factor * 2.0))
            for n in G
        }
    U = to_simple_undirected(G)
    if layout_type == "forceatlas2":
        pos = nx.rescale_layout_dict(nx.forceatlas2_layout(U, max_iter=250, seed=42), scale=1.0)
    elif layout_type == "spring":
        pos = nx.spring_layout(U, k=2.8 / np.sqrt(max(len(U), 1)), iterations=80, seed=42)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(U)
    elif layout_type == "shell_by_degree":
        order = sorted(U, key=lambda n: U.degree(n), reverse=True)
        n1 = max(1, round(0.08 * len(U)))
        n2 = n1 + max(1, round(0.25 * len(U)))
        shells = [s for s in (order[:n1], order[n1:n2], order[n2:]) if s]
        pos = nx.shell_layout(U, nlist=shells)
    else:
        pos = nx.circular_layout(U)
    return {n: (float(xy[0] * scale_factor), float(xy[1] * scale_factor)) for n, xy in pos.items()}


def _remove_overlaps(pos: Dict[str, Tuple[float, float]], radius: Dict[str, float],
                     margin: float = 8.0, max_iter: int = 250) -> Dict[str, Tuple[float, float]]:
    if len(pos) < 2:
        return pos
    nodes = list(pos)
    P = np.array([pos[n] for n in nodes], dtype=float)
    R = np.array([radius[n] for n in nodes], dtype=float) + margin / 2.0
    rng = np.random.default_rng(42)
    for _ in range(max_iter):
        pairs = cKDTree(P).query_pairs(2.0 * float(R.max()), output_type="ndarray")
        if len(pairs) == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        delta = P[j] - P[i]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        overlap = R[i] + R[j] - dist
        hit = overlap > 1e-5
        if not hit.any():
            break
        i, j, delta, dist, overlap = i[hit], j[hit], delta[hit], dist[hit], overlap[hit]
        same = dist < 1e-9
        if same.any():
            delta[same] = rng.normal(size=(same.sum(), 2))
            dist[same] = np.hypot(delta[same, 0], delta[same, 1])
        push = delta / dist[:, None] * (overlap / 2.0)[:, None]
        move = np.zeros_like(P)
        np.add.at(move, i, -push)
        np.add.at(move, j, push)
        count = np.bincount(np.concatenate([i, j]), minlength=len(P))[:, None]
        P += 1.2 * move / np.maximum(count, 1)
    return {n: (float(x), float(y)) for n, (x, y) in zip(nodes, P)}


def render_pyvis_html(
    G: nx.Graph,
    cent_df: pd.DataFrame,
    communities: Optional[Dict[str, int]] = None,
    layout: str = "forceatlas2",
    size_by: str = "degree",
    color_by: str = "community",
    node_size_range: Tuple[int, int] = (10, 30),
    scale_factor: float = 1350.0,
    no_overlap: bool = False,
    font_size: int = 13,
    label_top_n: int = 35,
    min_component_size: int = 1,
    show_edges: bool = True,
    selected_community: Optional[int] = None,
) -> str:
    """Render an interactive PyVis graph HTML string for the CCTV and rail network."""
    H = G
    if selected_community is not None and communities:
        keep = [n for n in G if communities.get(n) == selected_community]
        H = G.subgraph(keep).copy()
    elif min_component_size > 1 and len(G):
        keep = [n for c in nx.connected_components(to_simple_undirected(G)) if len(c) >= min_component_size for n in c]
        H = G.subgraph(keep).copy()
    if len(H) == 0:
        return "<html><body><p style='font-family:sans-serif;padding:24px'>No nodes remain after filtering.</p></body></html>"

    pos = _compute_layout(H, layout, scale_factor=scale_factor)
    metric_col = "Degree" if size_by == "single size" else MEASURE_COLUMN.get(size_by, "Degree")
    cent_vals = {r.Node: float(getattr(r, metric_col)) for r in cent_df.itertuples(index=False) if r.Node in H}
    min_c, max_c = min(cent_vals.values()), max(cent_vals.values())
    lo_s, hi_s = node_size_range

    def size_of(n: str) -> float:
        if H.nodes[n].get("kind") == "rail":
            return max(18.0, (lo_s + hi_s) * 0.55)
        if size_by == "single size" or max_c == min_c:
            return (lo_s + hi_s) / 2.0
        return lo_s + (cent_vals[n] - min_c) * (hi_s - lo_s) / (max_c - min_c)

    if no_overlap:
        pos = _remove_overlaps(pos, {n: size_of(n) + 4.0 for n in H})

    cam_cents = {n: v for n, v in cent_vals.items() if H.nodes[n].get("kind") == "camera"}
    labelled = set(sorted(cam_cents, key=cam_cents.get, reverse=True)[:label_top_n])
    labelled |= {n for n in H if H.nodes[n].get("kind") == "rail"}  # always label station hubs

    legend_html = ""
    node_color: Dict[str, str] = {}
    if color_by == "community" and communities:
        node_color = {n: COMMUNITY_PALETTE[communities.get(n, 0) % len(COMMUNITY_PALETTE)] for n in H}
        n_c = len({communities[n] for n in H if n in communities})
        legend_html = (
            '<div style="position:absolute;top:10px;left:10px;z-index:5;padding:7px 10px;'
            'background:rgba(255,255,255,0.93);border:1px solid #ddd;border-radius:5px;font:12px sans-serif;color:#111">'
            f'<b>Color:</b> {n_c} Spatial Communities<br><b>◆ Diamonds:</b> BTS/MRT Stations · <b>● Circles:</b> CCTV Cameras</div>'
        )
    elif color_by == "rain_effect":
        node_color = {n: EFFECT_HEX.get(H.nodes[n].get("effect_label", ""), "#97c2fc") for n in H}
        legend_html = (
            '<div style="position:absolute;top:10px;left:10px;z-index:5;padding:7px 10px;'
            'background:rgba(255,255,255,0.93);border:1px solid #ddd;border-radius:5px;font:12px sans-serif;color:#111">'
            '<b>◆ Gold:</b> Rail Hubs · <span style="color:#0891b2">● Clears</span> · '
            '<span style="color:#e11d48">● Denser</span> · <span style="color:#16a34a">● Unaffected</span></div>'
        )
    elif color_by in MEASURE_COLUMN:
        col = MEASURE_COLUMN[color_by]
        m_map = {r.Node: float(getattr(r, col)) for r in cent_df.itertuples(index=False)}
        shown = list(H)
        vals = np.array([m_map[n] for n in shown])
        r = rankdata(vals)
        frac = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.full(len(r), 0.5)
        node_color = {n: mcolors.to_hex(colormaps["viridis"](f)) for n, f in zip(shown, frac)}
    else:
        node_color = {n: "#97c2fc" for n in H}

    nt = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="black", directed=False, cdn_resources="in_line")
    cent_lookup = cent_df.set_index("Node")
    for n in H:
        row = cent_lookup.loc[n]
        is_rail = H.nodes[n].get("kind") == "rail"
        neighbors_desc = ", ".join(
            f"{nb} ({H.edges[n, nb].get('distance_m', '?')}m)" for nb in list(H.neighbors(n))[:5]
        )
        comm_line = f"Community: #{communities[n]}" if communities and n in communities else ""
        tip = (
            f"{n}\n{H.nodes[n].get('tooltip', '')}\n"
            f"Connected: {neighbors_desc or 'Isolated'}\n"
            f"Nearest Station: {H.nodes[n].get('nearest_station', '—')} ({H.nodes[n].get('nearest_station_m', '?')}m)\n"
            f"Links: {H.degree(n)} | Betweenness: {row['Betweenness']:.3g} {comm_line}"
        )
        node_dict = {
            "id": str(n),
            "label": str(n).replace("Rail: ", "🚉 ") if n in labelled else " ",
            "shape": "diamond" if is_rail else "dot",
            "size": float(size_of(n)),
            "x": float(pos[n][0]),
            "y": float(pos[n][1]),
            "physics": False,
            "color": "#ffd166" if (is_rail and color_by == "rain_effect") else node_color.get(n, "#97c2fc"),
            "title": tip,
            "font": {"color": "#111111", "bold": is_rail},
        }
        nt.nodes.append(node_dict)
        nt.node_ids.append(node_dict["id"])
        nt.node_map[node_dict["id"]] = node_dict

    for u, v, d in H.edges(data=True):
        is_transit = d.get("edge_type") == "transit"
        nt.edges.append({
            "from": str(u),
            "to": str(v),
            "width": 2.2 if is_transit else 1.1,
            "color": {"color": "#f59e0b" if is_transit else "rgba(100,116,139,0.5)"},
            "title": f"{'Station link' if is_transit else 'Road corridor'}: {d.get('distance_m', '?')} m",
        })

    nt.set_options(f"""
    {{
        "nodes": {{"font": {{"size": {int(font_size)}}}, "borderWidth": 1.5}},
        "edges": {{"smooth": false, "hidden": {str(not show_edges).lower()}}},
        "physics": {{"enabled": false}},
        "interaction": {{"hover": true, "navigationButtons": true, "tooltipDelay": 80, "zoomView": true, "dragView": true}}
    }}
    """)
    html = nt.generate_html()
    return html.replace("<body>", "<body>" + legend_html + JPG_BUTTON_HTML, 1)


def plot_degree_distribution(G: nx.Graph, log_log: bool = False) -> plt.Figure:
    """Plot degree histogram or log-log CCDF P(Degree >= k)."""
    degrees = np.array([d for _, d in G.degree()])
    fig, ax = plt.subplots(figsize=(6, 4))
    if log_log and len(degrees[degrees > 0]):
        k, counts = np.unique(degrees[degrees > 0], return_counts=True)
        ccdf = counts[::-1].cumsum()[::-1] / counts.sum()
        ax.loglog(k, ccdf, marker="o", linestyle="none", markersize=5, label="Observed nodes")
        if len(k) > 1 and k[-1] > k[0]:
            k_line = np.logspace(np.log10(k[0]), np.log10(k[-1]), 40)
            for alpha, style in ((2, "--"), (3, ":")):
                ax.loglog(k_line, (k_line / k[0]) ** -(alpha - 1), linestyle=style, color="gray",
                          label=f"α={alpha} (slope −{alpha - 1})")
        ax.set_ylim(ccdf.min() / 2, 1.5)
        ax.set_ylabel("P(Degree ≥ k)")
        ax.set_title("Degree Distribution (Log-Log CCDF)")
        ax.legend(fontsize=8)
    else:
        max_d = int(degrees.max()) if len(degrees) else 1
        ax.hist(degrees, bins=range(max_d + 2), align="left", rwidth=0.8)
        ax.set_ylabel("Frequency")
        ax.set_title("Degree Distribution (Histogram)")
    ax.set_xlabel("Degree k")
    fig.tight_layout()
    return fig
