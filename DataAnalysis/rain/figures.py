"""Static figures for the report and slides, made only from the result tables in DataAnalysis/output/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from DataAnalysis.rain.common import FIG_DIR, OUT_DIR  # noqa: E402

plt.rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "DejaVu Sans"]   # Thai station names
LABEL_COLOR = {"clears": "#2e9e5b", "denser": "#d1453b", "unaffected": "#9aa0a6", "too few wet hours": "none"}
CLASS_NAME = {"total": "All vehicles", "car": "Cars", "motorcycle": "Motorcycles"}
DPI = 160


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def camera_map(out: Path, fig_dir: Path, rail: pd.DataFrame) -> Path:
    cams = pd.read_csv(out / "camera.csv").merge(pd.read_csv(out / "camera_effect.csv"), on="camera_id", how="left")
    cams["label"] = cams["label"].fillna("too few wet hours")
    fig, ax = plt.subplots(figsize=(7, 7))
    if len(rail):
        ax.scatter(rail["longitude"], rail["latitude"], s=6, c="black", marker="s", alpha=0.35, label="BTS/MRT station")
    for label, color in LABEL_COLOR.items():
        d = cams[cams["label"] == label]
        ax.scatter(d["longitude"], d["latitude"], s=70, c=color, edgecolors="#444", linewidths=0.8,
                   label=f"{label} ({len(d)})")
    ax.set_title("What rain does at each camera (vehicles in frame vs the same hour on dry days)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, fig_dir / "camera_map.png")


def _bars(ax, labels, values, lows, highs, color):
    x = range(len(labels))
    err = [[v - lo for v, lo in zip(values, lows)], [hi - v for v, hi in zip(values, highs)]]
    ax.bar(x, values, color=color, yerr=err, capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(x), labels)


def city_effect(out: Path, fig_dir: Path) -> Path:
    t = pd.read_csv(out / "city_effect.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, window in zip(axes, ("all", "rush", "other")):
        d = t[t["window"] == window].set_index("cls").loc[["total", "car", "motorcycle"]]
        _bars(ax, [CLASS_NAME[c] for c in d.index], d["effect_pct"], d["ci_low"], d["ci_high"], "#4a78c2")
        ax.set_title({"all": "All hours", "rush": "Rush hours", "other": "Other hours"}[window])
    axes[0].set_ylabel("% change in vehicles in frame when raining")
    fig.suptitle("Rain vs the same camera and hour on dry days (bars: 95 % CI)")
    return _save(fig, fig_dir / "city_effect.png")


def natural_experiment(out: Path, fig_dir: Path) -> Path:
    d = pd.read_csv(out / "natural_experiment.csv").set_index("cls").loc[["total", "car", "motorcycle"]]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _bars(ax, [CLASS_NAME[c] for c in d.index], d["effect_pct_points"], d["ci_low"], d["ci_high"], "#7a5cb8")
    ax.set_ylabel("percentage points vs dry cameras nearby")
    ax.set_title(f"Same hour: rained-on camera vs dry cameras within 3 km\n"
                 f"({int(d['n_pairs'].iloc[0])} pairs in {int(d['n_hours'].iloc[0])} hours)")
    return _save(fig, fig_dir / "natural_experiment.png")


def rebound(out: Path, fig_dir: Path) -> Path:
    t = pd.read_csv(out / "rebound.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for cls, color in (("motorcycle", "#d1453b"), ("car", "#4a78c2")):
        d = t[t["cls"] == cls]
        ax.plot(d["minute"], d["ratio"], marker="o", color=color, label=CLASS_NAME[cls])
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("minutes from the first dry hour after rain")
    ax.set_ylabel("vehicles in frame / usual level")
    ax.set_title("Before and after the rain stops (1.0 = usual for that camera and hour)")
    ax.legend()
    return _save(fig, fig_dir / "rebound.png")


def mealtime(out: Path, fig_dir: Path) -> Path:
    t = pd.read_csv(out / "mealtime.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for h in t.loc[t["meal"], "hod"]:
        ax.axvspan(h - 0.5, h + 0.5, color="#f3d27a", alpha=0.4, lw=0)
    ax.errorbar(t["hod"], t["effect_pct"], yerr=[t["effect_pct"] - t["ci_low"], t["ci_high"] - t["effect_pct"]],
                fmt="o-", color="#d1453b", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("hour of day (shaded: meal times)")
    ax.set_ylabel("% change in motorcycles in frame")
    ax.set_title("Motorcycles in rain by hour of day")
    return _save(fig, fig_dir / "mealtime.png")


def denser_share(out: Path, fig_dir: Path) -> Path:
    t = pd.read_csv(out / "denser_share.csv").set_index("feature")
    names = {"near_rail": "near BTS/MRT\n(≤ 500 m)", "near_shelter": "near footbridge /\nexpressway (≤ 150 m)",
             "in_flood_area": "in a Traffy\nflood area"}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = range(len(t))
    ax.bar([i - 0.2 for i in x], 100 * t["denser_share"], 0.4, color="#d1453b",
           label=f"denser in rain (n={int(t['n_denser'].iloc[0])})")
    ax.bar([i + 0.2 for i in x], 100 * t["other_share"], 0.4, color="#9aa0a6",
           label=f"other cameras (n={int(t['n_other'].iloc[0])})")
    ax.set_xticks(list(x), [names[f] for f in t.index])
    ax.set_ylabel("% of cameras")
    ax.set_title("What the cameras that get denser in rain have in common")
    ax.legend()
    return _save(fig, fig_dir / "denser_share.png")


def policy_table(out: Path, fig_dir: Path) -> Path:
    t = pd.read_csv(out / "policy_stations.csv").head(10)
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.35 * max(len(t), 1)))
    ax.axis("off")
    if len(t):
        cells = [[r.station, int(r.denser_cameras), f"{r.moto_lost_per_wet_hour:.1f}"] for r in t.itertuples()]
        tab = ax.table(cellText=cells, colLabels=["Station", "Denser cameras ≤ 500 m", "Motorcycles lost / wet hour"],
                       loc="center", cellLoc="left")
        tab.scale(1, 1.3)
    else:
        ax.text(0.5, 0.5, "No station within 500 m of a camera that gets denser in rain", ha="center")
    ax.set_title("Stations to look at first for covered walkways")
    return _save(fig, fig_dir / "policy_stations.png")


def make_all(out_dir: Path = OUT_DIR, fig_dir: Path = FIG_DIR, rail: pd.DataFrame | None = None) -> list[Path]:
    if rail is None:
        from DataAnalysis.rain.common import load_rail
        rail = load_rail()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    return [camera_map(out_dir, fig_dir, rail), city_effect(out_dir, fig_dir), natural_experiment(out_dir, fig_dir),
            rebound(out_dir, fig_dir), mealtime(out_dir, fig_dir), denser_share(out_dir, fig_dir),
            policy_table(out_dir, fig_dir)]


if __name__ == "__main__":
    for p in make_all():
        print(p)
