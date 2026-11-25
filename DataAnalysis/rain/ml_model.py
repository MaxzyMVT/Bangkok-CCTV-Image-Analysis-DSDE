"""Supervised ML prediction & feature-correlation pipeline backed by PyDuckLake.

Builds hourly camera+weather+spatial feature datasets from PyDuckLake Silver/Gold tables,
trains regression (vehicle counts) and classification (traffic surge > +25% baseline) models
across both chronological (Aug 25-29 -> Aug 30-31) and spatial (GroupKFold by camera_id)
splits, computes empirical correlation tables, and stores results in PyDuckLake gold.*.

Usage:
    python DataAnalysis/rain/ml_model.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataAnalysis.rain.common import OUT_DIR, RUSH  # noqa: E402
from DataEngineering.lakehouse_pipeline import lake_has_table, query_lake, write_lake_table  # noqa: E402


WEATHER_CONTEXT_FEATURES = [
    "hod",
    "weekend",
    "is_rush",
    "rain_mm",
    "wet",
    "heavy",
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "visibility_km",
    "wind_kt",
    "latitude",
    "longitude",
    "rail_dist_m",
    "gauge_dist_m",
    "near_rail",
    "near_shelter",
    "in_flood_area",
    "max_flood_cm",
    "fm91_rain",
    "fm91_congestion",
]

WITH_BASELINE_FEATURES = WEATHER_CONTEXT_FEATURES + ["total_base", "car_base", "motorcycle_base"]


def build_feature_table(lakehouse_dir: Path | None = None, min_avg_bytes: int = 3000) -> pd.DataFrame:
    """Join Gold rain_camera_hour + rain_camera with Silver weather, METAR, flood, FM91, and frame quality."""
    ch = query_lake("SELECT * FROM lake.gold.rain_camera_hour", lakehouse_dir=lakehouse_dir)
    cam = query_lake("SELECT * FROM lake.gold.rain_camera", lakehouse_dir=lakehouse_dir)
    wx = query_lake(
        "SELECT CAST(datetime AS TIMESTAMP) AS hour_ts, precipitation_mean, precipitation_max, "
        "pct_points_raining, temperature_2m, relative_humidity_2m, cloud_cover "
        "FROM lake.silver.weather_hourly_bangkok",
        lakehouse_dir=lakehouse_dir,
    )
    metar = query_lake(
        "SELECT date_trunc('hour', CAST(datetime AS TIMESTAMP)) AS hour_ts, "
        "AVG(visibility_km) AS visibility_km, AVG(wind_kt) AS wind_kt "
        "FROM lake.silver.metar_observations GROUP BY 1",
        lakehouse_dir=lakehouse_dir,
    )
    flood = query_lake(
        "SELECT CAST(datetime AS TIMESTAMP) AS hour_ts, MAX(flood_cm) AS max_flood_cm "
        "FROM lake.silver.drainage_flood_hourly GROUP BY 1",
        lakehouse_dir=lakehouse_dir,
    )
    fm91 = query_lake(
        "SELECT date_trunc('hour', CAST(datetime AS TIMESTAMP)) AS hour_ts, "
        "SUM(COALESCE(mentions_rain, 0)) AS fm91_rain, "
        "SUM(COALESCE(mentions_congestion, 0)) AS fm91_congestion "
        "FROM lake.silver.fm91_news GROUP BY 1",
        lakehouse_dir=lakehouse_dir,
    )
    frame_bytes = query_lake(
        "SELECT TRY_CAST(camera_id AS INTEGER) AS camera_id, "
        "date_trunc('hour', CAST(captured_at AS TIMESTAMP)) AS hour_ts, "
        "AVG(file_size_bytes) AS avg_bytes "
        "FROM lake.silver.cctv_frames GROUP BY 1, 2",
        lakehouse_dir=lakehouse_dir,
    )

    for frame in (ch, wx, metar, flood, fm91, frame_bytes):
        frame["hour_ts"] = pd.to_datetime(frame["hour_ts"])
        if frame["hour_ts"].dt.tz is not None:
            frame["hour_ts"] = frame["hour_ts"].dt.tz_localize(None)

    df = (
        ch.merge(cam, on="camera_id", how="left")
        .merge(wx, on="hour_ts", how="left")
        .merge(metar, on="hour_ts", how="left")
        .merge(flood, on="hour_ts", how="left")
        .merge(fm91, on="hour_ts", how="left")
        .merge(frame_bytes, on=["camera_id", "hour_ts"], how="left")
    )
    df["avg_bytes"] = df["avg_bytes"].fillna(20000.0)
    df = df[df["avg_bytes"] >= min_avg_bytes].copy()

    df["is_rush"] = df["hod"].isin(RUSH).astype(int)
    for col in ("weekend", "wet", "heavy", "near_rail", "near_shelter", "in_flood_area"):
        df[col] = df[col].astype(bool).astype(int)
    for col in ("max_flood_cm", "fm91_rain", "fm91_congestion"):
        df[col] = df[col].fillna(0.0)
    for col in ("temperature_2m", "relative_humidity_2m", "cloud_cover", "visibility_km", "wind_kt"):
        df[col] = df[col].fillna(df[col].median())

    df["pct_vs_baseline"] = 100.0 * (df["total"] / df["total_base"].replace(0, np.nan) - 1.0)
    df["is_surge"] = (df["total"] > 1.25 * df["total_base"]).astype(int)
    return df.reset_index(drop=True)


def compute_correlations(df: pd.DataFrame, lakehouse_dir: Path | None = None) -> pd.DataFrame:
    """Compute Pearson correlations for traffic counts and rain response against predictors."""
    rows = []
    hourly_predictors = [
        ("rain_mm", "Hourly gauge rain (mm)"),
        ("temperature_2m", "Temperature (C)"),
        ("relative_humidity_2m", "Relative humidity (%)"),
        ("cloud_cover", "Cloud cover (%)"),
        ("visibility_km", "METAR visibility (km)"),
        ("rail_dist_m", "Distance to BTS/MRT (m)"),
        ("gauge_dist_m", "Distance to rain gauge (m)"),
        ("hod", "Hour of day"),
        ("weekend", "Weekend indicator"),
        ("is_rush", "Rush-hour indicator"),
        ("max_flood_cm", "Citywide max flood sensor (cm)"),
        ("fm91_congestion", "FM91 congestion mentions/hr"),
    ]
    for target in ("total", "car", "motorcycle", "pct_vs_baseline"):
        for col, label in hourly_predictors:
            sub = df[[target, col]].dropna()
            r = float(sub[target].corr(sub[col])) if len(sub) > 2 else float("nan")
            rows.append({
                "scope": "hourly_camera",
                "target": target,
                "feature": col,
                "feature_label": label,
                "pearson_r": round(r, 4) if not math.isnan(r) else None,
                "n_obs": int(len(sub)),
            })

    if lake_has_table("gold", "rain_camera", lakehouse_dir) and lake_has_table("gold", "rain_camera_effect", lakehouse_dir):
        cam = query_lake("SELECT * FROM lake.gold.rain_camera", lakehouse_dir=lakehouse_dir)
        eff = query_lake("SELECT * FROM lake.gold.rain_camera_effect", lakehouse_dir=lakehouse_dir)
        ce = cam.merge(eff, on="camera_id", how="inner")
        for col, label in (
            ("rail_dist_m", "Distance to BTS/MRT (m)"),
            ("gauge_dist_m", "Distance to rain gauge (m)"),
            ("near_rail", "Near rail (<=500m)"),
            ("near_shelter", "Near footbridge/expressway (<=150m)"),
            ("in_flood_area", "In Traffy flood cluster (<=100m)"),
            ("n_wet_hours", "Number of wet hours"),
        ):
            s = pd.to_numeric(ce[col].astype(float), errors="coerce")
            sub = pd.DataFrame({"effect_pct": ce["effect_pct"], "x": s}).dropna()
            r = float(sub["effect_pct"].corr(sub["x"])) if len(sub) > 2 else float("nan")
            rows.append({
                "scope": "camera_effect",
                "target": "effect_pct",
                "feature": col,
                "feature_label": label,
                "pearson_r": round(r, 4) if not math.isnan(r) else None,
                "n_obs": int(len(sub)),
            })
    return pd.DataFrame(rows)


def _eval_regressor(name: str, builder, X_tr, y_tr, X_te, y_te, target: str, split: str) -> tuple[dict, np.ndarray, object]:
    model = builder()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "task": "regression",
        "target": target,
        "model": name,
        "split": split,
        "mae": round(float(mean_absolute_error(y_te, pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_te, pred)), 4),
        "r2": round(float(r2_score(y_te, pred)), 4),
        "accuracy": None,
        "f1": None,
        "roc_auc": None,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }
    return metrics, pred, model


def run_ml_pipeline(lakehouse_dir: Path | None = None, write_lake: bool = True) -> dict[str, pd.DataFrame]:
    """Train regression & classification models and persist Gold ML + correlation tables."""
    df = build_feature_table(lakehouse_dir=lakehouse_dir)
    corr_df = compute_correlations(df, lakehouse_dir=lakehouse_dir)

    train_mask = df["hour_ts"] < pd.Timestamp("2026-08-30 00:00:00")
    test_mask = ~train_mask
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    metric_rows = []
    imp_rows = []
    pred_frames = []

    reg_specs = {
        "total (no baseline)": ("total", WEATHER_CONTEXT_FEATURES),
        "motorcycle (no baseline)": ("motorcycle", WEATHER_CONTEXT_FEATURES),
        "car (no baseline)": ("car", WEATHER_CONTEXT_FEATURES),
        "total (with camera baseline)": ("total", WITH_BASELINE_FEATURES),
        "motorcycle (with camera baseline)": ("motorcycle", WITH_BASELINE_FEATURES),
    }

    for label, (target, cols) in reg_specs.items():
        X_tr, y_tr = train_df[cols], train_df[target]
        X_te, y_te = test_df[cols], test_df[target]

        builders = [
            ("HistGradientBoosting", lambda: HistGradientBoostingRegressor(max_depth=6, max_iter=180, learning_rate=0.07, random_state=42)),
            ("RandomForest", lambda: RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
            ("Ridge", lambda: Ridge(alpha=1.0)),
        ]
        for mname, fn in builders:
            met, pred, fitted = _eval_regressor(mname, fn, X_tr, y_tr, X_te, y_te, label, "temporal_aug30_31")
            metric_rows.append(met)
            if mname == "HistGradientBoosting" and label in ("total (no baseline)", "total (with camera baseline)", "motorcycle (with camera baseline)"):
                perm = permutation_importance(fitted, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
                for f, im, istd in zip(cols, perm.importances_mean, perm.importances_std):
                    imp_rows.append({
                        "target": label,
                        "feature": f,
                        "importance_mean": round(float(im), 5),
                        "importance_std": round(float(istd), 5),
                    })
                if label == "total (with camera baseline)":
                    pf = test_df[["camera_id", "hour_ts", "hod", "weekend", "rain_mm", "total_base"]].copy()
                    pf["target"] = target
                    pf["actual"] = np.round(y_te.to_numpy(), 3)
                    pf["predicted"] = np.round(pred, 3)
                    pf["residual"] = np.round(pf["actual"] - pf["predicted"], 3)
                    pred_frames.append(pf)

        # Spatial 5-fold GroupKFold across cameras for primary HistGradientBoosting model
        gkf = GroupKFold(n_splits=5)
        oof_pred = np.zeros(len(df))
        for tr_i, va_i in gkf.split(df[cols], df[target], groups=df["camera_id"]):
            m_cv = HistGradientBoostingRegressor(max_depth=6, max_iter=150, learning_rate=0.07, random_state=42)
            m_cv.fit(df[cols].iloc[tr_i], df[target].iloc[tr_i])
            oof_pred[va_i] = m_cv.predict(df[cols].iloc[va_i])
        metric_rows.append({
            "task": "regression",
            "target": label,
            "model": "HistGradientBoosting",
            "split": "spatial_5fold_unseen_cameras",
            "mae": round(float(mean_absolute_error(df[target], oof_pred)), 4),
            "rmse": round(float(root_mean_squared_error(df[target], oof_pred)), 4),
            "r2": round(float(r2_score(df[target], oof_pred)), 4),
            "accuracy": None,
            "f1": None,
            "roc_auc": None,
            "n_train": int(len(df) * 0.8),
            "n_test": int(len(df)),
        })

    # Classification: congestion surge (> +25% vs matched baseline)
    clf = HistGradientBoostingClassifier(max_depth=6, max_iter=150, learning_rate=0.07, class_weight="balanced", random_state=42)
    clf.fit(train_df[WITH_BASELINE_FEATURES], train_df["is_surge"])
    prob = clf.predict_proba(test_df[WITH_BASELINE_FEATURES])[:, 1]
    pred_cls = (prob >= 0.5).astype(int)
    metric_rows.append({
        "task": "classification",
        "target": "is_surge (> +25% vs baseline)",
        "model": "HistGradientBoostingClassifier",
        "split": "temporal_aug30_31",
        "mae": None,
        "rmse": None,
        "r2": None,
        "accuracy": round(float(accuracy_score(test_df["is_surge"], pred_cls)), 4),
        "f1": round(float(f1_score(test_df["is_surge"], pred_cls, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(test_df["is_surge"], prob)), 4),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    })

    outputs = {
        "ml_metrics": pd.DataFrame(metric_rows),
        "ml_feature_importance": pd.DataFrame(imp_rows).sort_values(["target", "importance_mean"], ascending=[True, False]).reset_index(drop=True),
        "ml_predictions": pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(),
        "rain_correlations": corr_df,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        if frame.empty:
            continue
        if write_lake:
            write_lake_table("gold", name, frame, f"gold: update {name} from supervised ML pipeline", lakehouse_dir=lakehouse_dir)
        csv_name = name.removeprefix("rain_") if name.startswith("rain_") else name
        frame.to_csv(OUT_DIR / f"{csv_name}.csv", index=False, lineterminator="\n")
    return outputs


def fit_scenario_predictor(lakehouse_dir: Path | None = None):
    """Fit lightweight context regressors on all clean hours for Streamlit what-if scenario simulation."""
    df = build_feature_table(lakehouse_dir=lakehouse_dir)
    models = {}
    for target in ("total", "car", "motorcycle"):
        m = HistGradientBoostingRegressor(max_depth=6, max_iter=150, learning_rate=0.08, random_state=42)
        m.fit(df[WITH_BASELINE_FEATURES], df[target])
        models[target] = m
    return df, models


if __name__ == "__main__":
    res = run_ml_pipeline()
    print("=== ML Metrics ===")
    print(res["ml_metrics"].to_string(index=False))
    print("\n=== Top Feature Importances ===")
    print(res["ml_feature_importance"].groupby("target").head(6).to_string(index=False))
