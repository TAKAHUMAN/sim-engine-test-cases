from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import (
    PDParameters,
    PK_PARAMETER_NAMES,
    PKParameters,
    ResidualError,
    simulate_pd,
    simulate_pk,
    summarize_pk_metrics,
)


PK_TARGET_WEIGHTS = {
    "Cmax": 0.08,
    "AUC": 0.08,
    "Half_life": 0.15,
    "Tmax": 2.0,
}


def load_pk_targets(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_pd_targets(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _theta_to_pk(theta_log: np.ndarray) -> PKParameters:
    raw = np.exp(theta_log)
    cl_day1 = raw[4]
    cl_day15 = cl_day1 + raw[5]
    return PKParameters(
        ka_h=raw[0],
        vc_l=raw[1],
        q_l_h=raw[2],
        vp_l=raw[3],
        cl_day1_l_h=cl_day1,
        cl_day15_l_h=cl_day15,
        kaut_h=raw[6],
        km_mg_l=raw[7],
    )


def _pk_residuals(theta_log: np.ndarray, targets: pd.DataFrame) -> np.ndarray:
    try:
        params = _theta_to_pk(theta_log)
        predictions = summarize_pk_metrics(params, dt_h=0.2)
    except Exception:
        return np.repeat(1e3, len(targets))

    pred_long = predictions.melt(id_vars="occasion", var_name="metric", value_name="prediction")
    merged = targets.merge(pred_long, on=["occasion", "metric"], how="left")
    residuals = []
    for row in merged.itertuples(index=False):
        if not np.isfinite(row.prediction) or row.prediction <= 0:
            residuals.append(1e3)
            continue
        if row.metric == "Tmax":
            residuals.append((row.prediction - row.value) / PK_TARGET_WEIGHTS[row.metric])
        else:
            residuals.append(np.log(row.prediction / row.value) / PK_TARGET_WEIGHTS[row.metric])
    return np.asarray(residuals)


def fit_pk_model(targets: pd.DataFrame) -> tuple[PKParameters, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Calibrate fixed effects to aggregate paper PK summaries."""

    # Start near a known stable aggregate calibration. The sixth theta is the
    # CL increment, which enforces CLday15 > CLday1 for autoinduction.
    initial = np.log([0.24, 720.0, 115.0, 1400.0, 13.0, 137.0, 0.003, 4.5])
    lower = np.log([0.03, 100.0, 0.5, 100.0, 2.0, 1.0, 1e-4, 0.02])
    upper = np.log([2.0, 8000.0, 300.0, 20000.0, 200.0, 300.0, 0.1, 10.0])
    result = least_squares(
        _pk_residuals,
        initial,
        args=(targets,),
        bounds=(lower, upper),
        max_nfev=90,
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
    )

    params = _theta_to_pk(result.x)
    metrics = summarize_pk_metrics(params, dt_h=0.1)
    comparison = targets.merge(metrics.melt(id_vars="occasion", var_name="metric", value_name="predicted"), on=["occasion", "metric"])
    comparison = comparison.rename(columns={"value": "paper_value"})
    comparison["absolute_error"] = comparison["predicted"] - comparison["paper_value"]
    comparison["percent_error"] = 100.0 * comparison["absolute_error"] / comparison["paper_value"]

    diagnostics = {
        "ofv": float(np.sum(result.fun**2)),
        "optimizer_cost": float(result.cost),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "n_aggregate_targets": int(len(targets)),
    }
    parameter_table = pk_parameter_table(params, result)
    return params, parameter_table, comparison, diagnostics


def _wald_intervals(result, params: PKParameters) -> dict[str, tuple[float, float]]:
    estimates = asdict(params)
    intervals: dict[str, tuple[float, float]] = {}
    try:
        jac = result.jac
        cov_log = np.linalg.pinv(jac.T @ jac)
        se_log = np.sqrt(np.maximum(np.diag(cov_log), 0.0))
    except Exception:
        se_log = np.repeat(np.nan, 8)

    transformed = np.exp(result.x)
    lower_raw = transformed * np.exp(-1.96 * se_log)
    upper_raw = transformed * np.exp(1.96 * se_log)
    raw_names = [
        "ka_h",
        "vc_l",
        "q_l_h",
        "vp_l",
        "cl_day1_l_h",
        "cl_increment_l_h",
        "kaut_h",
        "km_mg_l",
    ]
    raw_ci = dict(zip(raw_names, zip(lower_raw, upper_raw)))
    for name in ["ka_h", "vc_l", "q_l_h", "vp_l", "cl_day1_l_h", "kaut_h", "km_mg_l"]:
        intervals[name] = raw_ci[name]
    intervals["cl_day15_l_h"] = (
        raw_ci["cl_day1_l_h"][0] + raw_ci["cl_increment_l_h"][0],
        raw_ci["cl_day1_l_h"][1] + raw_ci["cl_increment_l_h"][1],
    )
    for name, value in estimates.items():
        intervals.setdefault(name, (value, value))
    return intervals


def pk_parameter_table(params: PKParameters, result) -> pd.DataFrame:
    units = {
        "ka_h": "1/h",
        "vc_l": "L",
        "q_l_h": "L/h",
        "vp_l": "L",
        "cl_day1_l_h": "L/h",
        "cl_day15_l_h": "L/h",
        "kaut_h": "1/h",
        "km_mg_l": "mg/L",
        "bioavailability": "fraction",
    }
    intervals = _wald_intervals(result, params)
    rows = []
    for name, estimate in asdict(params).items():
        ci_low, ci_high = intervals.get(name, (np.nan, np.nan))
        rows.append(
            {
                "parameter": name,
                "estimate": estimate,
                "unit": units[name],
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "iiv_cv_percent": 30.0 if name in PK_PARAMETER_NAMES else np.nan,
                "estimation_note": (
                    "IIV fixed at 30% CV because raw individual PK data are unavailable"
                    if name in PK_PARAMETER_NAMES
                    else "Fixed; absolute oral bioavailability is not identifiable from summary oral data"
                ),
            }
        )
    return pd.DataFrame(rows)


def residual_error_table(residual_error: ResidualError) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parameter": "proportional_error",
                "estimate": residual_error.proportional,
                "unit": "fraction",
                "note": "Assumed for pseudo-observation diagnostics; replace when raw data are available",
            },
            {
                "parameter": "additive_error",
                "estimate": residual_error.additive_ng_ml,
                "unit": "ng/mL",
                "note": "Assumed for pseudo-observation diagnostics; replace when raw data are available",
            },
        ]
    )


def _pd_prediction_at_days(pk_profile: pd.DataFrame, params: PDParameters, days: list[int]) -> np.ndarray:
    end_h = max((day - 1) * 24.0 for day in days)
    pd_profile = simulate_pd(pk_profile, params, end_h=end_h, dt_h=1.0)
    return np.asarray(
        [
            np.interp((day - 1) * 24.0, pd_profile["time_h"], pd_profile["percent_change_from_baseline"])
            for day in days
        ]
    )


def calibrate_pd_models(pk_params: PKParameters, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pk_profile = simulate_pk(pk_params, end_h=26.0 * 24.0, dt_h=0.5)
    parameter_rows = []
    prediction_rows = []

    marker_settings = {
        "sVEGFR1": {
            "mode": "inhibition",
            "free": ["emax", "ec50_ng_ml", "kout_h"],
            "fixed": {},
            "x0": [1.0, 500.0, 0.01],
            "lb": [0.01, 1.0, 0.0001],
            "ub": [5.0, 5000.0, 1.0],
        },
        "sVEGFR2": {
            "mode": "inhibition",
            "free": ["emax", "ec50_ng_ml", "kout_h", "escape_max", "escape_k_h"],
            "fixed": {},
            "x0": [1.0, 200.0, 0.02, 0.40, 0.001],
            "lb": [0.01, 1.0, 0.0001, 0.0, 0.0],
            "ub": [5.0, 5000.0, 1.0, 5.0, 0.1],
        },
        "FGF23": {
            "mode": "stimulation",
            "free": ["emax"],
            "fixed": {"ec50_ng_ml": 200.0, "kout_h": 0.02, "baseline": 140.7, "baseline_unit": "pg/mL"},
            "x0": [2.0],
            "lb": [0.0],
            "ub": [5.0],
        },
        "VEGF": {
            "mode": "stimulation",
            "free": ["emax"],
            "fixed": {"ec50_ng_ml": 200.0, "kout_h": 0.02},
            "x0": [0.5],
            "lb": [0.0],
            "ub": [5.0],
        },
    }

    for marker, settings in marker_settings.items():
        marker_targets = targets[targets["marker"] == marker].copy()
        days = marker_targets["day"].astype(int).tolist()
        observed = marker_targets["percent_change_from_baseline"].astype(float).to_numpy()
        free = settings["free"]
        fixed = settings["fixed"]

        def make_params(x: np.ndarray) -> PDParameters:
            values = dict(
                marker=marker,
                mode=settings["mode"],
                emax=0.4,
                ec50_ng_ml=fixed.get("ec50_ng_ml", 200.0),
                kout_h=fixed.get("kout_h", 0.03),
                escape_max=fixed.get("escape_max", 0.0),
                escape_k_h=fixed.get("escape_k_h", 0.0),
                baseline=fixed.get("baseline", 100.0),
                baseline_unit=fixed.get("baseline_unit", "percent"),
            )
            for name, value in zip(free, x):
                values[name] = float(value)
            return PDParameters(**values)

        x0 = np.asarray(settings["x0"], dtype=float)
        lb = np.asarray(settings["lb"], dtype=float)
        ub = np.asarray(settings["ub"], dtype=float)

        def residuals(x: np.ndarray) -> np.ndarray:
            pred = _pd_prediction_at_days(pk_profile, make_params(x), days)
            scale = np.where(marker_targets["p_value"].astype(str).str.contains("0.0512"), 15.0, 6.0)
            return (pred - observed) / scale

        result = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            max_nfev=160,
            diff_step=1e-3,
            x_scale=np.maximum(np.abs(x0), 1.0),
        )
        pd_params = make_params(result.x)
        pd_profile = simulate_pd(pk_profile, pd_params, end_h=26.0 * 24.0, dt_h=1.0)
        for field, value in pd_params.__dict__.items():
            if field in {"marker", "mode", "baseline_unit"}:
                continue
            parameter_rows.append(
                {
                    "marker": marker,
                    "parameter": field,
                    "estimate": value,
                    "unit": _pd_unit(field),
                    "estimation_note": "Aggregate PD calibration; sparse summary data do not support IIV estimation",
                }
            )
        for target in marker_targets.itertuples(index=False):
            time_h = (int(target.day) - 1) * 24.0
            pred = float(np.interp(time_h, pd_profile["time_h"], pd_profile["percent_change_from_baseline"]))
            prediction_rows.append(
                {
                    "marker": marker,
                    "day": int(target.day),
                    "paper_percent_change": float(target.percent_change_from_baseline),
                    "predicted_percent_change": pred,
                    "absolute_error_percent_points": pred - float(target.percent_change_from_baseline),
                }
            )

    return pd.DataFrame(parameter_rows), pd.DataFrame(prediction_rows)


def _pd_unit(parameter: str) -> str:
    return {
        "emax": "fraction",
        "ec50_ng_ml": "ng/mL",
        "kout_h": "1/h",
        "escape_max": "fraction",
        "escape_k_h": "1/h",
        "baseline": "baseline unit",
    }[parameter]
