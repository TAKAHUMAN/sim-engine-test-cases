"""Validation and reporting utilities for the virtual PK trial."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from model.pk_model import BOOTSTRAP_95CI, DOSE_LEVELS_MG_M2, TYPICAL_70KG, PKParameters, as_parameter_dict
from simulation.pk_simulate import (
    DEFAULT_INFUSION_DURATION_H,
    SAMPLE_TIMES_H,
    auc_to_infinity,
    costeff_bsa_m2,
    dose_mg_from_mg_m2,
    predicted_concentrations,
    solve_amounts,
    summarize_auc_by_dose,
)


def concentration_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    """Median, 5th, and 95th concentration percentiles by dose, time, and analyte."""

    rows: list[dict[str, float | str]] = []
    analytes = {
        "temsirolimus": ("conc_TEM_pred", "conc_TEM_obs"),
        "sirolimus": ("conc_SIR_pred", "conc_SIR_obs"),
    }
    grouped = profiles.groupby(["dose_mg_m2", "time_h"], sort=True)
    for (dose_level, time_h), frame in grouped:
        for analyte, (pred_col, obs_col) in analytes.items():
            pred = frame[pred_col].to_numpy(dtype=float)
            obs = frame[obs_col].to_numpy(dtype=float)
            rows.append(
                {
                    "dose_mg_m2": float(dose_level),
                    "analyte": analyte,
                    "time_h": float(time_h),
                    "pred_p5": float(np.percentile(pred, 5)),
                    "pred_median": float(np.percentile(pred, 50)),
                    "pred_p95": float(np.percentile(pred, 95)),
                    "obs_p5": float(np.percentile(obs, 5)),
                    "obs_median": float(np.percentile(obs, 50)),
                    "obs_p95": float(np.percentile(obs, 95)),
                }
            )
    return pd.DataFrame.from_records(rows)


def plot_pcvpc(summary: pd.DataFrame, output_path: str | Path = "pcvpc.png") -> Path:
    """Create a pcVPC-style plot comparing predicted and observed percentiles."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    analytes = ["temsirolimus", "sirolimus"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for axis, analyte in zip(axes, analytes, strict=True):
        analyte_data = summary[summary["analyte"] == analyte]
        for dose_level in sorted(analyte_data["dose_mg_m2"].unique()):
            dose_data = analyte_data[analyte_data["dose_mg_m2"] == dose_level].sort_values("time_h")
            label = f"{dose_level:g} mg/m2"
            x = dose_data["time_h"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                dose_data["pred_p5"].to_numpy(dtype=float),
                dose_data["pred_p95"].to_numpy(dtype=float),
                alpha=0.12,
            )
            axis.plot(x, dose_data["pred_median"], linewidth=2, label=f"pred {label}")
            axis.scatter(x, dose_data["obs_median"], s=18, marker="o", label=f"obs {label}")
        axis.set_title(analyte.capitalize())
        axis.set_xlabel("Time after dose start (h)")
        axis.set_ylabel("Concentration (ng/mL)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def bootstrap_ci_checks(parameters: PKParameters = TYPICAL_70KG) -> pd.DataFrame:
    """Pass/fail checks for published bootstrap 95% confidence intervals."""

    parameter_values = as_parameter_dict(parameters)
    rows = []
    for name, (low, high) in BOOTSTRAP_95CI.items():
        value = parameter_values[name]
        rows.append(
            {
                "parameter": name,
                "typical_value": value,
                "ci_low": low,
                "ci_high": high,
                "pass": bool(low <= value <= high),
            }
        )
    return pd.DataFrame.from_records(rows)


def _median_profile_targets(profiles: pd.DataFrame, source: str) -> pd.DataFrame:
    if source not in {"pred", "obs"}:
        raise ValueError("source must be 'pred' or 'obs'")
    suffix = "pred" if source == "pred" else "obs"
    return (
        profiles[profiles["time_h"] > 0.0]
        .groupby(["dose_mg_m2", "time_h"], as_index=False)
        .agg(
            tem_target=(f"conc_TEM_{suffix}", "median"),
            sir_target=(f"conc_SIR_{suffix}", "median"),
        )
        .sort_values(["dose_mg_m2", "time_h"])
    )


def _simulate_candidate_medians(
    candidate_70kg: PKParameters,
    group_doses: pd.DataFrame,
    sample_times_h: np.ndarray,
    infusion_duration_h: float,
) -> dict[tuple[float, float], tuple[float, float]]:
    from model.pk_model import allometric_parameters

    predictions: dict[tuple[float, float], tuple[float, float]] = {}
    for row in group_doses.itertuples(index=False):
        parameters = allometric_parameters(float(row.BW), typical_70kg=candidate_70kg)
        amounts = solve_amounts(
            parameters,
            float(row.dose_mg),
            sample_times_h,
            infusion_duration_h=infusion_duration_h,
        )
        tem, sir = predicted_concentrations(amounts, parameters)
        for time_h, tem_c, sir_c in zip(sample_times_h, tem, sir, strict=True):
            predictions[(float(row.dose_mg_m2), float(time_h))] = (float(tem_c), float(sir_c))
    return predictions


def recover_typical_parameters(
    profiles: pd.DataFrame,
    aucs: pd.DataFrame,
    *,
    source: str = "pred",
    infusion_duration_h: float = DEFAULT_INFUSION_DURATION_H,
) -> pd.DataFrame:
    """Estimate selected typical parameters by NCA plus median-profile fitting.

    CL_TEM is estimated by individual NCA and normalized to 70 kg. V1, CL_SIR,
    V4, and Fm are then fit to dose-level median profiles with the other Table 3
    structural parameters fixed.
    """

    cl_tem_individual = (aucs["dose_mg"] * 1000.0 / aucs["auc_TEM"]) / (
        (aucs["BW"] / 70.0) ** 0.75
    )
    cl_tem_estimate = float(np.nanmedian(cl_tem_individual))

    targets = _median_profile_targets(profiles, source=source)
    group_doses = (
        aucs.groupby("dose_mg_m2", as_index=False)
        .agg(BW=("BW", "median"), dose_mg=("dose_mg", "median"))
        .sort_values("dose_mg_m2")
    )
    sample_times = np.array(sorted(targets["time_h"].unique()), dtype=float)
    fixed_cl_candidate = replace(TYPICAL_70KG, CL_TEM=cl_tem_estimate)

    def objective(x: np.ndarray) -> np.ndarray:
        candidate = replace(
            fixed_cl_candidate,
            V1=float(x[0]),
            CL_SIR=float(x[1]),
            V4=float(x[2]),
            Fm=float(x[3]),
        )
        predictions = _simulate_candidate_medians(
            candidate,
            group_doses,
            sample_times,
            infusion_duration_h,
        )
        residuals: list[float] = []
        for row in targets.itertuples(index=False):
            tem_pred, sir_pred = predictions[(float(row.dose_mg_m2), float(row.time_h))]
            tem_target = max(float(row.tem_target), 1e-6)
            sir_target = max(float(row.sir_target), 1e-6)
            residuals.append(np.log(max(tem_pred, 1e-6)) - np.log(tem_target))
            residuals.append(np.log(max(sir_pred, 1e-6)) - np.log(sir_target))
        return np.asarray(residuals, dtype=float)

    result = least_squares(
        objective,
        x0=np.array([TYPICAL_70KG.V1, TYPICAL_70KG.CL_SIR, TYPICAL_70KG.V4, TYPICAL_70KG.Fm]),
        bounds=(
            np.array([2.0, 0.5, 5.0, 0.05]),
            np.array([100.0, 40.0, 300.0, 0.95]),
        ),
        max_nfev=250,
        xtol=1e-7,
        ftol=1e-7,
    )
    estimates = {
        "CL_TEM": cl_tem_estimate,
        "V1": float(result.x[0]),
        "CL_SIR": float(result.x[1]),
        "V4": float(result.x[2]),
        "Fm": float(result.x[3]),
    }
    published = as_parameter_dict(TYPICAL_70KG)
    rows = []
    for name, estimate in estimates.items():
        paper = published[name]
        rows.append(
            {
                "parameter": name,
                "estimated": estimate,
                "paper": paper,
                "deviation_pct": 100.0 * (estimate - paper) / paper,
                "method": "NCA" if name == "CL_TEM" else "median-profile fit",
            }
        )
    return pd.DataFrame.from_records(rows)


def validation_outputs(
    profiles: pd.DataFrame,
    aucs: pd.DataFrame,
    *,
    output_dir: str | Path = ".",
) -> dict[str, pd.DataFrame | Path]:
    """Compute all validation outputs requested by the reproduction prompt."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = concentration_summary(profiles)
    auc_summary = summarize_auc_by_dose(aucs)
    recovered = recover_typical_parameters(profiles, aucs)
    ci_checks = bootstrap_ci_checks()
    pcvpc_path = plot_pcvpc(summary, output_path / "pcvpc.png")
    return {
        "summary": summary,
        "auc_summary": auc_summary,
        "recovered": recovered,
        "ci_checks": ci_checks,
        "pcvpc_path": pcvpc_path,
    }


def paper_auc_ratio_target() -> dict[float, str]:
    """Paper-reported comparator for the sirolimus:temsirolimus AUC ratio."""

    return {8.0: "not reported", 10.0: "not reported", 15.0: "1.21 +/- 1.18"}
