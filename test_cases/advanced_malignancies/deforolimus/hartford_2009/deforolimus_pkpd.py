#!/usr/bin/env python3
"""
PK/PD model for deforolimus (AP23573) from Hartford et al. 2009.

This script uses only the published summary values captured for this test case:
summary NCA PK data, median 4E-BP1 phosphorylation inhibition values, and the
reported AUC/tumor-size correlation summary. The exploratory TGI block also
reports the explicit kg and kd assumptions because no individual tumor data
exist.

The PK model is a 2-compartment IV infusion model. For speed, PK is solved
analytically with matrix exponentials instead of repeatedly calling an ODE
solver. The PD effect compartment is solved with an exact step update under
piecewise-linear PK forcing.

No concentration-time points or individual tumor data are invented.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Keep matplotlib from trying to write to ~/.matplotlib on restricted systems.
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Hartford et al. 2009 summary data
# ---------------------------------------------------------------------------

pk_data = pd.DataFrame(
    {
        "dose_mg": [6.25, 12.5, 25.0, 50.0, 75.0, 100.0],
        "n": [1, 3, 4, 15, 16, 4],
        "Cmax_ngmL": [329, 394, 570, 982, 1195, 1255],
        "AUC_ugh_mL": [3.8, 5.1, 9.0, 10.7, 12.7, 13.4],
        "thalf_h": [52.2, 47.0, 46.2, 44.9, 47.3, 46.0],
        "CL_Lh": [1.7, 2.7, 2.9, 4.9, 6.4, 7.7],
        "Vss_L": [80.3, 111.0, 136.0, 235.0, 323.0, 373.0],
    }
)
pk_data["Cmax_ugmL"] = pk_data["Cmax_ngmL"] / 1000.0

pd_data = pd.DataFrame(
    {
        "time_h": [0, 1, 24, 48, 168],
        "inhibition_pct": [0, 95, 90, 90, 75],
    }
)
pd_data["inhibition_frac"] = pd_data["inhibition_pct"] / 100.0
pd_data["time_from_start_h"] = [0.0, 1.5, 24.5, 48.5, 168.0]

tumor_auc_correlation = {
    "r": -0.43,
    "p_value": 0.015,
    "n": 32,
    "assessment_time_weeks": 8,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DOSE_REF = 50.0
INFUSION_DURATION_H = 0.5
T_END_H = 168.0
WEEKLY_INTERVAL_H = 168.0
N_WEEKLY_DOSES = 8
MULTIDOSE_T_END_H = WEEKLY_INTERVAL_H * N_WEEKLY_DOSES
PK_OBJECTIVE_WEIGHTS = {"Cmax": 1.0, "AUC": 1.0, "thalf": 2.0}
TGI_KG_H_INV = 0.005
FINAL_PD_KE0_H_INV = 0.05243587
FINAL_PD_IC50_UGML = 0.00470150
FINAL_PD_IMAX = 1.0
VIRTUAL_TGI_N = tumor_auc_correlation["n"]
VIRTUAL_TGI_REPS = 1000
VIRTUAL_TGI_SEED = 20260510
VIRTUAL_TGI_KG_SIGMA = 0.5
VIRTUAL_TGI_SENSITIVITY_SIGMA = 0.4


def power_func(dose: np.ndarray | float, param_ref: float, exponent: float) -> np.ndarray | float:
    """Power relationship at the 50 mg reference dose."""
    return param_ref * (np.asarray(dose) / DOSE_REF) ** exponent


def make_time_grid(t_end: float = T_END_H, extra_times: list[float] | None = None) -> np.ndarray:
    """Compact but smooth grid for plotting and metric extraction."""
    parts = [
        np.linspace(0.0, INFUSION_DURATION_H, 61),
        np.linspace(INFUSION_DURATION_H, 6.0, 121),
        np.linspace(6.0, t_end, 360),
    ]
    if extra_times:
        parts.append(np.asarray(extra_times, dtype=float))
    t = np.unique(np.round(np.concatenate(parts), 10))
    return t[(t >= 0.0) & (t <= t_end)]


def weekly_dose_times(n_doses: int = N_WEEKLY_DOSES, interval_h: float = WEEKLY_INTERVAL_H) -> np.ndarray:
    """Weekly dose start times for the 8-week exploratory simulation."""
    return np.arange(n_doses, dtype=float) * interval_h


def make_multidose_time_grid(
    t_end: float = MULTIDOSE_T_END_H,
    dose_times: np.ndarray | None = None,
    interval_h: float = WEEKLY_INTERVAL_H,
) -> np.ndarray:
    """Grid with dense points around each infusion plus the full 8-week span."""
    if dose_times is None:
        dose_times = weekly_dose_times()
    local = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, INFUSION_DURATION_H, 31),
                np.linspace(INFUSION_DURATION_H, 24.0, 80),
                np.linspace(24.0, interval_h, 96),
            ]
        )
    )
    parts = [np.linspace(0.0, t_end, 1000)]
    for dose_time in dose_times:
        shifted = dose_time + local
        parts.append(shifted[shifted <= t_end])
    t = np.unique(np.round(np.concatenate(parts), 10))
    return t[(t >= 0.0) & (t <= t_end)]


def matrix_exp_for_times(k_matrix: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Vectorized matrix exponential for one 2x2 matrix across many times."""
    eigvals, eigvecs = np.linalg.eig(k_matrix)
    inv_eigvecs = np.linalg.inv(eigvecs)
    exp_vals = np.exp(np.outer(times, eigvals))
    exp_mats = np.einsum("ij,tj,jk->tik", eigvecs, exp_vals, inv_eigvecs)
    exp_mats = np.real_if_close(exp_mats, tol=1000)
    if np.iscomplexobj(exp_mats):
        exp_mats = exp_mats.real
    return exp_mats


def dose_dependent_pk_params(
    dose: float,
    CL_ref: float,
    alpha_CL: float,
    Vss_ref: float,
    alpha_Vss: float,
    Q: float,
    frac_V1: float,
) -> dict[str, float]:
    """Dose-dependent CL and Vss plus compartment volumes."""
    CL = float(power_func(dose, CL_ref, alpha_CL))
    Vss = float(power_func(dose, Vss_ref, alpha_Vss))
    V1 = frac_V1 * Vss
    V2 = (1.0 - frac_V1) * Vss
    return {"CL": CL, "Vss": Vss, "Q": Q, "V1": V1, "V2": V2}


# ---------------------------------------------------------------------------
# Fast PK model
# ---------------------------------------------------------------------------

def simulate_pk(
    dose: float,
    CL_ref: float,
    alpha_CL: float,
    Vss_ref: float,
    alpha_Vss: float,
    Q: float,
    frac_V1: float,
    t_end: float = T_END_H,
    times: np.ndarray | None = None,
    return_params: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Simulate the 2-compartment IV infusion model.

    Amounts are mg, volumes are L, and concentration is ug/mL.
    Since 1 mg/L equals 1 ug/mL, A1/V1 is already ug/mL.
    """
    if not (0.0 < frac_V1 < 1.0):
        raise ValueError("frac_V1 must be between 0 and 1.")
    if Q <= 0.0:
        raise ValueError("Q must be positive.")

    if times is None:
        times = make_time_grid(t_end)
    times = np.asarray(times, dtype=float)

    params = dose_dependent_pk_params(dose, CL_ref, alpha_CL, Vss_ref, alpha_Vss, Q, frac_V1)
    CL, V1, V2 = params["CL"], params["V1"], params["V2"]

    k_matrix = np.array(
        [
            [-(CL + Q) / V1, Q / V2],
            [Q / V1, -Q / V2],
        ],
        dtype=float,
    )
    rate = np.array([dose / INFUSION_DURATION_H, 0.0], dtype=float)
    kinv_rate = np.linalg.solve(k_matrix, rate)
    identity = np.eye(2)

    amounts = np.zeros((times.size, 2), dtype=float)
    during = times <= INFUSION_DURATION_H

    if np.any(during):
        exp_pre = matrix_exp_for_times(k_matrix, times[during])
        amounts[during] = np.einsum("tij,j->ti", exp_pre - identity, kinv_rate)

    exp_end = matrix_exp_for_times(k_matrix, np.array([INFUSION_DURATION_H]))[0]
    amount_end = (exp_end - identity) @ kinv_rate

    after = ~during
    if np.any(after):
        exp_post = matrix_exp_for_times(k_matrix, times[after] - INFUSION_DURATION_H)
        amounts[after] = np.einsum("tij,j->ti", exp_post, amount_end)

    C1 = amounts[:, 0] / V1
    C1 = np.where(C1 < 0.0, np.maximum(C1, 0.0), C1)

    if return_params:
        return times, C1, params
    return times, C1


def terminal_half_life(CL: float, Q: float, V1: float, V2: float) -> float:
    """Terminal half-life from the slow eigenvalue of the 2-compartment system."""
    k_matrix = np.array(
        [
            [-(CL + Q) / V1, Q / V2],
            [Q / V1, -Q / V2],
        ],
        dtype=float,
    )
    eigvals = np.linalg.eigvals(k_matrix).real
    terminal_lambda = -np.max(eigvals)
    return float(np.log(2.0) / terminal_lambda)


def terminal_half_life_from_profile(t: np.ndarray, C1: np.ndarray, terminal_fraction: float = 0.20) -> float:
    """Terminal half-life from log-linear terminal slope of the simulated profile."""
    t = np.asarray(t, dtype=float)
    C1 = np.asarray(C1, dtype=float)
    n_terminal = max(10, int(np.ceil(t.size * terminal_fraction)))
    t_terminal = t[-n_terminal:]
    C_terminal = C1[-n_terminal:]
    valid = C_terminal > 0.0
    if np.sum(valid) < 3:
        return np.nan

    slope, _ = np.polyfit(t_terminal[valid], np.log(C_terminal[valid]), 1)
    lambda_z = -float(slope)
    if lambda_z <= 0.0:
        return np.nan
    return float(np.log(2.0) / lambda_z)


def get_pk_metrics(
    t: np.ndarray,
    C1: np.ndarray,
    dose: float,
    CL: float,
    Q: float,
    V1: float,
    V2: float,
) -> dict[str, float]:
    """Extract Cmax, exact AUC0-inf, and terminal-slope half-life."""
    thalf_terminal = terminal_half_life_from_profile(t, C1)
    if not np.isfinite(thalf_terminal):
        thalf_terminal = terminal_half_life(CL, Q, V1, V2)
    return {
        "Cmax_ugmL": float(np.max(C1)),
        "AUC_ugh_mL": float(dose / CL),
        "thalf_h": float(thalf_terminal),
        "thalf_eigen_h": terminal_half_life(CL, Q, V1, V2),
    }


def pk_objective(
    transformed_params: np.ndarray,
    data: pd.DataFrame,
    CL_ref: float,
    alpha_CL: float,
    Vss_ref: float,
    alpha_Vss: float,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Fit Q and frac_V1 to observed Cmax, AUC, and terminal half-life.

    AUC is exact for a linear IV model: AUC0-inf = dose / CL. Therefore, the
    AUC part depends on the dose-clearance power fit, not Q or frac_V1.
    """
    if weights is None:
        weights = PK_OBJECTIVE_WEIGHTS
    log_Q, logit_frac = transformed_params

    if not (np.log(1e-4) <= log_Q <= np.log(50.0)):
        return 1e12

    Q = float(np.exp(log_Q))
    frac_V1 = float(1.0 / (1.0 + np.exp(-logit_frac)))
    if Q <= 0.0 or Q > 50.0 or frac_V1 <= 0.05 or frac_V1 >= 0.95:
        return 1e12

    total_error = 0.0
    metric_times = make_time_grid(extra_times=[INFUSION_DURATION_H])

    for _, row in data.iterrows():
        dose = float(row["dose_mg"])
        try:
            t, C1, params = simulate_pk(
                dose,
                CL_ref,
                alpha_CL,
                Vss_ref,
                alpha_Vss,
                Q,
                frac_V1,
                times=metric_times,
                return_params=True,
            )
            pred = get_pk_metrics(t, C1, dose, params["CL"], Q, params["V1"], params["V2"])
        except Exception:
            return 1e12

        err_Cmax = weights["Cmax"] * ((row["Cmax_ugmL"] - pred["Cmax_ugmL"]) / row["Cmax_ugmL"]) ** 2
        err_AUC = weights["AUC"] * ((row["AUC_ugh_mL"] - pred["AUC_ugh_mL"]) / row["AUC_ugh_mL"]) ** 2
        err_thalf = weights["thalf"] * ((row["thalf_h"] - pred["thalf_h"]) / row["thalf_h"]) ** 2
        total_error += float(err_Cmax + err_AUC + err_thalf)

    return total_error


def fit_pk_model(data: pd.DataFrame, weights: dict[str, float] | None = None) -> dict[str, object]:
    """Fit dose-power CL/Vss and optimize Q/frac_V1."""
    if weights is None:
        weights = PK_OBJECTIVE_WEIGHTS
    doses = data["dose_mg"].values.astype(float)

    (CL_ref, alpha_CL), pcov_CL = curve_fit(
        power_func,
        doses,
        data["CL_Lh"].values.astype(float),
        p0=[4.9, 0.7],
        maxfev=20000,
    )
    (Vss_ref, alpha_Vss), pcov_Vss = curve_fit(
        power_func,
        doses,
        data["Vss_L"].values.astype(float),
        p0=[235.0, 0.6],
        maxfev=20000,
    )

    starts = [
        [np.log(0.05), np.log(0.10 / 0.90)],
        [np.log(0.10), np.log(0.20 / 0.80)],
        [np.log(0.25), np.log(0.30 / 0.70)],
        [np.log(2.0), np.log(0.30 / 0.70)],
        [np.log(1.0), np.log(0.25 / 0.75)],
        [np.log(3.0), np.log(0.40 / 0.60)],
        [np.log(0.5), np.log(0.20 / 0.80)],
        [np.log(5.0), np.log(0.35 / 0.65)],
    ]

    best_result = None
    best_obj = np.inf
    history = []
    for i, x0 in enumerate(starts, start=1):
        result = minimize(
            pk_objective,
            np.asarray(x0, dtype=float),
            args=(data, CL_ref, alpha_CL, Vss_ref, alpha_Vss, weights),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-9},
        )
        Q_start = float(np.exp(x0[0]))
        frac_start = float(1.0 / (1.0 + np.exp(-x0[1])))
        history.append(
            {
                "start": i,
                "Q_start_Lh": Q_start,
                "frac_V1_start": frac_start,
                "weight_Cmax": weights["Cmax"],
                "weight_AUC": weights["AUC"],
                "weight_thalf": weights["thalf"],
                "objective": float(result.fun),
                "success": bool(result.success),
            }
        )
        if result.fun < best_obj:
            best_obj = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("PK optimization failed from all starting points.")

    Q_opt = float(np.exp(best_result.x[0]))
    frac_V1_opt = float(1.0 / (1.0 + np.exp(-best_result.x[1])))

    return {
        "CL_ref": float(CL_ref),
        "alpha_CL": float(alpha_CL),
        "pcov_CL": pcov_CL,
        "Vss_ref": float(Vss_ref),
        "alpha_Vss": float(alpha_Vss),
        "pcov_Vss": pcov_Vss,
        "Q": Q_opt,
        "frac_V1": frac_V1_opt,
        "weights": dict(weights),
        "objective": best_obj,
        "optimizer_result": best_result,
        "history": pd.DataFrame(history),
    }


def build_pk_gof_table(data: pd.DataFrame, fit: dict[str, object]) -> tuple[pd.DataFrame, dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Generate PK predictions and goodness-of-fit rows."""
    rows = []
    profiles = {}
    plot_times = make_time_grid(extra_times=pd_data["time_from_start_h"].tolist())

    for _, row in data.iterrows():
        dose = float(row["dose_mg"])
        t, C1, params = simulate_pk(
            dose,
            fit["CL_ref"],
            fit["alpha_CL"],
            fit["Vss_ref"],
            fit["alpha_Vss"],
            fit["Q"],
            fit["frac_V1"],
            times=plot_times,
            return_params=True,
        )
        metrics = get_pk_metrics(t, C1, dose, params["CL"], fit["Q"], params["V1"], params["V2"])
        profiles[dose] = (t, C1)
        rows.append(
            {
                "dose_mg": dose,
                "n": int(row["n"]),
                "Cmax_obs_ugmL": float(row["Cmax_ugmL"]),
                "Cmax_pred_ugmL": metrics["Cmax_ugmL"],
                "Cmax_pct_error": 100.0 * (metrics["Cmax_ugmL"] - row["Cmax_ugmL"]) / row["Cmax_ugmL"],
                "AUC_obs_ugh_mL": float(row["AUC_ugh_mL"]),
                "AUC_pred_ugh_mL": metrics["AUC_ugh_mL"],
                "AUC_pct_error": 100.0 * (metrics["AUC_ugh_mL"] - row["AUC_ugh_mL"]) / row["AUC_ugh_mL"],
                "thalf_obs_h": float(row["thalf_h"]),
                "thalf_pred_h": metrics["thalf_h"],
                "thalf_eigen_h": metrics["thalf_eigen_h"],
                "thalf_pct_error": 100.0 * (metrics["thalf_h"] - row["thalf_h"]) / row["thalf_h"],
                "CL_pred_Lh": params["CL"],
                "Vss_pred_L": params["Vss"],
                "V1_pred_L": params["V1"],
                "V2_pred_L": params["V2"],
            }
        )

    return pd.DataFrame(rows), profiles


# ---------------------------------------------------------------------------
# Fast PD model
# ---------------------------------------------------------------------------

def effect_compartment_piecewise_linear(t: np.ndarray, C1: np.ndarray, ke0: float) -> np.ndarray:
    """Exact effect-compartment update when C1 is linear between grid points."""
    if ke0 <= 0.0:
        raise ValueError("ke0 must be positive.")
    t = np.asarray(t, dtype=float)
    C1 = np.asarray(C1, dtype=float)

    Ce = np.zeros_like(C1)
    for i in range(1, t.size):
        h = t[i] - t[i - 1]
        if h <= 0.0:
            Ce[i] = Ce[i - 1]
            continue
        c0 = C1[i - 1]
        slope = (C1[i] - C1[i - 1]) / h
        decay = np.exp(-ke0 * h)
        Ce[i] = (
            Ce[i - 1] * decay
            + c0 * (1.0 - decay)
            + slope * (h - (1.0 - decay) / ke0)
        )

    return np.maximum(Ce, 0.0)


def simulate_pkpd(
    dose: float,
    fit: dict[str, object],
    ke0: float,
    IC50: float,
    Imax: float = 1.0,
    t_end: float = T_END_H,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate central PK, effect compartment, and inhibition."""
    if times is None:
        times = make_time_grid(t_end, extra_times=pd_data["time_from_start_h"].tolist())
    t, C1 = simulate_pk(
        dose,
        fit["CL_ref"],
        fit["alpha_CL"],
        fit["Vss_ref"],
        fit["alpha_Vss"],
        fit["Q"],
        fit["frac_V1"],
        t_end=t_end,
        times=times,
    )
    Ce = effect_compartment_piecewise_linear(t, C1, ke0)
    inhibition = Imax * Ce / (IC50 + Ce)
    inhibition = np.clip(inhibition, 0.0, 1.0)
    return t, C1, Ce, inhibition


def constrained_imax_from_logit(logit_imax: float, lower: float = 0.90, upper: float = 1.00) -> float:
    """Map an unconstrained value to Imax in [0.90, 1.00]."""
    sigmoid = 1.0 / (1.0 + np.exp(-logit_imax))
    return float(lower + (upper - lower) * sigmoid)


def logit_for_constrained_imax(imax: float, lower: float = 0.90, upper: float = 1.00) -> float:
    """Starting-value helper for constrained Imax."""
    scaled = (imax - lower) / (upper - lower)
    scaled = float(np.clip(scaled, 1e-6, 1.0 - 1e-6))
    return float(np.log(scaled / (1.0 - scaled)))


def pd_objective(
    transformed_params: np.ndarray,
    data: pd.DataFrame,
    dose_for_fitting: float,
    pk_fit: dict[str, object],
    free_imax: bool = True,
) -> float:
    """Fit ke0 and IC50 to the supplied inhibition data."""
    log_ke0, log_IC50 = transformed_params[:2]
    ke0 = float(np.exp(log_ke0))
    IC50 = float(np.exp(log_IC50))
    Imax = constrained_imax_from_logit(float(transformed_params[2])) if free_imax else 1.0

    if ke0 <= 0.0 or ke0 > 10.0 or IC50 <= 0.0 or IC50 > 10.0 or Imax < 0.90 or Imax > 1.0:
        return 1e12

    try:
        t, _, _, inhibition = simulate_pkpd(dose_for_fitting, pk_fit, ke0, IC50, Imax=Imax)
    except Exception:
        return 1e12

    pred = np.interp(data["time_from_start_h"].values, t, inhibition)
    pred = np.clip(pred, 0.0, 1.0)

    weights = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    residuals = weights * (data["inhibition_frac"].values - pred) ** 2
    return float(np.sum(residuals))


def fit_pd_model(
    data: pd.DataFrame,
    pk_fit: dict[str, object],
    dose_for_fitting: float = 75.0,
    free_imax: bool = True,
) -> dict[str, object]:
    """Optimize ke0 and IC50, with optional Imax constrained to [0.90, 1.00]."""
    starts = [
        [np.log(0.02), np.log(0.01), logit_for_constrained_imax(0.98)],
        [np.log(0.05), np.log(0.05), logit_for_constrained_imax(0.95)],
        [np.log(0.01), np.log(0.001), logit_for_constrained_imax(0.99)],
        [np.log(0.1), np.log(0.1), logit_for_constrained_imax(0.93)],
        [np.log(0.03), np.log(0.005), logit_for_constrained_imax(0.97)],
        [np.log(0.008), np.log(0.02), logit_for_constrained_imax(1.00 - 1e-6)],
    ]
    if not free_imax:
        starts = [start[:2] for start in starts]
    bounds = [(np.log(1e-5), np.log(10.0)), (np.log(1e-8), np.log(10.0))]
    if free_imax:
        bounds.append((-30.0, 30.0))

    best_result = None
    best_obj = np.inf
    history = []
    for i, x0 in enumerate(starts, start=1):
        result = minimize(
            pd_objective,
            np.asarray(x0, dtype=float),
            args=(data, dose_for_fitting, pk_fit, free_imax),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 4000, "ftol": 1e-14, "gtol": 1e-10},
        )
        history.append(
            {
                "start": i,
                "ke0_start_h_inv": float(np.exp(x0[0])),
                "IC50_start_ugmL": float(np.exp(x0[1])),
                "Imax_start": constrained_imax_from_logit(float(x0[2])) if free_imax else 1.0,
                "objective": float(result.fun),
                "success": bool(result.success),
            }
        )
        if result.fun < best_obj:
            best_obj = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("PD optimization failed from all starting points.")

    ke0 = float(np.exp(best_result.x[0]))
    IC50 = float(np.exp(best_result.x[1]))
    Imax = constrained_imax_from_logit(float(best_result.x[2])) if free_imax else 1.0

    return {
        "dose_for_fitting_mg": dose_for_fitting,
        "ke0": ke0,
        "ke0_half_life_h": float(np.log(2.0) / ke0),
        "IC50": IC50,
        "Imax": Imax,
        "Imax_free": bool(free_imax),
        "Imax_constraint": "[0.90, 1.00]" if free_imax else "fixed at 1.00",
        "objective": best_obj,
        "optimizer_result": best_result,
        "history": pd.DataFrame(history),
    }


def final_fixed_imax_pd_model(dose_for_fitting: float = 75.0) -> dict[str, object]:
    """Final biologically interpretable PD parameter set selected for simulations."""
    return {
        "dose_for_fitting_mg": dose_for_fitting,
        "ke0": FINAL_PD_KE0_H_INV,
        "ke0_half_life_h": float(np.log(2.0) / FINAL_PD_KE0_H_INV),
        "IC50": FINAL_PD_IC50_UGML,
        "Imax": FINAL_PD_IMAX,
        "Imax_free": False,
        "Imax_constraint": "fixed at 1.00",
        "objective": np.nan,
        "history": pd.DataFrame(
            [
                {
                    "note": "Final fixed-Imax PD set selected from the interpretable Iteration 1 fit.",
                    "ke0_h_inv": FINAL_PD_KE0_H_INV,
                    "IC50_ugmL": FINAL_PD_IC50_UGML,
                    "Imax": FINAL_PD_IMAX,
                }
            ]
        ),
    }


def build_pd_gof_table(data: pd.DataFrame, pk_fit: dict[str, object], pd_fit: dict[str, object]) -> pd.DataFrame:
    """Predicted inhibition at the observed PD sampling times."""
    t, C1, Ce, inhibition = simulate_pkpd(
        pd_fit["dose_for_fitting_mg"],
        pk_fit,
        pd_fit["ke0"],
        pd_fit["IC50"],
        pd_fit["Imax"],
    )
    pred = np.interp(data["time_from_start_h"].values, t, inhibition) * 100.0
    C1_obs = np.interp(data["time_from_start_h"].values, t, C1)
    Ce_obs = np.interp(data["time_from_start_h"].values, t, Ce)

    out = data.copy()
    out["C1_at_time_ugmL"] = C1_obs
    out["Ce_at_time_ugmL"] = Ce_obs
    out["inhibition_pred_pct"] = pred
    out["residual_pct_points"] = out["inhibition_pct"] - out["inhibition_pred_pct"]
    return out


# ---------------------------------------------------------------------------
# Multidose PK/PD, sensitivity, and exploratory TGI
# ---------------------------------------------------------------------------

def simulate_multidose_pk(
    dose: float,
    pk_fit: dict[str, object],
    dose_times: np.ndarray | None = None,
    t_end: float = MULTIDOSE_T_END_H,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate repeated weekly dosing by linear superposition.

    For a fixed dose, CL/Vss/Q/V1/V2 are fixed, so superposition is equivalent
    to restarting the same linear ODE at each dose time with the previous
    residual amounts carried forward.
    """
    if dose_times is None:
        dose_times = weekly_dose_times()
    if times is None:
        times = make_multidose_time_grid(t_end=t_end, dose_times=dose_times)
    t = np.asarray(times, dtype=float)
    C_total = np.zeros_like(t)

    for dose_time in dose_times:
        active = t >= dose_time
        if not np.any(active):
            continue
        rel_t = t[active] - dose_time
        _, C_rel = simulate_pk(
            dose,
            pk_fit["CL_ref"],
            pk_fit["alpha_CL"],
            pk_fit["Vss_ref"],
            pk_fit["alpha_Vss"],
            pk_fit["Q"],
            pk_fit["frac_V1"],
            t_end=float(np.max(rel_t)),
            times=rel_t,
        )
        C_total[active] += C_rel

    return t, C_total


def simulate_multidose_pkpd(
    dose: float,
    pk_fit: dict[str, object],
    pd_fit: dict[str, object],
    dose_times: np.ndarray | None = None,
    t_end: float = MULTIDOSE_T_END_H,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repeated-dose PK plus effect compartment and inhibition."""
    t, C1 = simulate_multidose_pk(dose, pk_fit, dose_times=dose_times, t_end=t_end, times=times)
    Ce = effect_compartment_piecewise_linear(t, C1, pd_fit["ke0"])
    inhibition = pd_fit["Imax"] * Ce / (pd_fit["IC50"] + Ce)
    inhibition = np.clip(inhibition, 0.0, 1.0)
    return t, C1, Ce, inhibition


def build_multidose_pkpd_table(pk_fit: dict[str, object], pd_fit: dict[str, object], dose: float = 75.0) -> pd.DataFrame:
    """Full 8-week PK/PD profile for the selected dose."""
    dose_times = weekly_dose_times()
    t, C1, Ce, inhibition = simulate_multidose_pkpd(dose, pk_fit, pd_fit, dose_times=dose_times)
    return pd.DataFrame(
        {
            "time_h": t,
            "dose_mg": dose,
            "C1_ugmL": C1,
            "Ce_ugmL": Ce,
            "inhibition_pct": 100.0 * inhibition,
            "is_weekly_dose_time": np.isin(np.round(t, 10), np.round(dose_times, 10)),
        }
    )


def build_ke0_sensitivity_profiles(
    pk_fit: dict[str, object],
    pd_fit: dict[str, object],
    dose: float = 75.0,
) -> pd.DataFrame:
    """Single-dose inhibition profiles at ke0/2, fitted ke0, and ke0*2."""
    rows = []
    multipliers = [(0.5, "ke0 half"), (1.0, "ke0 fitted"), (2.0, "ke0 double")]
    for multiplier, label in multipliers:
        ke0 = pd_fit["ke0"] * multiplier
        t, C1, Ce, inhibition = simulate_pkpd(
            dose,
            pk_fit,
            ke0,
            pd_fit["IC50"],
            pd_fit["Imax"],
        )
        rows.append(
            pd.DataFrame(
                {
                    "scenario": label,
                    "ke0_multiplier": multiplier,
                    "ke0_h_inv": ke0,
                    "time_h": t,
                    "C1_ugmL": C1,
                    "Ce_ugmL": Ce,
                    "inhibition_pct": 100.0 * inhibition,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def solve_tgi_profile(t: np.ndarray, C1: np.ndarray, kg: float, kd: float) -> np.ndarray:
    """Solve dW/dt = kg*W - kd*C1(t)*W with W(0)=1 by cumulative integration."""
    t = np.asarray(t, dtype=float)
    C1 = np.asarray(C1, dtype=float)
    integrand = kg - kd * C1
    dt = np.diff(t)
    area = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dt)])
    W = np.exp(area)
    return W


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r with a finite fallback for constant vectors."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[finite]
    y = np.asarray(y, dtype=float)[finite]
    if x.size < 3:
        return np.nan, np.nan
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def scan_tgi_kd(pk_fit: dict[str, object], pk_gof: pd.DataFrame, kg: float = TGI_KG_H_INV) -> tuple[float, pd.DataFrame]:
    """
    Scan kd values and choose the one closest to the reported negative r.

    This is not a fitted tumor model. It only asks whether the simple
    concentration-driven ODE can reproduce the reported direction.
    """
    target_r = tumor_auc_correlation["r"]
    kd_grid = np.unique(np.concatenate([[0.0], np.logspace(-4, 0, 220)]))
    dose_times = weekly_dose_times()
    scan_rows = []

    precomputed = {}
    for dose in pk_gof["dose_mg"].values.astype(float):
        t, C1 = simulate_multidose_pk(dose, pk_fit, dose_times=dose_times)
        precomputed[dose] = (t, C1, float(np.trapezoid(C1, t)))

    single_auc = pk_gof["AUC_pred_ugh_mL"].values.astype(float)
    for kd in kd_grid:
        pct_changes = []
        auc_8w = []
        for dose in pk_gof["dose_mg"].values.astype(float):
            t, C1, auc_multi = precomputed[dose]
            W = solve_tgi_profile(t, C1, kg, kd)
            pct_changes.append(100.0 * (W[-1] - 1.0))
            auc_8w.append(auc_multi)
        pct_changes_array = np.asarray(pct_changes)
        r_single, p_single = safe_pearsonr(single_auc, np.asarray(pct_changes))
        r_multi, p_multi = safe_pearsonr(np.asarray(auc_8w), np.asarray(pct_changes))
        eligible_non_saturated = (
            np.isfinite(r_single)
            and r_single < 0.0
            and pct_changes_array.min() < 0.0
            and pct_changes_array.max() > 0.0
            and pct_changes_array.min() > -95.0
            and pct_changes_array.max() < 10000.0
        )
        scan_rows.append(
            {
                "kg_h_inv": kg,
                "kd_h_inv_per_ugmL": kd,
                "r_vs_single_dose_AUC": r_single,
                "p_vs_single_dose_AUC": p_single,
                "r_vs_8week_AUC": r_multi,
                "p_vs_8week_AUC": p_multi,
                "min_tumor_change_pct": float(pct_changes_array.min()),
                "max_tumor_change_pct": float(pct_changes_array.max()),
                "eligible_non_saturated_directional": bool(eligible_non_saturated),
                "abs_error_vs_reported_r": abs(r_single - target_r) if np.isfinite(r_single) else np.inf,
            }
        )

    scan_table = pd.DataFrame(scan_rows)
    eligible = scan_table[scan_table["eligible_non_saturated_directional"]]
    if eligible.empty:
        best_row = scan_table.loc[scan_table["abs_error_vs_reported_r"].idxmin()]
    else:
        best_row = eligible.loc[eligible["abs_error_vs_reported_r"].idxmin()]
    return float(best_row["kd_h_inv_per_ugmL"]), scan_table


def build_tgi_exploratory_table(
    pk_gof: pd.DataFrame,
    pk_fit: dict[str, object],
    kg: float = TGI_KG_H_INV,
    kd_override: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use the reported correlation and the requested exploratory tumor ODE.

    Because there are no individual tumor-size values, kg/kd are not
    identifiable. The returned tumor percent changes are assumption-based
    simulations, not observed or fitted clinical response estimates.
    """
    if kd_override is None:
        best_kd, scan_table = scan_tgi_kd(pk_fit, pk_gof, kg=kg)
    else:
        best_kd = float(kd_override)
        _, scan_table = scan_tgi_kd(pk_fit, pk_gof, kg=kg)
    r = tumor_auc_correlation["r"]
    auc = pk_gof["AUC_pred_ugh_mL"].values.astype(float)
    auc_sd = float(np.std(auc, ddof=0))
    if auc_sd == 0.0:
        auc_z = np.zeros_like(auc)
    else:
        auc_z = (auc - float(np.mean(auc))) / auc_sd

    rows = []
    dose_times = weekly_dose_times()
    for _, pk_row in pk_gof.iterrows():
        dose = float(pk_row["dose_mg"])
        t, C1 = simulate_multidose_pk(dose, pk_fit, dose_times=dose_times)
        W = solve_tgi_profile(t, C1, kg, best_kd)
        rows.append(
            {
                "dose_mg": dose,
                "single_dose_AUC_pred_ugh_mL": float(pk_row["AUC_pred_ugh_mL"]),
                "eight_week_AUC_0_1344_ugh_mL": float(np.trapezoid(C1, t)),
                "W_8weeks_normalized": float(W[-1]),
                "simulated_tumor_change_pct": float(100.0 * (W[-1] - 1.0)),
                "kg_h_inv_assumed": kg,
                "kd_h_inv_per_ugmL_scanned": best_kd,
            }
        )

    out = pd.DataFrame(rows)
    out["AUC_z"] = auc_z
    out["expected_tumor_change_z_from_reported_r"] = r * auc_z
    r_single, p_single = safe_pearsonr(out["single_dose_AUC_pred_ugh_mL"].values, out["simulated_tumor_change_pct"].values)
    r_multi, p_multi = safe_pearsonr(out["eight_week_AUC_0_1344_ugh_mL"].values, out["simulated_tumor_change_pct"].values)
    out["simulated_r_vs_single_dose_AUC"] = r_single
    out["simulated_p_vs_single_dose_AUC"] = p_single
    out["simulated_r_vs_8week_AUC"] = r_multi
    out["simulated_p_vs_8week_AUC"] = p_multi
    out["reported_r"] = r
    out["reported_p_value"] = tumor_auc_correlation["p_value"]
    out["reported_n"] = tumor_auc_correlation["n"]
    out["assessment_time_weeks"] = tumor_auc_correlation["assessment_time_weeks"]
    out["note"] = (
        "Exploratory ODE simulation only; no individual tumor data, no fitted kg/kd, "
        "and no observed percent tumor change available."
    )
    return out, scan_table


def rowwise_pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast Pearson r for rows of x/y with nan for constant rows."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_masked = np.where(finite, x, np.nan)
    y_masked = np.where(finite, y, np.nan)
    x_center = x_masked - np.nanmean(x_masked, axis=1, keepdims=True)
    y_center = y_masked - np.nanmean(y_masked, axis=1, keepdims=True)
    numerator = np.nansum(x_center * y_center, axis=1)
    denominator = np.sqrt(np.nansum(x_center**2, axis=1) * np.nansum(y_center**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = numerator / denominator
    r[denominator == 0.0] = np.nan
    return r


def precompute_tgi_dose_auc(pk_gof: pd.DataFrame, pk_fit: dict[str, object]) -> pd.DataFrame:
    """Single-dose and 8-week AUC by dose for deterministic and stochastic TGI."""
    rows = []
    for _, row in pk_gof.iterrows():
        dose = float(row["dose_mg"])
        t, C1 = simulate_multidose_pk(dose, pk_fit)
        rows.append(
            {
                "dose_mg": dose,
                "single_dose_AUC_pred_ugh_mL": float(row["AUC_pred_ugh_mL"]),
                "eight_week_AUC_0_1344_ugh_mL": float(np.trapezoid(C1, t)),
            }
        )
    return pd.DataFrame(rows)


def build_stochastic_tgi_simulation(
    pk_gof: pd.DataFrame,
    pk_fit: dict[str, object],
    kg_geomean: float = TGI_KG_H_INV,
    kg_sigma: float = VIRTUAL_TGI_KG_SIGMA,
    sensitivity_sigma: float = VIRTUAL_TGI_SENSITIVITY_SIGMA,
    n_patients: int = VIRTUAL_TGI_N,
    n_reps: int = VIRTUAL_TGI_REPS,
    seed: int = VIRTUAL_TGI_SEED,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add between-patient variability to the exploratory TGI simulation.

    Dose assignment uses the source cohort n values as weights, then
    samples N=32 virtual patients to match the reported evaluable sample size.
    """
    auc_lookup = precompute_tgi_dose_auc(pk_gof, pk_fit)
    doses = auc_lookup["dose_mg"].values.astype(float)
    dose_n = pk_data.set_index("dose_mg").loc[doses, "n"].values.astype(float)
    dose_prob = dose_n / np.sum(dose_n)
    single_auc = auc_lookup["single_dose_AUC_pred_ugh_mL"].values.astype(float)
    auc_8w = auc_lookup["eight_week_AUC_0_1344_ugh_mL"].values.astype(float)

    rng = np.random.default_rng(seed)
    assigned_idx = rng.choice(np.arange(doses.size), size=(n_reps, n_patients), p=dose_prob)
    assigned_single_auc = single_auc[assigned_idx]
    assigned_auc_8w = auc_8w[assigned_idx]
    assigned_dose = doses[assigned_idx]

    kg_draws = rng.lognormal(mean=np.log(kg_geomean), sigma=kg_sigma, size=(n_reps, n_patients))
    sensitivity_draws = rng.lognormal(mean=0.0, sigma=sensitivity_sigma, size=(n_reps, n_patients))

    kd_grid = np.logspace(-2, 1, 90)
    summary_rows = []
    replicate_store: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for kd_base in kd_grid:
        exponent = kg_draws * MULTIDOSE_T_END_H - kd_base * sensitivity_draws * assigned_auc_8w
        exponent = np.clip(exponent, -50.0, 50.0)
        tumor_change_pct = 100.0 * (np.exp(exponent) - 1.0)
        r_values = rowwise_pearsonr(assigned_single_auc, tumor_change_pct)
        saturation_fraction = np.mean(tumor_change_pct <= -99.9, axis=1)
        finite_r = r_values[np.isfinite(r_values)]
        if finite_r.size == 0:
            median_r = np.nan
            low_r = np.nan
            high_r = np.nan
        else:
            median_r = float(np.median(finite_r))
            low_r, high_r = np.percentile(finite_r, [2.5, 97.5])
        median_saturation = float(np.median(saturation_fraction))
        eligible = np.isfinite(median_r) and -0.50 <= median_r <= -0.30
        summary_rows.append(
            {
                "kd_base_h_inv_per_ugmL": float(kd_base),
                "median_r": median_r,
                "r_2p5": float(low_r) if np.isfinite(low_r) else np.nan,
                "r_97p5": float(high_r) if np.isfinite(high_r) else np.nan,
                "median_saturation_fraction_pct_le_minus99p9": median_saturation,
                "eligible_target_range": bool(eligible),
                "abs_error_vs_reported_r": abs(median_r - tumor_auc_correlation["r"]) if np.isfinite(median_r) else np.inf,
            }
        )
        replicate_store[float(kd_base)] = (r_values, tumor_change_pct)

    summary = pd.DataFrame(summary_rows)
    eligible_summary = summary[summary["eligible_target_range"]]
    if eligible_summary.empty:
        selected = summary.loc[summary["abs_error_vs_reported_r"].idxmin()]
    else:
        selected = eligible_summary.loc[eligible_summary["abs_error_vs_reported_r"].idxmin()]

    selected_kd = float(selected["kd_base_h_inv_per_ugmL"])
    selected_r, selected_tumor_change = replicate_store[selected_kd]
    replicate_results = pd.DataFrame(
        {
            "replicate": np.arange(1, n_reps + 1),
            "kd_base_h_inv_per_ugmL": selected_kd,
            "r_auc_tumor_change": selected_r,
            "mean_tumor_change_pct": np.mean(selected_tumor_change, axis=1),
            "median_tumor_change_pct": np.median(selected_tumor_change, axis=1),
            "saturation_fraction_pct_le_minus99p9": np.mean(selected_tumor_change <= -99.9, axis=1),
        }
    )
    median_r = float(np.nanmedian(selected_r))
    representative_rep_idx = int(np.nanargmin(np.abs(selected_r - median_r)))
    example_patients = pd.DataFrame(
        {
            "replicate": representative_rep_idx + 1,
            "patient_id": np.arange(1, n_patients + 1),
            "dose_mg": assigned_dose[representative_rep_idx],
            "single_dose_AUC_pred_ugh_mL": assigned_single_auc[representative_rep_idx],
            "eight_week_AUC_0_1344_ugh_mL": assigned_auc_8w[representative_rep_idx],
            "kg_h_inv": kg_draws[representative_rep_idx],
            "sensitivity_factor": sensitivity_draws[representative_rep_idx],
            "effective_kd_h_inv_per_ugmL": selected_kd * sensitivity_draws[representative_rep_idx],
            "simulated_tumor_change_pct": selected_tumor_change[representative_rep_idx],
            "replicate_r_auc_tumor_change": selected_r[representative_rep_idx],
        }
    )

    summary["selected_for_final"] = summary["kd_base_h_inv_per_ugmL"] == selected_kd
    summary["n_virtual_patients"] = n_patients
    summary["n_replicates"] = n_reps
    summary["kg_geomean_h_inv"] = kg_geomean
    summary["kg_lognormal_sigma"] = kg_sigma
    summary["sensitivity_lognormal_sigma"] = sensitivity_sigma
    summary["seed"] = seed
    summary["dose_assignment_note"] = "Dose cohorts sampled using source PK n values as weights."

    return selected_kd, summary, replicate_results, example_patients


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_pk_nonlinear_plot(fit: dict[str, object]) -> Path:
    doses = pk_data["dose_mg"].values
    dose_range = np.linspace(5.0, 110.0, 250)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(doses, pk_data["CL_Lh"], color="red", s=80, zorder=5, label="Observed")
    axes[0].plot(
        dose_range,
        power_func(dose_range, fit["CL_ref"], fit["alpha_CL"]),
        color="steelblue",
        linewidth=2,
        label=f"Fit: {fit['CL_ref']:.2f}*(D/50)^{fit['alpha_CL']:.2f}",
    )
    axes[0].set_xlabel("Dose (mg)")
    axes[0].set_ylabel("CL (L/h)")
    axes[0].set_title("Clearance vs Dose")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(doses, pk_data["Vss_L"], color="red", s=80, zorder=5, label="Observed")
    axes[1].plot(
        dose_range,
        power_func(dose_range, fit["Vss_ref"], fit["alpha_Vss"]),
        color="steelblue",
        linewidth=2,
        label=f"Fit: {fit['Vss_ref']:.1f}*(D/50)^{fit['alpha_Vss']:.2f}",
    )
    axes[1].set_xlabel("Dose (mg)")
    axes[1].set_ylabel("Vss (L)")
    axes[1].set_title("Volume of Distribution vs Dose")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIGURE_DIR / "PK_nonlinear_relationships.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pk_profiles_plot(profiles: dict[float, tuple[np.ndarray, np.ndarray]]) -> Path:
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(profiles)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for color, (dose, (t, C1)) in zip(colors, profiles.items()):
        axes[0].plot(t, C1, color=color, linewidth=2, label=f"{dose:g} mg")
        mask = C1 > 0.0
        axes[1].semilogy(t[mask], C1[mask], color=color, linewidth=2, label=f"{dose:g} mg")

    for ax in axes:
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (ug/mL)")
        ax.set_xlim(0.0, T_END_H)
        ax.axvline(INFUSION_DURATION_H, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Dose", fontsize=9)

    axes[0].set_title("Deforolimus PK, linear scale")
    axes[1].set_title("Deforolimus PK, semi-log scale")

    fig.tight_layout()
    path = FIGURE_DIR / "PK_concentration_time_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pk_gof_plot(pk_gof: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        pk_gof["Cmax_obs_ugmL"],
        pk_gof["Cmax_pred_ugmL"],
        s=pk_gof["n"] * 10,
        color="steelblue",
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )
    for _, row in pk_gof.iterrows():
        axes[0].annotate(f"{row['dose_mg']:g} mg", (row["Cmax_obs_ugmL"], row["Cmax_pred_ugmL"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    lim0 = [0.0, 1.1 * max(pk_gof["Cmax_obs_ugmL"].max(), pk_gof["Cmax_pred_ugmL"].max())]
    axes[0].plot(lim0, lim0, "r--", linewidth=1.5)
    axes[0].set_xlabel("Observed Cmax (ug/mL)")
    axes[0].set_ylabel("Predicted Cmax (ug/mL)")
    axes[0].set_title("Cmax: predicted vs observed")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        pk_gof["AUC_obs_ugh_mL"],
        pk_gof["AUC_pred_ugh_mL"],
        s=pk_gof["n"] * 10,
        color="darkorange",
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )
    for _, row in pk_gof.iterrows():
        axes[1].annotate(f"{row['dose_mg']:g} mg", (row["AUC_obs_ugh_mL"], row["AUC_pred_ugh_mL"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    lim1 = [0.0, 1.1 * max(pk_gof["AUC_obs_ugh_mL"].max(), pk_gof["AUC_pred_ugh_mL"].max())]
    axes[1].plot(lim1, lim1, "r--", linewidth=1.5)
    axes[1].set_xlabel("Observed AUC (ug*h/mL)")
    axes[1].set_ylabel("Predicted AUC (ug*h/mL)")
    axes[1].set_title("AUC: predicted vs observed")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIGURE_DIR / "PK_goodness_of_fit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pd_profile_plot(pk_fit: dict[str, object], pd_fit: dict[str, object], pd_gof: pd.DataFrame) -> Path:
    t, C1, Ce, inhibition = simulate_pkpd(
        pd_fit["dose_for_fitting_mg"],
        pk_fit,
        pd_fit["ke0"],
        pd_fit["IC50"],
        pd_fit["Imax"],
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    axes[0].plot(t, C1, color="steelblue", linewidth=2, label="C1 central")
    axes[0].plot(t, Ce, color="crimson", linestyle="--", linewidth=2, label="Ce effect")
    axes[0].set_ylabel("Concentration (ug/mL)")
    axes[0].set_title(f"PK/PD profile at {pd_fit['dose_for_fitting_mg']:g} mg")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, 100.0 * inhibition, color="seagreen", linewidth=2, label="Predicted inhibition")
    axes[1].scatter(
        pd_gof["time_from_start_h"],
        pd_gof["inhibition_pct"],
        color="black",
        s=60,
        zorder=5,
        label="Observed median",
    )
    axes[1].set_ylabel("4E-BP1 inhibition (%)")
    axes[1].set_ylim(-5.0, 105.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(t[C1 > 0], C1[C1 > 0], color="steelblue", linewidth=2, label="C1 central")
    axes[2].semilogy(t[Ce > 0], Ce[Ce > 0], color="crimson", linestyle="--", linewidth=2, label="Ce effect")
    axes[2].axhline(pd_fit["IC50"], color="gray", linestyle=":", linewidth=2, label="IC50")
    axes[2].set_xlabel("Time from infusion start (h)")
    axes[2].set_ylabel("Concentration (ug/mL)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.axvline(INFUSION_DURATION_H, color="gray", linestyle="--", alpha=0.35)
        ax.set_xlim(0.0, T_END_H)

    fig.tight_layout()
    path = FIGURE_DIR / "PD_pkpd_profile_75mg.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pd_gof_plot(pd_gof: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pd_gof["time_from_start_h"], pd_gof["inhibition_pred_pct"], color="seagreen", marker="o", linewidth=2, label="Predicted")
    ax.scatter(pd_gof["time_from_start_h"], pd_gof["inhibition_pct"], color="black", s=70, zorder=5, label="Observed")
    ax.set_xlabel("Time from infusion start (h)")
    ax.set_ylabel("4E-BP1 inhibition (%)")
    ax.set_ylim(-5.0, 105.0)
    ax.set_title("PD goodness of fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "PD_goodness_of_fit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pd_dose_comparison_plot(pk_fit: dict[str, object], pd_fit: dict[str, object]) -> Path:
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pk_data)))
    fig, ax = plt.subplots(figsize=(11, 6))
    for color, dose in zip(colors, pk_data["dose_mg"]):
        t, _, _, inhibition = simulate_pkpd(float(dose), pk_fit, pd_fit["ke0"], pd_fit["IC50"], pd_fit["Imax"])
        ax.plot(t, 100.0 * inhibition, color=color, linewidth=2, label=f"{dose:g} mg")
    ax.scatter(pd_data["time_from_start_h"], pd_data["inhibition_pct"], color="black", s=55, zorder=5, label="Observed PD data")
    ax.set_xlabel("Time from infusion start (h)")
    ax.set_ylabel("4E-BP1 inhibition (%)")
    ax.set_ylim(-5.0, 105.0)
    ax.set_xlim(0.0, T_END_H)
    ax.set_title("PD simulation across doses")
    ax.legend(title="Dose", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "PD_dose_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_multidose_pkpd_plot(multidose_table: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    t = multidose_table["time_h"].values
    C1 = multidose_table["C1_ugmL"].values
    Ce = multidose_table["Ce_ugmL"].values
    inhibition = multidose_table["inhibition_pct"].values
    dose_times = weekly_dose_times()

    axes[0].plot(t, C1, color="steelblue", linewidth=1.8, label="C1 central")
    axes[0].plot(t, Ce, color="crimson", linestyle="--", linewidth=1.8, label="Ce effect")
    axes[0].set_ylabel("Concentration (ug/mL)")
    axes[0].set_title("8 weekly doses at 75 mg: PK/PD accumulation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, inhibition, color="seagreen", linewidth=1.8)
    axes[1].set_ylabel("4E-BP1 inhibition (%)")
    axes[1].set_ylim(-5.0, 105.0)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(t[C1 > 0], C1[C1 > 0], color="steelblue", linewidth=1.8, label="C1 central")
    axes[2].semilogy(t[Ce > 0], Ce[Ce > 0], color="crimson", linestyle="--", linewidth=1.8, label="Ce effect")
    axes[2].set_xlabel("Time from first infusion start (h)")
    axes[2].set_ylabel("Concentration (ug/mL)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        for dose_time in dose_times:
            ax.axvline(dose_time, color="gray", linestyle=":", alpha=0.35)
        ax.set_xlim(0.0, MULTIDOSE_T_END_H)

    fig.tight_layout()
    path = FIGURE_DIR / "PKPD_multidose_75mg_8weeks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_ke0_sensitivity_plot(sensitivity_table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"ke0 half": "darkorange", "ke0 fitted": "seagreen", "ke0 double": "steelblue"}
    for scenario, sub in sensitivity_table.groupby("scenario", sort=False):
        ax.plot(
            sub["time_h"],
            sub["inhibition_pct"],
            linewidth=2,
            color=colors.get(scenario, None),
            label=scenario,
        )
    ax.scatter(pd_data["time_from_start_h"], pd_data["inhibition_pct"], color="black", s=55, zorder=5, label="Observed PD data")
    ax.set_xlabel("Time from infusion start (h)")
    ax.set_ylabel("4E-BP1 inhibition (%)")
    ax.set_ylim(-5.0, 105.0)
    ax.set_xlim(0.0, T_END_H)
    ax.set_title("ke0 sensitivity analysis at 75 mg")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "PD_ke0_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_tgi_plot(tgi_table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tgi_table["single_dose_AUC_pred_ugh_mL"],
        tgi_table["simulated_tumor_change_pct"],
        color="purple",
        marker="o",
        linewidth=2,
    )
    for _, row in tgi_table.iterrows():
        ax.annotate(
            f"{row['dose_mg']:g} mg",
            (row["single_dose_AUC_pred_ugh_mL"], row["simulated_tumor_change_pct"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Single-dose model-predicted AUC (ug*h/mL)")
    ax.set_ylabel("Simulated tumor change at 8 weeks (%)")
    ax.set_title(
        f"Exploratory TGI ODE: simulated r={tgi_table['simulated_r_vs_single_dose_AUC'].iloc[0]:.2f}; "
        f"reported r={tumor_auc_correlation['r']}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "TGI_exploratory_simulated_percent_change.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_tgi_standardized_index_plot(tgi_table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tgi_table["single_dose_AUC_pred_ugh_mL"],
        tgi_table["expected_tumor_change_z_from_reported_r"],
        color="slateblue",
        marker="o",
        linewidth=2,
    )
    for _, row in tgi_table.iterrows():
        ax.annotate(
            f"{row['dose_mg']:g} mg",
            (row["single_dose_AUC_pred_ugh_mL"], row["expected_tumor_change_z_from_reported_r"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Single-dose model-predicted AUC (ug*h/mL)")
    ax.set_ylabel("Expected tumor-change z-score")
    ax.set_title(
        f"Reported standardized AUC-response index: r={tumor_auc_correlation['r']}, "
        f"P={tumor_auc_correlation['p_value']}, n={tumor_auc_correlation['n']}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "TGI_reported_auc_standardized_index.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_final_parameter_table(
    pk_fit: dict[str, object],
    pd_fit: dict[str, object],
    kd_base: float,
) -> pd.DataFrame:
    """Clean final parameter table used for downstream interpretation."""
    p50 = dose_dependent_pk_params(
        50.0,
        pk_fit["CL_ref"],
        pk_fit["alpha_CL"],
        pk_fit["Vss_ref"],
        pk_fit["alpha_Vss"],
        pk_fit["Q"],
        pk_fit["frac_V1"],
    )
    rows = [
        {
            "model": "PK",
            "parameter": "CL(dose)",
            "value": f"{pk_fit['CL_ref']:.3f} * (dose/50)^{pk_fit['alpha_CL']:.3f}",
            "unit": "L/h",
            "note": "Power fit to NCA CL values",
        },
        {
            "model": "PK",
            "parameter": "Vss(dose)",
            "value": f"{pk_fit['Vss_ref']:.1f} * (dose/50)^{pk_fit['alpha_Vss']:.3f}",
            "unit": "L",
            "note": "Power fit to NCA Vss values",
        },
        {"model": "PK", "parameter": "Q", "value": f"{pk_fit['Q']:.3f}", "unit": "L/h", "note": "Fitted with Cmax, AUC, and t1/2 objective"},
        {"model": "PK", "parameter": "frac_V1", "value": f"{pk_fit['frac_V1']:.3f}", "unit": "fraction", "note": "Central fraction of Vss"},
        {"model": "PK", "parameter": "V1(50 mg)", "value": f"{p50['V1']:.1f}", "unit": "L", "note": "Derived from Vss and frac_V1"},
        {"model": "PK", "parameter": "V2(50 mg)", "value": f"{p50['V2']:.1f}", "unit": "L", "note": "Derived from Vss and frac_V1"},
        {"model": "PD", "parameter": "ke0", "value": f"{pd_fit['ke0']:.4f}", "unit": "h^-1", "note": f"Fixed-Imax final set; t1/2={pd_fit['ke0_half_life_h']:.1f} h"},
        {"model": "PD", "parameter": "IC50", "value": f"{pd_fit['IC50']:.5f}", "unit": "ug/mL", "note": f"{pd_fit['IC50'] * 1000.0:.1f} ng/mL"},
        {"model": "PD", "parameter": "Imax", "value": f"{pd_fit['Imax']:.1f}", "unit": "fraction", "note": "Fixed for biological interpretability"},
        {"model": "TGI", "parameter": "kg", "value": f"{TGI_KG_H_INV:.4f}", "unit": "h^-1", "note": "Assumed, not fitted"},
        {"model": "TGI", "parameter": "kd_base", "value": f"{kd_base:.4f}", "unit": "h^-1 per ug/mL", "note": "Selected from stochastic variability scan"},
    ]
    return pd.DataFrame(rows)


def save_parameters(
    pk_fit: dict[str, object],
    pd_fit: dict[str, object],
    tgi_table: pd.DataFrame | None = None,
    stochastic_kd_base: float | None = None,
) -> Path:
    rows = [
        {"section": "PK", "parameter": "CL_ref_Lh_at_50mg", "value": pk_fit["CL_ref"]},
        {"section": "PK", "parameter": "alpha_CL", "value": pk_fit["alpha_CL"]},
        {"section": "PK", "parameter": "Vss_ref_L_at_50mg", "value": pk_fit["Vss_ref"]},
        {"section": "PK", "parameter": "alpha_Vss", "value": pk_fit["alpha_Vss"]},
        {"section": "PK", "parameter": "Q_Lh", "value": pk_fit["Q"]},
        {"section": "PK", "parameter": "frac_V1", "value": pk_fit["frac_V1"]},
        {"section": "PK", "parameter": "objective_weight_Cmax", "value": pk_fit["weights"]["Cmax"]},
        {"section": "PK", "parameter": "objective_weight_AUC", "value": pk_fit["weights"]["AUC"]},
        {"section": "PK", "parameter": "objective_weight_thalf", "value": pk_fit["weights"]["thalf"]},
        {"section": "PK", "parameter": "objective", "value": pk_fit["objective"]},
        {"section": "PD", "parameter": "dose_for_fitting_mg", "value": pd_fit["dose_for_fitting_mg"]},
        {"section": "PD", "parameter": "ke0_h_inv", "value": pd_fit["ke0"]},
        {"section": "PD", "parameter": "ke0_half_life_h", "value": pd_fit["ke0_half_life_h"]},
        {"section": "PD", "parameter": "IC50_ugmL", "value": pd_fit["IC50"]},
        {"section": "PD", "parameter": "Imax", "value": pd_fit["Imax"]},
        {"section": "PD", "parameter": "Imax_free", "value": pd_fit["Imax_free"]},
        {"section": "PD", "parameter": "objective", "value": pd_fit["objective"]},
    ]
    if tgi_table is not None and not tgi_table.empty:
        kd_value = stochastic_kd_base if stochastic_kd_base is not None else tgi_table["kd_h_inv_per_ugmL_scanned"].iloc[0]
        rows.extend(
            [
                {"section": "TGI", "parameter": "kg_h_inv_assumed", "value": tgi_table["kg_h_inv_assumed"].iloc[0]},
                {"section": "TGI", "parameter": "kd_base_h_inv_per_ugmL_stochastic", "value": kd_value},
                {"section": "TGI", "parameter": "kd_h_inv_per_ugmL_deterministic_directional", "value": tgi_table["kd_h_inv_per_ugmL_scanned"].iloc[0]},
                {"section": "TGI", "parameter": "simulated_r_vs_single_dose_AUC", "value": tgi_table["simulated_r_vs_single_dose_AUC"].iloc[0]},
                {"section": "TGI", "parameter": "reported_r", "value": tumor_auc_correlation["r"]},
            ]
        )
    path = OUTPUT_DIR / "model_parameters.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_model_qualification_table(pk_gof: pd.DataFrame, pd_gof: pd.DataFrame, pd_fit: dict[str, object]) -> pd.DataFrame:
    """Qualification checks requested for the final model."""
    def pk_value(dose: float, column: str) -> float:
        return float(pk_gof.loc[np.isclose(pk_gof["dose_mg"], dose), column].iloc[0])

    def pd_value(time_h: float, column: str) -> float:
        return float(pd_gof.loc[np.isclose(pd_gof["time_from_start_h"], time_h), column].iloc[0])

    c1_168 = pd_value(168.0, "C1_at_time_ugmL")
    ratio_168 = c1_168 / pd_fit["IC50"]

    checks = [
        ("Cmax at 75mg (ug/mL)", 1.195, pk_value(75.0, "Cmax_pred_ugmL"), 0.20),
        ("AUC at 75mg (ug*h/mL)", 12.7, pk_value(75.0, "AUC_pred_ugh_mL"), 0.20),
        ("t1/2 at 75mg (h)", 47.3, pk_value(75.0, "thalf_pred_h"), 0.15),
        ("Cmax at 50mg (ug/mL)", 0.982, pk_value(50.0, "Cmax_pred_ugmL"), 0.20),
        ("AUC at 50mg (ug*h/mL)", 10.7, pk_value(50.0, "AUC_pred_ugh_mL"), 0.20),
        ("Inhibition at 1.5h (%)", 95.0, pd_value(1.5, "inhibition_pred_pct"), 0.10),
        ("Inhibition at 168h (%)", 75.0, pd_value(168.0, "inhibition_pred_pct"), 0.10),
    ]

    rows = []
    for parameter, reference, predicted, threshold in checks:
        frac_diff = abs(predicted - reference) / reference
        rows.append(
            {
                "parameter": parameter,
                "reference_value": reference,
                "model_predicted_value": predicted,
                "percent_difference": 100.0 * frac_diff,
                "threshold_percent": 100.0 * threshold,
                "status": "PASS" if frac_diff <= threshold else "REVIEW",
            }
        )

    rows.append(
        {
            "parameter": "Trough/IC50 ratio at 168h",
            "reference_value": ">1",
            "model_predicted_value": ratio_168,
            "percent_difference": np.nan,
            "threshold_percent": np.nan,
            "status": "PASS" if ratio_168 > 1.0 else "REVIEW",
        }
    )
    return pd.DataFrame(rows)


def build_minimum_effective_dose_table(pk_fit: dict[str, object], pd_fit: dict[str, object]) -> tuple[pd.DataFrame, float | None]:
    """Dose scan for minimum dose maintaining >=50% inhibition at 168 h."""
    rows = []
    for dose in np.arange(10.0, 100.0 + 0.1, 5.0):
        t, C1, Ce, inhibition = simulate_pkpd(dose, pk_fit, pd_fit["ke0"], pd_fit["IC50"], pd_fit["Imax"])
        inh_168 = float(np.interp(168.0, t, inhibition))
        c1_168 = float(np.interp(168.0, t, C1))
        rows.append(
            {
                "dose_mg": dose,
                "C1_168h_ugmL": c1_168,
                "Ce_168h_ugmL": float(np.interp(168.0, t, Ce)),
                "inhibition_168h_frac": inh_168,
                "inhibition_168h_pct": 100.0 * inh_168,
                "meets_50pct_threshold": inh_168 >= 0.50,
            }
        )
    table = pd.DataFrame(rows)
    passing = table[table["meets_50pct_threshold"]]
    min_dose = float(passing["dose_mg"].iloc[0]) if not passing.empty else None
    return table, min_dose


def save_minimum_effective_dose_plot(dose_table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dose_table["dose_mg"], dose_table["inhibition_168h_pct"], color="seagreen", marker="o", linewidth=2)
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1.5, label="50% inhibition")
    passing = dose_table[dose_table["meets_50pct_threshold"]]
    if not passing.empty:
        min_dose = float(passing["dose_mg"].iloc[0])
        ax.axvline(min_dose, color="crimson", linestyle=":", linewidth=1.5, label=f"Minimum: {min_dose:g} mg")
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel("Inhibition at 168 h (%)")
    ax.set_title("Minimum dose maintaining >50% mTOR inhibition at 7 days")
    ax.set_ylim(0.0, 105.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "minimum_dose_50pct_inhibition_168h.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_tgi_stochastic_r_plot(stochastic_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stochastic_summary["kd_base_h_inv_per_ugmL"], stochastic_summary["median_r"], color="steelblue", linewidth=2)
    ax.fill_between(
        stochastic_summary["kd_base_h_inv_per_ugmL"],
        stochastic_summary["r_2p5"],
        stochastic_summary["r_97p5"],
        color="steelblue",
        alpha=0.18,
        label="95% simulation interval",
    )
    selected = stochastic_summary[stochastic_summary["selected_for_final"]]
    if not selected.empty:
        ax.scatter(selected["kd_base_h_inv_per_ugmL"], selected["median_r"], color="crimson", s=70, zorder=5, label="Selected kd")
    ax.axhline(tumor_auc_correlation["r"], color="black", linestyle="--", linewidth=1.5, label="Reported r")
    ax.axhspan(-0.5, -0.3, color="gray", alpha=0.12, label="Target range")
    ax.set_xscale("log")
    ax.set_xlabel("kd base (h^-1 per ug/mL)")
    ax.set_ylabel("Pearson r: AUC vs tumor change")
    ax.set_title("Stochastic TGI variability calibration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "TGI_stochastic_r_scan.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_clinical_summary_figure(
    profiles: dict[float, tuple[np.ndarray, np.ndarray]],
    pk_gof: pd.DataFrame,
    pk_fit: dict[str, object],
    pd_fit: dict[str, object],
    pd_gof: pd.DataFrame,
    multidose_table: pd.DataFrame,
) -> Path:
    """Single 4-panel clinical communication figure."""
    fig = plt.figure(figsize=(15, 11))
    outer = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    panel_d = outer[1, 1].subgridspec(2, 1, hspace=0.12)
    ax_d1 = fig.add_subplot(panel_d[0, 0])
    ax_d2 = fig.add_subplot(panel_d[1, 0], sharex=ax_d1)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(profiles)))
    for color, (dose, (t, C1)) in zip(colors, profiles.items()):
        mask = C1 > 0.0
        ax_a.semilogy(t[mask], C1[mask], color=color, linewidth=2, label=f"{dose:g} mg")
    ax_a.scatter(pk_gof["dose_mg"] * 0.0 + INFUSION_DURATION_H, pk_gof["Cmax_obs_ugmL"], color="black", s=35, zorder=6, label="Observed Cmax")
    for _, row in pk_gof.iterrows():
        ax_a.annotate(
            f"{row['dose_mg']:g} mg\nAUC {row['AUC_obs_ugh_mL']:.1f}",
            (INFUSION_DURATION_H, row["Cmax_obs_ugmL"]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=7,
        )
    ax_a.set_xlabel("Time (h)")
    ax_a.set_ylabel("Concentration (ug/mL)")
    ax_a.set_title("A. Single-dose PK, 0-168 h")
    ax_a.set_xlim(0.0, T_END_H)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(title="Dose", fontsize=8)

    ax_b.scatter(pk_gof["Cmax_obs_ugmL"], pk_gof["Cmax_pred_ugmL"], color="steelblue", marker="o", s=60, label="Cmax (ug/mL)")
    ax_b.scatter(pk_gof["AUC_obs_ugh_mL"], pk_gof["AUC_pred_ugh_mL"], color="darkorange", marker="s", s=60, label="AUC (ug*h/mL)")
    max_val = 1.1 * max(pk_gof["AUC_obs_ugh_mL"].max(), pk_gof["AUC_pred_ugh_mL"].max())
    ax_b.plot([0.0, max_val], [0.0, max_val], "k--", linewidth=1.2, label="Identity")
    ax_b.set_xlabel("Observed value")
    ax_b.set_ylabel("Predicted value")
    ax_b.set_title("B. PK goodness of fit")
    ax_b.set_xlim(0.0, max_val)
    ax_b.set_ylim(0.0, max_val)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=8)

    t_pd, _, Ce_pd, inhibition = simulate_pkpd(75.0, pk_fit, pd_fit["ke0"], pd_fit["IC50"], pd_fit["Imax"])
    ax_c.plot(t_pd, 100.0 * inhibition, color="seagreen", linewidth=2, label="Inhibition")
    ax_c.errorbar(
        pd_gof["time_from_start_h"],
        pd_gof["inhibition_pct"],
        yerr=5.0,
        fmt="o",
        color="black",
        ecolor="gray",
        elinewidth=1,
        capsize=3,
        label="Observed median +/-5%",
    )
    ax_c.set_xlabel("Time (h)")
    ax_c.set_ylabel("4E-BP1 inhibition (%)")
    ax_c.set_ylim(-5.0, 105.0)
    ax_c.set_xlim(0.0, T_END_H)
    ax_c.set_title("C. PD at 75 mg, fixed Imax")
    ax_c.grid(True, alpha=0.3)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(t_pd, Ce_pd, color="crimson", linestyle="--", linewidth=1.5, label="Ce")
    ax_c2.set_ylabel("Ce (ug/mL)")
    lines, labels = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines + lines2, labels + labels2, fontsize=8, loc="lower left")

    t_multi = multidose_table["time_h"].values
    c_multi = multidose_table["C1_ugmL"].values
    inh_multi = multidose_table["inhibition_pct"].values
    ax_d1.plot(t_multi, c_multi, color="steelblue", linewidth=1.6)
    ax_d1.axhline(pd_fit["IC50"], color="gray", linestyle="--", linewidth=1.2, label="IC50")
    ax_d1.set_ylabel("C1 (ug/mL)")
    ax_d1.set_title("D. 75 mg weekly x 8 weeks")
    ax_d1.grid(True, alpha=0.3)
    ax_d1.legend(fontsize=8)
    ax_d2.plot(t_multi, inh_multi, color="seagreen", linewidth=1.6)
    ax_d2.axhline(50.0, color="gray", linestyle="--", linewidth=1.2, label="50% inhibition")
    ax_d2.set_xlabel("Time (h)")
    ax_d2.set_ylabel("Inhibition (%)")
    ax_d2.set_ylim(-5.0, 105.0)
    ax_d2.grid(True, alpha=0.3)
    ax_d2.legend(fontsize=8)
    for ax in (ax_d1, ax_d2):
        for dose_time in weekly_dose_times():
            ax.axvline(dose_time, color="gray", linestyle=":", alpha=0.25)
        ax.set_xlim(0.0, MULTIDOSE_T_END_H)
    plt.setp(ax_d1.get_xticklabels(), visible=False)

    path = FIGURE_DIR / "clinical_simulation_4panel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    print("Fitting deforolimus PK/PD model...")
    print("Data rule: observed fits use only published summary values.")

    pk_fit = fit_pk_model(pk_data, weights=PK_OBJECTIVE_WEIGHTS)
    pk_gof, profiles = build_pk_gof_table(pk_data, pk_fit)

    pd_free_fit = fit_pd_model(pd_data, pk_fit, dose_for_fitting=75.0, free_imax=True)
    pd_fit = final_fixed_imax_pd_model(dose_for_fitting=75.0)
    pd_gof = build_pd_gof_table(pd_data, pk_fit, pd_fit)
    multidose_table = build_multidose_pkpd_table(pk_fit, pd_fit, dose=75.0)
    ke0_sensitivity_table = build_ke0_sensitivity_profiles(pk_fit, pd_fit, dose=75.0)
    stochastic_kd, stochastic_summary, stochastic_replicates, stochastic_example = build_stochastic_tgi_simulation(pk_gof, pk_fit)
    tgi_table, tgi_scan_table = build_tgi_exploratory_table(pk_gof, pk_fit)
    final_parameter_table = build_final_parameter_table(pk_fit, pd_fit, stochastic_kd)
    qualification_table = build_model_qualification_table(pk_gof, pd_gof, pd_fit)
    minimum_dose_table, minimum_effective_dose = build_minimum_effective_dose_table(pk_fit, pd_fit)

    print("\n=== FINAL MODEL PARAMETERS ===")
    print(final_parameter_table.to_string(index=False))

    parameter_path = save_parameters(pk_fit, pd_fit, tgi_table, stochastic_kd_base=stochastic_kd)
    final_parameter_path = OUTPUT_DIR / "final_model_parameters_table.csv"
    pk_gof_path = OUTPUT_DIR / "pk_goodness_of_fit.csv"
    pd_gof_path = OUTPUT_DIR / "pd_goodness_of_fit.csv"
    multidose_path = OUTPUT_DIR / "multidose_pkpd_75mg_8weeks.csv"
    ke0_sensitivity_path = OUTPUT_DIR / "pd_ke0_sensitivity_profiles.csv"
    tgi_path = OUTPUT_DIR / "tgi_exploratory_simulation.csv"
    tgi_scan_path = OUTPUT_DIR / "tgi_kd_scan.csv"
    tgi_stochastic_summary_path = OUTPUT_DIR / "tgi_stochastic_variability_summary.csv"
    tgi_stochastic_replicates_path = OUTPUT_DIR / "tgi_stochastic_replicate_results.csv"
    tgi_stochastic_example_path = OUTPUT_DIR / "tgi_stochastic_representative_virtual_patients.csv"
    qualification_path = OUTPUT_DIR / "model_qualification_table.csv"
    minimum_dose_path = OUTPUT_DIR / "minimum_dose_50pct_inhibition_168h.csv"
    pk_history_path = OUTPUT_DIR / "pk_optimization_starts.csv"
    pd_history_path = OUTPUT_DIR / "pd_optimization_starts.csv"
    pd_free_history_path = OUTPUT_DIR / "pd_free_imax_diagnostic_starts.csv"

    final_parameter_table.to_csv(final_parameter_path, index=False)
    pk_gof.to_csv(pk_gof_path, index=False)
    pd_gof.to_csv(pd_gof_path, index=False)
    multidose_table.to_csv(multidose_path, index=False)
    ke0_sensitivity_table.to_csv(ke0_sensitivity_path, index=False)
    tgi_table.to_csv(tgi_path, index=False)
    tgi_scan_table.to_csv(tgi_scan_path, index=False)
    stochastic_summary.to_csv(tgi_stochastic_summary_path, index=False)
    stochastic_replicates.to_csv(tgi_stochastic_replicates_path, index=False)
    stochastic_example.to_csv(tgi_stochastic_example_path, index=False)
    qualification_table.to_csv(qualification_path, index=False)
    minimum_dose_table.to_csv(minimum_dose_path, index=False)
    pk_fit["history"].to_csv(pk_history_path, index=False)
    pd_fit["history"].to_csv(pd_history_path, index=False)
    pd_free_fit["history"].to_csv(pd_free_history_path, index=False)

    plot_paths = [
        save_pk_nonlinear_plot(pk_fit),
        save_pk_profiles_plot(profiles),
        save_pk_gof_plot(pk_gof),
        save_pd_profile_plot(pk_fit, pd_fit, pd_gof),
        save_pd_gof_plot(pd_gof),
        save_pd_dose_comparison_plot(pk_fit, pd_fit),
        save_multidose_pkpd_plot(multidose_table),
        save_ke0_sensitivity_plot(ke0_sensitivity_table),
        save_tgi_plot(tgi_table),
        save_tgi_standardized_index_plot(tgi_table),
        save_tgi_stochastic_r_plot(stochastic_summary),
        save_minimum_effective_dose_plot(minimum_dose_table),
        save_clinical_summary_figure(profiles, pk_gof, pk_fit, pd_fit, pd_gof, multidose_table),
    ]

    print("\n=== OPTIMIZED PK PARAMETERS ===")
    print(f"CL_ref     = {pk_fit['CL_ref']:.6f} L/h at 50 mg")
    print(f"alpha_CL   = {pk_fit['alpha_CL']:.6f}")
    print(f"Vss_ref    = {pk_fit['Vss_ref']:.6f} L at 50 mg")
    print(f"alpha_Vss  = {pk_fit['alpha_Vss']:.6f}")
    print(f"Q          = {pk_fit['Q']:.6f} L/h")
    print(f"frac_V1    = {pk_fit['frac_V1']:.6f}")
    print(
        "Weights    = "
        f"Cmax {pk_fit['weights']['Cmax']:.1f}, "
        f"AUC {pk_fit['weights']['AUC']:.1f}, "
        f"thalf {pk_fit['weights']['thalf']:.1f}"
    )
    print(f"Objective  = {pk_fit['objective']:.6f}")

    print("\n=== PK GOODNESS OF FIT ===")
    print(
        pk_gof[
            [
                "dose_mg",
                "Cmax_obs_ugmL",
                "Cmax_pred_ugmL",
                "Cmax_pct_error",
                "AUC_obs_ugh_mL",
                "AUC_pred_ugh_mL",
                "AUC_pct_error",
                "thalf_obs_h",
                "thalf_pred_h",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )

    print("\n=== OPTIMIZED PD PARAMETERS ===")
    print("Final downstream PD uses the fixed-Imax interpretable parameter set.")
    print(f"Dose fit   = {pd_fit['dose_for_fitting_mg']:.1f} mg")
    print(f"ke0        = {pd_fit['ke0']:.8f} 1/h")
    print(f"ke0 half   = {pd_fit['ke0_half_life_h']:.4f} h")
    print(f"IC50       = {pd_fit['IC50']:.8f} ug/mL")
    print(f"Imax       = {pd_fit['Imax']:.6f} ({pd_fit['Imax_constraint']})")
    print(
        "Free-Imax diagnostic only: "
        f"Imax={pd_free_fit['Imax']:.4f}, ke0={pd_free_fit['ke0']:.4f} 1/h, "
        f"IC50={pd_free_fit['IC50']:.6f} ug/mL"
    )

    print("\n=== PD GOODNESS OF FIT ===")
    print(
        pd_gof[
            [
                "time_from_start_h",
                "inhibition_pct",
                "inhibition_pred_pct",
                "residual_pct_points",
                "C1_at_time_ugmL",
                "Ce_at_time_ugmL",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )

    print("\n=== 8-WEEK MULTIDOSE SUMMARY AT 75 MG ===")
    print(f"Cmax over 8 weeks = {multidose_table['C1_ugmL'].max():.4f} ug/mL")
    print(f"Ctrough at 1344 h = {multidose_table['C1_ugmL'].iloc[-1]:.4f} ug/mL")
    print(f"Inhibition at 1344 h = {multidose_table['inhibition_pct'].iloc[-1]:.2f}%")

    print("\n=== KE0 SENSITIVITY AT SELECTED TIMES ===")
    sensitivity_times = pd_data["time_from_start_h"].values
    sensitivity_rows = []
    for scenario, sub in ke0_sensitivity_table.groupby("scenario", sort=False):
        pred = np.interp(sensitivity_times, sub["time_h"].values, sub["inhibition_pct"].values)
        sensitivity_rows.append(
            {
                "scenario": scenario,
                "ke0_h_inv": sub["ke0_h_inv"].iloc[0],
                "inh_1p5h_pct": pred[1],
                "inh_24p5h_pct": pred[2],
                "inh_48p5h_pct": pred[3],
                "inh_168h_pct": pred[4],
            }
        )
    print(pd.DataFrame(sensitivity_rows).to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    print("\n=== EXPLORATORY TGI NOTE ===")
    print(
        "No individual tumor-size data were supplied. The deterministic TGI table is a "
        "no-variability directional check; the stochastic virtual-patient scan is the calibrated output."
    )
    print(
        tgi_table[
            [
                "dose_mg",
                "single_dose_AUC_pred_ugh_mL",
                "eight_week_AUC_0_1344_ugh_mL",
                "simulated_tumor_change_pct",
                "simulated_r_vs_single_dose_AUC",
                "reported_r",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )

    selected_stoch = stochastic_summary[stochastic_summary["selected_for_final"]].iloc[0]
    print("\n=== STOCHASTIC TGI VARIABILITY ===")
    print(
        f"N={VIRTUAL_TGI_N} virtual patients, {VIRTUAL_TGI_REPS} replicates, "
        f"kg sigma={VIRTUAL_TGI_KG_SIGMA}, sensitivity sigma={VIRTUAL_TGI_SENSITIVITY_SIGMA}"
    )
    print(
        f"Selected kd_base={stochastic_kd:.6f}; median r={selected_stoch['median_r']:.3f} "
        f"(95% simulation interval {selected_stoch['r_2p5']:.3f} to {selected_stoch['r_97p5']:.3f}); "
        f"reported r={tumor_auc_correlation['r']:.2f}"
    )

    print("\n=== MODEL QUALIFICATION TABLE ===")
    print(qualification_table.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    print("\n=== MINIMUM DOSE FOR >=50% INHIBITION AT 168H ===")
    if minimum_effective_dose is None:
        print("No dose from 10 to 100 mg maintained >=50% inhibition at 168 h.")
    else:
        print(f"Minimum dose = {minimum_effective_dose:.0f} mg")

    print("\nSaved CSV outputs:")
    for path in [
        parameter_path,
        final_parameter_path,
        pk_gof_path,
        pd_gof_path,
        multidose_path,
        ke0_sensitivity_path,
        tgi_path,
        tgi_scan_path,
        tgi_stochastic_summary_path,
        tgi_stochastic_replicates_path,
        tgi_stochastic_example_path,
        qualification_path,
        minimum_dose_path,
        pk_history_path,
        pd_history_path,
        pd_free_history_path,
    ]:
        print(f"  {path}")

    print("\nSaved plots:")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
