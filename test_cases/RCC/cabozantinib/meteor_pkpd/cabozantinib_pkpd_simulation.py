"""
Cabozantinib population PKPD simulation for RCC (METEOR-inspired).

Implements:
1) Population PK model (2-compartment, dual absorption, QD oral dosing, IIV)
2) Tumor growth PD model driven by time-varying exposure
3) PFS extended Cox-like simulation with time-varying Cavg3w
4) Safety ER Cox-like models with time-varying exposure windows
5) Full simulation pipeline and six-panel figure output

Notes:
- This is a virtual population simulation (no real patient data).
- Parameter values and calibration targets follow the user-specified prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")
np.random.seed(42)


# =============================================================================
# Global constants
# =============================================================================
N_PATIENTS = 300
SIM_DAYS = 365
SIM_HOURS = SIM_DAYS * 24
DT_HOURS = 1.0
TIME_H = np.arange(0, SIM_HOURS + DT_HOURS, DT_HOURS)
TIME_D = TIME_H / 24.0

DOSE_INTERVAL_H = 24.0
CHECK_INTERVAL_H = 14 * 24.0

# PK typical parameters
CL_TYPICAL = 2.17
VC_TYPICAL = 104.0
VP = 195.1
Q = 31.21
KA_REF = 0.979
D1 = 2.40
F1_TYPICAL = 0.854
ALAG1 = 0.784

# IIV variances
OMEGA2_CL = 0.202
OMEGA2_VC = 0.233
OMEGA2_KA = 2.063
OMEGA2_F1 = 0.466

# Residual error
SIGMA2_PK = 0.118
SIGMA_TUMOR = 5.0

# Dose modification model (calibrated, evaluated per 2-week check)
BASELINE_LOG_HAZARD_DMAK = float(np.log(0.0513))
BETA_CONC_DMAK = 0.0
MIN_TIME_BEFORE_FIRST_MOD_H = 28.0 * 24.0
P_REDUCE_60 = 0.60
P_HOLD_60 = 0.30
P_STAY_60 = 0.10
P_REDUCE_40 = 0.45
P_HOLD_40 = 0.35
P_STAY_40 = 0.20
HOLD_RETURN_PRIOR_PROB = 0.70
HOLD_MEAN_INTERVALS = 1.5

# Tumor model
KGROW_TYPICAL = 0.000488
KDMAX_TYPICAL = 0.002800
KDMAXTOT = 0.001777
KTOL = np.log(2.0) / 25.6
EC50_TUMOR = 251.0

# Tumor IIV
KGROW_CV = 0.30
KDMAX_CV = 0.25
Y0_MEAN = 60.0
Y0_CV = 0.72

# PFS
EC50_PFS = 100.0
BETA_DRUG_PFS = -2.5
TARGET_MEDIAN_PFS_MONTHS = 7.4

# Safety (target HR 60 mg vs 20 mg)
SAFETY_AE = {
    "PPES": {"window_h": 14 * 24.0, "target_hr": 2.21, "beta": np.log(2.21) / 750.0},
    "Fatigue": {"window_h": 14 * 24.0, "target_hr": 2.01, "beta": np.log(2.01) / 750.0},
    "Hypertension": {"window_h": 24.0, "target_hr": 1.85, "beta": np.log(1.85) / 750.0},
    "Diarrhea": {"window_h": 14 * 24.0, "target_hr": 1.78, "beta": np.log(1.78) / 750.0},
}

STARTING_DOSES = [20, 40, 60]

# To smoothly approximate lagged bolus into A1.
# A 1-hour pulse improves dose capture by LSODA relative to extremely short pulses.
SHORT_INFUSION_H = 1.0
PK_VALIDATION_CACHE_FILE = os.path.join(
    os.path.dirname(__file__),
    "pk_validation_cache.json",
)


@dataclass
class PKParams:
    """Container for individual PK parameters."""

    cl: float
    vc: float
    vp: float
    q: float
    ka: float
    f1: float
    f2: float


def _lognormal_samples_from_mean_cv(mean: float, cv: float, n: int) -> np.ndarray:
    """Generate lognormal samples from arithmetic mean and CV."""
    sigma = np.sqrt(np.log(1.0 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mu, sigma, n)


def _pk_validation_signature(n_patients: int, doses: List[int]) -> str:
    """Return a stable hash for PK validation-relevant settings."""
    payload = {
        "n_patients": int(n_patients),
        "doses": list(map(int, doses)),
        "SIM_DAYS": SIM_DAYS,
        "DT_HOURS": DT_HOURS,
        "CL_TYPICAL": CL_TYPICAL,
        "VC_TYPICAL": VC_TYPICAL,
        "VP": VP,
        "Q": Q,
        "KA_REF": KA_REF,
        "D1": D1,
        "F1_TYPICAL": F1_TYPICAL,
        "ALAG1": ALAG1,
        "OMEGA2_CL": OMEGA2_CL,
        "OMEGA2_VC": OMEGA2_VC,
        "OMEGA2_KA": OMEGA2_KA,
        "OMEGA2_F1": OMEGA2_F1,
        "SHORT_INFUSION_H": SHORT_INFUSION_H,
        "pk_engine": "rk4_substep_0.25h",
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_pk_validation_cache(signature: str) -> pd.DataFrame | None:
    """Load cached PK validation results if the signature matches."""
    if not os.path.exists(PK_VALIDATION_CACHE_FILE):
        return None

    try:
        with open(PK_VALIDATION_CACHE_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("signature") != signature:
        return None

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    return pd.DataFrame(rows)


def _save_pk_validation_cache(signature: str, df: pd.DataFrame) -> None:
    """Persist PK validation results for reuse across reruns."""
    payload = {
        "signature": signature,
        "rows": df.to_dict(orient="records"),
    }
    with open(PK_VALIDATION_CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def generate_virtual_population(n: int = N_PATIENTS) -> pd.DataFrame:
    """Generate virtual RCC patients and IIV random effects."""
    wt = _lognormal_samples_from_mean_cv(80.0, 0.20, n)
    wt = np.clip(wt, 40.0, 140.0)

    rho_cl_vc = 0.5
    cov_matrix = [
        [OMEGA2_CL, rho_cl_vc * np.sqrt(OMEGA2_CL * OMEGA2_VC)],
        [rho_cl_vc * np.sqrt(OMEGA2_CL * OMEGA2_VC), OMEGA2_VC],
    ]
    etas_cl_vc = np.random.multivariate_normal([0.0, 0.0], cov_matrix, n)
    eta_cl = etas_cl_vc[:, 0]
    eta_vc = etas_cl_vc[:, 1]

    sd_ka = np.sqrt(OMEGA2_KA)
    eta_ka = np.random.normal(0.0, sd_ka, n)
    eta_ka = np.clip(eta_ka, -2.0 * sd_ka, 2.0 * sd_ka)

    eta_f1 = np.random.normal(0.0, np.sqrt(OMEGA2_F1), n)

    y0 = _lognormal_samples_from_mean_cv(Y0_MEAN, Y0_CV, n)
    kgrow = _lognormal_samples_from_mean_cv(KGROW_TYPICAL, KGROW_CV, n)
    kdmax = _lognormal_samples_from_mean_cv(KDMAX_TYPICAL, KDMAX_CV, n)

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "WT": wt,
            "eta_CL": eta_cl,
            "eta_Vc": eta_vc,
            "eta_Ka": eta_ka,
            "eta_F1": eta_f1,
            "Y0": y0,
            "kgrow": kgrow,
            "kdmax": kdmax,
        }
    )


def calculate_individual_pk_params(patient: pd.Series, dose_mg: float) -> PKParams:
    """Calculate PK parameters for one patient at a given dose."""
    cl = CL_TYPICAL * np.exp(patient["eta_CL"])
    vc = VC_TYPICAL * (patient["WT"] / 80.0) ** 1.019 * np.exp(patient["eta_Vc"])

    ka_typ = KA_REF * (max(dose_mg, 1e-6) / 60.0) ** 0.677
    ka = ka_typ * np.exp(patient["eta_Ka"])
    ka = float(np.clip(ka, 0.2 * ka_typ, 5.0 * ka_typ))

    logit_f1_typ = np.log(F1_TYPICAL / (1.0 - F1_TYPICAL))
    f1 = 1.0 / (1.0 + np.exp(-(logit_f1_typ + patient["eta_F1"])))
    f1 = float(np.clip(f1, 0.01, 0.99))

    return PKParams(cl=cl, vc=vc, vp=VP, q=Q, ka=ka, f1=f1, f2=1.0 - f1)


def _short_infusion_rate(t: float, start: float, duration: float, amount_mg: float) -> float:
    """Return infusion rate (mg/h) for a short interval to approximate a bolus."""
    if start <= t < start + duration:
        return amount_mg / duration
    return 0.0


def _build_block_dose_times(start_h: float, end_h: float) -> np.ndarray:
    """Build QD dose times in [start_h, end_h)."""
    first = np.ceil(start_h / DOSE_INTERVAL_H) * DOSE_INTERVAL_H
    if np.isclose(first, start_h):
        first = start_h
    times = np.arange(first, end_h + 1e-9, DOSE_INTERVAL_H)
    return times[times < end_h + 1e-9]


def _pk_rhs(
    t: float,
    state: np.ndarray,
    block_dose_times: np.ndarray,
    current_dose: int,
    on_hold: bool,
    pkp: PKParams,
) -> np.ndarray:
    """Continuous PK right-hand side for one time point."""
    rate_fo = 0.0
    rate_zo = 0.0

    if len(block_dose_times) > 0 and current_dose > 0 and not on_hold:
        fo_amt = pkp.f1 * current_dose
        zo_rate = (pkp.f2 * current_dose) / D1
        for dt in block_dose_times:
            rate_fo += _short_infusion_rate(t, dt + ALAG1, SHORT_INFUSION_H, fo_amt)
            if dt <= t < dt + D1:
                rate_zo += zo_rate

    dy1 = rate_fo - pkp.ka * state[0]
    dy2 = (
        pkp.ka * state[0]
        + rate_zo
        - (pkp.cl / pkp.vc) * state[1]
        - pkp.q * (state[1] / pkp.vc - state[2] / pkp.vp)
    )
    dy3 = pkp.q * (state[1] / pkp.vc - state[2] / pkp.vp)
    return np.array([dy1, dy2, dy3], dtype=float)


def _simulate_pk_block_rk4(
    state: np.ndarray,
    t_block: np.ndarray,
    block_dose_times: np.ndarray,
    current_dose: int,
    on_hold: bool,
    pkp: PKParams,
    substep_h: float = 0.25,
) -> np.ndarray:
    """Integrate one PK block on the existing hourly grid using RK4 substeps."""
    y = state.astype(float).copy()
    y_out = np.zeros((3, len(t_block)), dtype=float)
    y_out[:, 0] = y

    for i in range(1, len(t_block)):
        t0 = float(t_block[i - 1])
        t1 = float(t_block[i])
        t = t0

        while t < t1 - 1e-12:
            h = min(substep_h, t1 - t)
            k1 = _pk_rhs(t, y, block_dose_times, current_dose, on_hold, pkp)
            k2 = _pk_rhs(t + 0.5 * h, y + 0.5 * h * k1, block_dose_times, current_dose, on_hold, pkp)
            k3 = _pk_rhs(t + 0.5 * h, y + 0.5 * h * k2, block_dose_times, current_dose, on_hold, pkp)
            k4 = _pk_rhs(t + h, y + h * k3, block_dose_times, current_dose, on_hold, pkp)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y = np.maximum(y, 0.0)
            t += h

        y_out[:, i] = y

    return y_out


def _rolling_mean_timewindow(time_h: np.ndarray, conc: np.ndarray, window_h: float) -> np.ndarray:
    """Vectorized rolling mean concentration over a backward time window."""
    dt = np.diff(time_h)
    cum = np.zeros(len(time_h))
    cum[1:] = np.cumsum(0.5 * (conc[:-1] + conc[1:]) * dt)
    out = np.zeros(len(time_h))
    for i, t in enumerate(time_h):
        t0 = t - window_h
        if t0 <= time_h[0]:
            dt_span = t - time_h[0]
            out[i] = cum[i] / dt_span if dt_span > 0 else conc[0]
        else:
            j = np.searchsorted(time_h, t0, side="left")
            dt_span = t - time_h[j]
            out[i] = (cum[i] - cum[j]) / dt_span if dt_span > 0 else conc[i]
    return out


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility trapezoidal integration across NumPy versions."""
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    return float(np.trapezoid(y, x))


def compute_cavg_steady_state(
    t_array: np.ndarray,
    c_array: np.ndarray,
    t_start: float = 300.0 * 24.0,
    t_end: float = 365.0 * 24.0,
) -> float:
    """Compute Cavg as AUC/tau over the true steady-state window."""
    t_hi = min(float(t_end), float(t_array[-1]))
    mask = (t_array >= t_start) & (t_array <= t_hi)
    if np.count_nonzero(mask) < 2:
        return float("nan")

    t_ss = t_array[mask]
    c_ss = c_array[mask]
    auc = _trapz_compat(c_ss, t_ss)
    tau_total = t_ss[-1] - t_ss[0]
    if tau_total <= 0:
        return float("nan")
    return auc / tau_total


def create_typical_patient() -> pd.Series:
    """Return a typical RCC patient with no IIV for PK validation."""
    return pd.Series(
        {
            "id": 0,
            "WT": 80.0,
            "eta_CL": 0.0,
            "eta_Vc": 0.0,
            "eta_Ka": 0.0,
            "eta_F1": 0.0,
            "Y0": Y0_MEAN,
            "kgrow": KGROW_TYPICAL,
            "kdmax": KDMAX_TYPICAL,
        }
    )


def simulate_pk(
    patient: pd.Series,
    starting_dose: int,
    enable_dose_modifications: bool = True,
    allow_dose_escalations: bool = True,
) -> Dict[str, object]:
    """Simulate one-patient PK profile with 2-week dose-modification checks.

    Uses a fixed-step RK4 block integrator on the hourly output grid.
    Concentration is reported in ng/mL (from mg/L * 1000).
    """
    # Internal PK units: amount in mg, volume in L, clearance in L/h.
    # Concentration is converted to ng/mL as: C_ng_mL = A2_mg * 1000 / Vc_L.
    n_t = len(TIME_H)
    a1 = np.zeros(n_t)
    a2 = np.zeros(n_t)
    a3 = np.zeros(n_t)
    dose_at_t = np.zeros(n_t)

    state = np.array([0.0, 0.0, 0.0], dtype=float)
    current_dose = int(starting_dose)
    hold_prior_dose = int(starting_dose)
    hold_remaining_intervals = 0
    dose_ceiling = int(starting_dose)

    # Geometric support starts at 1 interval; mean=1/p => p = 1/mean.
    hold_geom_p = 1.0 / HOLD_MEAN_INTERVALS

    dose_events: List[Tuple[float, int]] = []
    mod_events: List[Tuple[float, int, int]] = []

    block_starts = np.arange(0.0, SIM_HOURS, CHECK_INTERVAL_H)

    for bs in block_starts:
        be = min(bs + CHECK_INTERVAL_H, SIM_HOURS)
        block_mask = (TIME_H >= bs) & (TIME_H <= be)
        t_block = TIME_H[block_mask]

        if len(t_block) < 2:
            continue

        on_hold = hold_remaining_intervals > 0
        block_dose_times = _build_block_dose_times(bs, be)
        if on_hold or current_dose <= 0:
            block_dose_times = np.array([], dtype=float)

        pkp = calculate_individual_pk_params(patient, max(current_dose, 1e-6))

        if len(block_dose_times) > 0:
            for dt in block_dose_times:
                dose_events.append((float(dt), int(current_dose)))

        y_block = _simulate_pk_block_rk4(
            state=state,
            t_block=t_block,
            block_dose_times=block_dose_times,
            current_dose=current_dose,
            on_hold=on_hold,
            pkp=pkp,
        )

        idx = np.where(block_mask)[0]
        a1[idx] = y_block[0]
        a2[idx] = y_block[1]
        a3[idx] = y_block[2]
        dose_at_t[idx] = current_dose if (current_dose > 0 and not on_hold) else 0

        state = y_block[:, -1]

        # If patient is on hold, consume one interval and then resume/reduce.
        if hold_remaining_intervals > 0:
            hold_remaining_intervals -= 1
            if hold_remaining_intervals == 0:
                old_dose = current_dose
                if np.random.rand() < HOLD_RETURN_PRIOR_PROB:
                    candidate_dose = hold_prior_dose
                else:
                    if hold_prior_dose == 60:
                        candidate_dose = 40
                    elif hold_prior_dose == 40:
                        candidate_dose = 20
                    else:
                        candidate_dose = 20

                if not allow_dose_escalations:
                    candidate_dose = min(int(candidate_dose), int(dose_ceiling))
                current_dose = int(candidate_dose)

                if current_dose > 0:
                    dose_ceiling = min(dose_ceiling, current_dose)
                if current_dose != old_dose:
                    mod_events.append((float(be), int(old_dose), int(current_dose)))
            continue

        # Dose modification decision at block end (except at end of follow-up)
        if be >= SIM_HOURS or not enable_dose_modifications:
            continue
        if be < MIN_TIME_BEFORE_FIRST_MOD_H:
            continue

        c_block_ng = (a2[idx] / max(pkp.vc, 1e-9)) * 1000.0
        if len(c_block_ng) > 1:
            cavg_block = float(np.mean(c_block_ng))
        else:
            cavg_block = 0.0

        hazard = np.exp(BASELINE_LOG_HAZARD_DMAK + BETA_CONC_DMAK * cavg_block)
        p_mod = 1.0 - np.exp(-hazard)

        old_dose = current_dose
        if np.random.rand() < p_mod:
            u = np.random.rand()
            if current_dose == 60:
                if u < P_REDUCE_60:
                    current_dose = 40
                elif u < P_REDUCE_60 + P_HOLD_60:
                    hold_prior_dose = 60
                    current_dose = 0
                    hold_remaining_intervals = int(np.random.geometric(hold_geom_p))
                else:
                    # Administrative modification event with no effective dose change.
                    current_dose = 60
            elif current_dose == 40:
                if u < P_REDUCE_40:
                    current_dose = 20
                elif u < P_REDUCE_40 + P_HOLD_40:
                    hold_prior_dose = 40
                    current_dose = 0
                    hold_remaining_intervals = int(np.random.geometric(hold_geom_p))
                else:
                    current_dose = 40
            elif current_dose == 20:
                # No further reduction below 20 mg in this simplified calibration.
                current_dose = 20

        if current_dose > 0:
            dose_ceiling = min(dose_ceiling, current_dose)

        if current_dose != old_dose:
            mod_events.append((float(be), int(old_dose), int(current_dose)))

    vc_last = calculate_individual_pk_params(patient, max(current_dose, 1e-6)).vc
    conc_pred_ng = (a2 / max(vc_last, 1e-9)) * 1000.0
    eps = np.random.normal(0.0, np.sqrt(SIGMA2_PK), size=conc_pred_ng.shape)
    conc_obs_ng = np.clip(conc_pred_ng * np.exp(eps), 0.0, None)

    cavg1d = _rolling_mean_timewindow(TIME_H, conc_pred_ng, 24.0)
    cavg2w = _rolling_mean_timewindow(TIME_H, conc_pred_ng, 14 * 24.0)
    cavg3w = _rolling_mean_timewindow(TIME_H, conc_pred_ng, 21 * 24.0)

    return {
        "time_h": TIME_H.copy(),
        "conc_pred_ng_ml": conc_pred_ng,
        "conc_obs_ng_ml": conc_obs_ng,
        "cavg1d": cavg1d,
        "cavg2w": cavg2w,
        "cavg3w": cavg3w,
        "dose_at_t": dose_at_t,
        "dose_events": dose_events,
        "dose_mod_events": mod_events,
        "final_dose": current_dose,
    }


def validate_pk_targets() -> pd.DataFrame:
    """Run deterministic PK-only validation without IIV or dose modifications."""
    patient = create_typical_patient()
    rows = []

    for dose in STARTING_DOSES:
        pk = simulate_pk(patient, dose, enable_dose_modifications=False)
        cavg_day29 = compute_cavg_steady_state(
            pk["time_h"], pk["conc_pred_ng_ml"], t_start=24.0 * 24.0, t_end=29.0 * 24.0
        )
        cavg_day57 = compute_cavg_steady_state(
            pk["time_h"], pk["conc_pred_ng_ml"], t_start=52.0 * 24.0, t_end=57.0 * 24.0
        )
        rows.append(
            {
                "dose_mg": dose,
                "cavg_day29": cavg_day29,
                "cavg_day57": cavg_day57,
            }
        )

    return pd.DataFrame(rows)


def validate_pk_cavg(
    doses: List[int] | None = None,
    n_patients: int = 300,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Run PK-only validation (no DMAK) using Cavg over days 300-365."""
    if doses is None:
        doses = STARTING_DOSES

    if n_jobs is None:
        cpu = os.cpu_count() or 1
        n_jobs = max(1, cpu - 1)
    n_jobs = max(1, int(n_jobs))

    cl_ml_h = CL_TYPICAL * 1000.0
    targets = {20: 384.0, 40: 768.0, 60: 1152.0}
    patients = generate_virtual_population(n_patients)
    rows = []

    for dose in doses:
        cavg_theory = (dose * 1_000_000.0) / (cl_ml_h * 24.0)
        work_items = []
        for i, (_, p) in enumerate(patients.iterrows(), start=1):
            seed = 20_000_000 + dose * 100_000 + i
            work_items.append((p.to_dict(), dose, seed))

        if n_jobs == 1:
            cavg_list = list(map(_simulate_one_patient_pk_only_worker, work_items))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                cavg_list = list(ex.map(_simulate_one_patient_pk_only_worker, work_items, chunksize=4))

        med = float(np.nanmedian(cavg_list))
        tgt = targets.get(dose, np.nan)
        ratio = med / tgt if np.isfinite(tgt) and tgt > 0 else np.nan
        rows.append(
            {
                "dose_mg": dose,
                "theoretical_cavg_ss": cavg_theory,
                "median_cavg_ss": med,
                "target_cavg": tgt,
                "ratio": ratio,
                "within_5pct": abs(ratio - 1.0) <= 0.05 if np.isfinite(ratio) else False,
            }
        )

    return pd.DataFrame(rows)


def _simulate_one_patient_pk_only_worker(args: Tuple[dict, int, int]) -> float:
    """Worker for PK-only Cavg verification (no dose modifications)."""
    patient_dict, dose, seed = args
    np.random.seed(seed)
    p = pd.Series(patient_dict)
    pk_i = simulate_pk(p, dose, enable_dose_modifications=False)
    return float(
        compute_cavg_steady_state(
            pk_i["time_h"], pk_i["conc_pred_ng_ml"], t_start=300.0 * 24.0, t_end=365.0 * 24.0
        )
    )


def simulate_tumor(patient: pd.Series, pk: Dict[str, object]) -> Dict[str, object]:
    """Simulate tumor dynamics with resistance attenuation and RECIST outputs."""
    time_h = pk["time_h"]
    # Tumor model works on days; convert PK time axis explicitly.
    time_d = time_h / 24.0
    conc = pk["conc_pred_ng_ml"]

    days = np.arange(0, SIM_DAYS + 1)
    cavg_daily = np.zeros_like(days, dtype=float)

    for d in days:
        m = (time_d >= d) & (time_d < (d + 1))
        if np.count_nonzero(m) > 1:
            cavg_daily[d] = _trapz_compat(conc[m], time_d[m])
        elif np.count_nonzero(m) == 1:
            cavg_daily[d] = conc[m][0]

    kgrow = float(patient["kgrow"])
    kdmax = float(patient["kdmax"])
    y0 = float(patient["Y0"])

    def ode_tumor(t: float, y: np.ndarray) -> List[float]:
        d_idx = int(np.clip(np.floor(t), 0, SIM_DAYS))
        cavg = cavg_daily[d_idx]
        effect = cavg / (EC50_TUMOR + cavg)

        kd_eff = kdmax - KDMAXTOT * (1.0 - np.exp(-KTOL * t))
        kd_eff = max(kd_eff, 0.0)

        dydt = kgrow * y[0] - kd_eff * effect * y[0]
        return [dydt]

    sol = solve_ivp(
        ode_tumor,
        (0.0, SIM_DAYS),
        [y0],
        method="LSODA",
        t_eval=days.astype(float),
        rtol=1e-6,
        atol=1e-8,
    )

    if sol.success:
        y_pred = sol.y[0]
    else:
        y_pred = np.full(len(days), y0)

    y_obs = np.clip(y_pred + np.random.normal(0.0, SIGMA_TUMOR, len(y_pred)), 0.1, None)

    pct_change = 100.0 * (y_pred - y0) / max(y0, 1e-9)

    # RECIST assessment at protocol-specified 8-week windows only.
    assessment_days = np.array([56, 112, 168, 224, 280, 336], dtype=float)
    ranking = {"CR": 4, "PR": 3, "SD": 2, "PD": 1}
    best = "PD"

    # Use interpolated model-predicted size at protocol visits.
    interp = interp1d(days.astype(float), y_pred, kind="linear", bounds_error=False, fill_value=y_pred[-1])
    baseline = float(y0)
    nadir = baseline

    for ad in assessment_days:
        if ad > float(days[-1]):
            break
        yi = float(interp(ad))
        nadir = min(nadir, yi)

        p_base = 100.0 * (yi - baseline) / max(baseline, 1e-9)
        p_nadir = 100.0 * (yi - nadir) / max(nadir, 1e-9)

        if yi < 1.0:
            r = "CR"
        elif p_base <= -30.0:
            r = "PR"
        elif p_nadir >= 20.0 and p_base >= 20.0:
            r = "PD"
        else:
            r = "SD"

        if ranking[r] > ranking[best]:
            best = r

    return {
        "time_days": days,
        "tumor_pred_mm": y_pred,
        "tumor_obs_mm": y_obs,
        "pct_change": pct_change,
        "best_response": best,
    }


def _estimate_h0_for_target_median(beta_drug: float, target_median_months: float) -> float:
    """Calibrate baseline daily hazard for 60 mg reference effect."""
    ref_cavg = 1125.0
    ref_effect = ref_cavg / (EC50_PFS + ref_cavg)
    href = np.log(2.0) / (target_median_months * 30.0)
    return href / np.exp(beta_drug * ref_effect)


def simulate_pfs(pk: Dict[str, object], beta_drug: float = BETA_DRUG_PFS) -> Dict[str, object]:
    """Simulate PFS event time via daily piecewise hazard with Cavg3w covariate."""
    cavg3w = pk["cavg3w"]

    h0 = _estimate_h0_for_target_median(beta_drug, TARGET_MEDIAN_PFS_MONTHS)

    event_day = SIM_DAYS
    event = 0

    for d in range(1, SIM_DAYS + 1):
        i = int(d * 24)
        c = cavg3w[min(i, len(cavg3w) - 1)]
        drug_term = np.exp(beta_drug * (c / (EC50_PFS + c)))
        h_d = h0 * drug_term
        p = 1.0 - np.exp(-h_d)
        if np.random.rand() < p:
            event_day = d
            event = 1
            break

    return {"pfs_days": event_day, "event": event}


def _baseline_h0_ae(beta_ae: float, target_incidence_60: float = 0.45) -> float:
    """Set AE baseline hazard so 60 mg has reasonable annual incidence."""
    c60 = 1125.0
    multiplier = np.exp(beta_ae * c60)
    h_ref = -np.log(max(1.0 - target_incidence_60, 1e-9)) / SIM_DAYS
    return h_ref / multiplier


def simulate_safety(pk: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Simulate safety AE event times for all specified AEs."""
    out: Dict[str, Dict[str, object]] = {}
    time_h = pk["time_h"]
    conc = pk["conc_pred_ng_ml"]

    for ae, cfg in SAFETY_AE.items():
        window_h = cfg["window_h"]
        beta = cfg["beta"]
        cavg_w = _rolling_mean_timewindow(time_h, conc, window_h)

        h0 = _baseline_h0_ae(beta)
        event_day = SIM_DAYS
        event = 0

        for d in range(1, SIM_DAYS + 1):
            i = int(d * 24)
            c = cavg_w[min(i, len(cavg_w) - 1)]
            h_d = h0 * np.exp(beta * c)
            p = 1.0 - np.exp(-h_d)
            if np.random.rand() < p:
                event_day = d
                event = 1
                break

        out[ae] = {
            "event_day": event_day,
            "event": event,
            "cavg_window": cavg_w,
        }

    return out


def _km_curve(times: np.ndarray, events: np.ndarray, max_time: int = SIM_DAYS) -> pd.DataFrame:
    """Compute Kaplan-Meier with Greenwood CI."""
    df = pd.DataFrame({"time": times, "event": events}).sort_values("time")
    unique_t = np.sort(df.loc[df["time"] <= max_time, "time"].unique())

    at_risk = len(df)
    s = 1.0
    greenwood = 0.0

    rows = [{"time": 0.0, "surv": 1.0, "lcl": 1.0, "ucl": 1.0}]

    for t in unique_t:
        d = int(((df["time"] == t) & (df["event"] == 1)).sum())
        c = int(((df["time"] == t) & (df["event"] == 0)).sum())

        if at_risk > 0:
            if d > 0:
                s *= 1.0 - d / at_risk
                if at_risk - d > 0:
                    greenwood += d / (at_risk * (at_risk - d))

            if s > 0:
                se_s = s * np.sqrt(greenwood)
                lcl = max(0.0, s - 1.96 * se_s)
                ucl = min(1.0, s + 1.96 * se_s)
            else:
                lcl, ucl = 0.0, 0.0

            rows.append({"time": float(t), "surv": float(s), "lcl": float(lcl), "ucl": float(ucl)})

        at_risk -= (d + c)

    return pd.DataFrame(rows)


def _nelson_aalen(times: np.ndarray, events: np.ndarray, max_time: int = SIM_DAYS) -> pd.DataFrame:
    """Compute Nelson-Aalen cumulative hazard."""
    df = pd.DataFrame({"time": times, "event": events}).sort_values("time")
    unique_t = np.sort(df.loc[df["time"] <= max_time, "time"].unique())

    at_risk = len(df)
    cum_h = 0.0
    rows = [{"time": 0.0, "cum_haz": 0.0}]

    for t in unique_t:
        d = int(((df["time"] == t) & (df["event"] == 1)).sum())
        c = int(((df["time"] == t) & (df["event"] == 0)).sum())
        if at_risk > 0 and d > 0:
            cum_h += d / at_risk
        rows.append({"time": float(t), "cum_haz": float(cum_h)})
        at_risk -= (d + c)

    return pd.DataFrame(rows)


def _hazard_ratio_cox(
    times_ref: np.ndarray,
    events_ref: np.ndarray,
    times_cmp: np.ndarray,
    events_cmp: np.ndarray,
) -> float:
    """Compute Cox PH hazard ratio for comparison group vs reference group."""
    df = pd.DataFrame(
        {
            "duration": np.concatenate([times_ref, times_cmp]),
            "event": np.concatenate([events_ref, events_cmp]),
            "group": np.concatenate([np.zeros(len(times_ref)), np.ones(len(times_cmp))]),
        }
    )
    if df["event"].sum() <= 0:
        return float("nan")

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")
    return float(np.exp(cph.params_["group"]))


def _simulate_one_patient_worker(args: Tuple[dict, int, int, bool]) -> Dict[str, object]:
    """Worker to simulate one patient for one starting dose.

    Uses a deterministic per-patient seed so parallel and serial modes are reproducible.
    """
    patient_dict, dose, seed, allow_dose_escalations = args
    np.random.seed(seed)
    p = pd.Series(patient_dict)

    pk = simulate_pk(p, dose, allow_dose_escalations=allow_dose_escalations)
    tumor = simulate_tumor(p, pk)
    pfs = simulate_pfs(pk)
    safety = simulate_safety(pk)

    cavg_ss = compute_cavg_steady_state(
        pk["time_h"], pk["conc_pred_ng_ml"], t_start=300.0 * 24.0, t_end=365.0 * 24.0
    )
    final_pct = float(tumor["pct_change"][-1])
    bor = tumor["best_response"]

    row = {
        "id": int(p["id"]),
        "start_dose": dose,
        "cavg_ss": cavg_ss,
        "tumor_pct_final": final_pct,
        "bor": bor,
        "pfs_days": int(pfs["pfs_days"]),
        "pfs_event": int(pfs["event"]),
    }
    for ae in SAFETY_AE:
        row[f"{ae}_event_day"] = int(safety[ae]["event_day"])
        row[f"{ae}_event"] = int(safety[ae]["event"])

    mask_30d = TIME_H <= 30 * 24
    return {
        "row": row,
        "pk30": pk["conc_pred_ng_ml"][mask_30d],
        "tumor_pct": tumor["pct_change"],
    }


def run_full_simulation(
    n_patients: int = N_PATIENTS,
    starting_doses: List[int] | None = None,
    n_jobs: int | None = None,
    validate_pk_first: bool = True,
    pk_validation_n: int = 300,
    allow_dose_escalations: bool = True,
) -> Dict[str, object]:
    """Run full pipeline across dose groups and return all patient-level outputs."""
    if starting_doses is None:
        starting_doses = STARTING_DOSES

    if n_jobs is None:
        cpu = os.cpu_count() or 1
        n_jobs = max(1, cpu - 1)
    n_jobs = max(1, int(n_jobs))

    if validate_pk_first:
        validation_signature = _pk_validation_signature(pk_validation_n, starting_doses)
        df_pk_val = _load_pk_validation_cache(validation_signature)
        print("PK-only Cavg validation (IIV on, no DMAK; days 300-365)")
        print("Theoretical Cavg_ss formula: Dose*1e6 / (CL_mL_h*24)")
        if df_pk_val is None:
            df_pk_val = validate_pk_cavg(n_patients=pk_validation_n)
            _save_pk_validation_cache(validation_signature, df_pk_val)
            print(f"Computed new PK validation and cached it at {PK_VALIDATION_CACHE_FILE}")
        else:
            print(f"Using cached PK validation from {PK_VALIDATION_CACHE_FILE}")
        print(df_pk_val.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
        if not bool(df_pk_val["within_5pct"].all()):
            print("WARNING: PK Cavg out of +/-5%; check Ka dose scaling and bioavailable fraction handling.")
        print()

    patients = generate_virtual_population(n_patients)
    all_group_results: Dict[int, Dict[str, object]] = {}
    patient_rows = []

    def _fmt_seconds(seconds: float) -> str:
        total = int(max(0.0, seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m:02d}m {s:02d}s"

    for dose in starting_doses:
        print(f"Simulating group: {dose} mg (N={n_patients})")
        dose_start = time.perf_counter()

        pk30_list = []
        tumor_pct_list = []

        work_items = []
        for i, (_, p) in enumerate(patients.iterrows(), start=1):
            if i % 100 == 0:
                print(f"  {dose} mg: queued patient {i}/{n_patients}")
            seed = 10_000_000 + dose * 100_000 + i
            work_items.append((p.to_dict(), dose, seed, allow_dose_escalations))

        def _consume_results(results_iter) -> None:
            for i, out in enumerate(results_iter, start=1):
                if i % 25 == 0 or i == n_patients:
                    elapsed = time.perf_counter() - dose_start
                    rate = i / max(elapsed, 1e-9)
                    eta = (n_patients - i) / max(rate, 1e-9)
                    print(
                        f"  {dose} mg: completed {i}/{n_patients} "
                        f"| elapsed {_fmt_seconds(elapsed)} "
                        f"| ETA {_fmt_seconds(eta)}"
                    )
                patient_rows.append(out["row"])
                pk30_list.append(out["pk30"])
                tumor_pct_list.append(out["tumor_pct"])

        if n_jobs == 1:
            _consume_results(map(_simulate_one_patient_worker, work_items))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                _consume_results(ex.map(_simulate_one_patient_worker, work_items, chunksize=4))

        print(f"  {dose} mg: finished in {_fmt_seconds(time.perf_counter() - dose_start)}")

        all_group_results[dose] = {
            "pk30": np.array(pk30_list),
            "tumor_pct": np.array(tumor_pct_list),
        }

    patient_df = pd.DataFrame(patient_rows)

    cavg_mean = {
        d: float(np.mean(patient_df.loc[patient_df.start_dose == d, "cavg_ss"])) for d in [20, 40, 60]
    }
    print("Mean Cavg by starting dose group (with DMAK active):")
    print(f"  Started 20mg: actual mean Cavg = {cavg_mean[20]:.1f} ng/mL")
    print(f"  Started 40mg: actual mean Cavg = {cavg_mean[40]:.1f} ng/mL")
    print(f"  Started 60mg: actual mean Cavg = {cavg_mean[60]:.1f} ng/mL")

    if cavg_mean[20] > 400.0 and allow_dose_escalations:
        print("20 mg mean actual Cavg exceeded 400 ng/mL; re-running with DMAK dose escalations disabled.")
        return run_full_simulation(
            n_patients=n_patients,
            starting_doses=starting_doses,
            n_jobs=n_jobs,
            validate_pk_first=validate_pk_first,
            pk_validation_n=pk_validation_n,
            allow_dose_escalations=False,
        )

    return {"groups": all_group_results, "patients": patient_df}


def plot_results(results: Dict[str, object], save_path: str = "cabozantinib_pkpd_results.png") -> None:
    """Create the requested 6-panel figure."""
    groups = results["groups"]
    pdat = results["patients"]

    colors = {20: "#2E8B57", 40: "#D27D2D", 60: "#2459A8"}

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # 1) PK profiles first 30 days for 60 mg
    t30 = TIME_H[TIME_H <= 30 * 24] / 24.0
    conc_mat = groups[60]["pk30"]

    med = np.nanmedian(conc_mat, axis=0)
    p05 = np.nanpercentile(conc_mat, 5, axis=0)
    p95 = np.nanpercentile(conc_mat, 95, axis=0)

    ax1.fill_between(t30, p05, p95, alpha=0.25, color=colors[60], label="90% CI")
    ax1.plot(t30, med, color=colors[60], lw=2.0, label="Median")
    ax1.set_title("PK: 60 mg concentration (first 30 days)")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Concentration (ng/mL)")
    ax1.legend()

    # 2) Steady-state Cavg distributions
    bins = 40
    for d in [20, 40, 60]:
        vals = pdat.loc[pdat["start_dose"] == d, "cavg_ss"].values
        ax2.hist(vals, bins=bins, density=True, alpha=0.4, color=colors[d], label=f"{d} mg")
        ax2.axvline(np.nanmedian(vals), color=colors[d], linestyle="--")
    ax2.set_title("Steady-state Cavg distribution")
    ax2.set_xlabel("Cavg (ng/mL)")
    ax2.set_ylabel("Density")
    ax2.legend()

    # 3) Tumor median % change over time
    weeks = np.arange(0, SIM_DAYS + 1) / 7.0
    for d in [20, 40, 60]:
        mat = groups[d]["tumor_pct"]
        med_pct = np.nanmedian(mat, axis=0)
        ax3.plot(weeks, med_pct, color=colors[d], lw=2.0, label=f"{d} mg")
    ax3.axhline(-4.45, color=colors[20], linestyle=":", alpha=0.8)
    ax3.axhline(-9.1, color=colors[40], linestyle=":", alpha=0.8)
    ax3.axhline(-11.9, color=colors[60], linestyle=":", alpha=0.8)
    ax3.set_title("Tumor size: median % change from baseline")
    ax3.set_xlabel("Weeks")
    ax3.set_ylabel("% change")
    ax3.legend()

    # 4) KM PFS with 95% CI and HR annotations
    km_by_dose = {}
    for d in [20, 40, 60]:
        sub = pdat[pdat["start_dose"] == d]
        km = _km_curve(sub["pfs_days"].values, sub["pfs_event"].values)
        km_by_dose[d] = km
        t_mo = km["time"].values / 30.0
        ax4.step(t_mo, km["surv"].values, where="post", color=colors[d], lw=2.0, label=f"{d} mg")
        ax4.fill_between(
            t_mo,
            km["lcl"].values,
            km["ucl"].values,
            step="post",
            color=colors[d],
            alpha=0.15,
        )

    sub60 = pdat[pdat["start_dose"] == 60]
    sub40 = pdat[pdat["start_dose"] == 40]
    sub20 = pdat[pdat["start_dose"] == 20]

    hr40 = _hazard_ratio_cox(
        sub60["pfs_days"].values,
        sub60["pfs_event"].values,
        sub40["pfs_days"].values,
        sub40["pfs_event"].values,
    )
    hr20 = _hazard_ratio_cox(
        sub60["pfs_days"].values,
        sub60["pfs_event"].values,
        sub20["pfs_days"].values,
        sub20["pfs_event"].values,
    )

    ax4.text(0.03, 0.20, f"HR 40 vs 60: {hr40:.2f} (target 1.10)", transform=ax4.transAxes)
    ax4.text(0.03, 0.12, f"HR 20 vs 60: {hr20:.2f} (target 1.39)", transform=ax4.transAxes)
    ax4.set_title("PFS Kaplan-Meier (95% CI)")
    ax4.set_xlabel("Months")
    ax4.set_ylabel("PFS probability")
    ax4.set_ylim(0.0, 1.0)
    ax4.legend()

    # 5) Cumulative AE hazards (Nelson-Aalen)
    ae_colors = {
        "PPES": "#B22222",
        "Fatigue": "#8B3FBF",
        "Hypertension": "#8B5A2B",
        "Diarrhea": "#0E9C98",
    }
    ls = {20: ":", 40: "--", 60: "-"}

    for ae in SAFETY_AE:
        for d in [20, 40, 60]:
            sub = pdat[pdat["start_dose"] == d]
            na = _nelson_aalen(sub[f"{ae}_event_day"].values, sub[f"{ae}_event"].values)
            ax5.step(
                na["time"].values / 30.0,
                na["cum_haz"].values,
                where="post",
                color=ae_colors[ae],
                linestyle=ls[d],
                lw=1.6,
                label=f"{ae} {d} mg" if d == 60 else None,
            )

    ax5.set_title("Cumulative AE hazard (20/40/60 mg)")
    ax5.set_xlabel("Months")
    ax5.set_ylabel("Cumulative hazard")
    ax5.legend(fontsize=8, ncol=2)

    # 6) Best overall response bar chart
    categories = ["CR", "PR", "SD", "PD"]
    x = np.arange(len(categories))
    width = 0.24

    for i, d in enumerate([20, 40, 60]):
        sub = pdat[pdat["start_dose"] == d]
        vals = [100.0 * np.mean(sub["bor"] == c) for c in categories]
        ax6.bar(x + i * width, vals, width=width, color=colors[d], alpha=0.85, label=f"{d} mg")

    ax6.set_xticks(x + width)
    ax6.set_xticklabels(categories)
    ax6.set_ylabel("Patients (%)")
    ax6.set_title("Best overall response")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def print_summary_table(results: Dict[str, object]) -> None:
    """Print summary metrics versus prompt-specified targets."""
    pdat = results["patients"]

    target_cavg = {20: 375.0, 40: 750.0, 60: 1125.0}
    target_tumor = {20: -4.5, 40: -9.1, 60: -11.9}
    target_orr = {20: 8.7, 40: 15.6, 60: 19.1}

    print("\n" + "=" * 96)
    print("SIMULATION SUMMARY VS TARGETS")
    print("=" * 96)
    print(f"{'Metric':<36}{'20 mg':>14}{'40 mg':>14}{'60 mg':>14}{'Target':>18}")
    print("-" * 96)

    cavg_med = {d: np.median(pdat.loc[pdat.start_dose == d, "cavg_ss"]) for d in [20, 40, 60]}
    cavg_mean = {d: np.mean(pdat.loc[pdat.start_dose == d, "cavg_ss"]) for d in [20, 40, 60]}
    print(
        f"{'Steady-state Cavg (ng/mL)':<36}"
        f"{cavg_med[20]:>14.1f}{cavg_med[40]:>14.1f}{cavg_med[60]:>14.1f}"
        f"{'375/750/1125':>18}"
    )
    print("Mean actual Cavg by starting dose (with DMAK):")
    print(f"{'':<36}{cavg_mean[20]:>14.1f}{cavg_mean[40]:>14.1f}{cavg_mean[60]:>14.1f}{'':>18}")

    tumor_med = {
        d: np.median(pdat.loc[pdat.start_dose == d, "tumor_pct_final"]) for d in [20, 40, 60]
    }
    print(
        f"{'Tumor % change at ~1 year':<36}"
        f"{tumor_med[20]:>14.2f}{tumor_med[40]:>14.2f}{tumor_med[60]:>14.2f}"
        f"{'-4.5/-9.1/-11.9':>18}"
    )

    orr = {}
    for d in [20, 40, 60]:
        sub = pdat[pdat.start_dose == d]
        orr[d] = 100.0 * np.mean((sub["bor"] == "CR") | (sub["bor"] == "PR"))
    print(
        f"{'ORR (CR+PR) %':<36}"
        f"{orr[20]:>14.2f}{orr[40]:>14.2f}{orr[60]:>14.2f}"
        f"{'8.7/15.6/19.1':>18}"
    )

    sub60 = pdat[pdat.start_dose == 60]
    sub40 = pdat[pdat.start_dose == 40]
    sub20 = pdat[pdat.start_dose == 20]

    hr40 = _hazard_ratio_cox(
        sub60["pfs_days"].values,
        sub60["pfs_event"].values,
        sub40["pfs_days"].values,
        sub40["pfs_event"].values,
    )
    hr20 = _hazard_ratio_cox(
        sub60["pfs_days"].values,
        sub60["pfs_event"].values,
        sub20["pfs_days"].values,
        sub20["pfs_event"].values,
    )

    print(f"{'PFS HR 40 vs 60':<36}{'':>14}{hr40:>14.2f}{'':>14}{'target 1.10':>18}")
    print(f"{'PFS HR 20 vs 60':<36}{hr20:>14.2f}{'':>14}{'':>14}{'target 1.39':>18}")
    print("Note: HRs are Cox proportional hazards estimates.")

    print("\nSafety HRs (60 vs 20) from model betas and delta Cavg=750 ng/mL")
    print("-" * 96)
    print(f"{'AE':<20}{'Sim HR':>12}{'Target HR':>14}")
    for ae, cfg in SAFETY_AE.items():
        sim_hr = np.exp(cfg["beta"] * 750.0)
        print(f"{ae:<20}{sim_hr:>12.2f}{cfg['target_hr']:>14.2f}")

    print("=" * 96)


if __name__ == "__main__":
    print("Running Cabozantinib RCC PKPD simulation")
    print(f"N per group: {N_PATIENTS}")
    print(f"Duration: {SIM_DAYS} days")

    print("\nPK-only validation (typical patient, no IIV, no dose modifications)")
    print(validate_pk_targets().to_string(index=False, float_format=lambda x: f"{x:0.1f}"))

    print("\nPK-only Cavg validation (IIV on, no dose modifications; days 300-365)")
    print(validate_pk_cavg().to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    res = run_full_simulation(n_patients=N_PATIENTS, starting_doses=STARTING_DOSES)
    plot_results(res)
    print_summary_table(res)

    print("\nOutputs:")
    print("- Figure: cabozantinib_pkpd_results.png")
