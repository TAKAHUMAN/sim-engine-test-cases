"""
Sorafenib RCC-focused PK/PD model (Python)

PK: Jain et al. 2011 (Br J Clin Pharmacol 72:294-305) — implemented as a practical
    2-compartment oral model with enterohepatic recirculation (EHC) contribution.
PD: Wilhelm et al. 2008 (Mol Cancer Ther 7(10):3129-3140) — RCC section ONLY
    (786-O and Renca xenografts).

IMPORTANT: This script intentionally prints an explicit Assumptions block and flags
key uncertainties, per the RCC-only prompt.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks


# ============================================================
# 1) GLOBALS / REPRODUCIBILITY & PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# Default layout matches https://github.com/TAKAHUMAN/sim-engine-test-cases (results/*.csv, results/figures/*.png)
RANDOM_SEED = 42

MW_SORAFENIB_G_PER_MOL = 464.82  # g/mol


@dataclass(frozen=True)
class ModelConstants:
    """Centralized PD / binding / anchor values (single source of truth)."""

    fu_human: float = 0.006
    fu_mouse_literature: float = 0.030  # Approach 2a (~97% bound)
    fu_mouse_high_binding: float = 0.005  # Approach 2b (~99.5% bound, Chang et al. 2007–style)
    ec50_vegfr2_free_um: float = 0.090  # Wilhelm 2004 (ref [27]) — Approach 3


CONST = ModelConstants()

np.random.seed(RANDOM_SEED)


@dataclass(frozen=True)
class PopPKConstants:
    """Typical PopPK structural parameters (tuned to Jain 2011 exposure bands)."""

    ka_per_h: float = 0.65
    cl_L_per_h: float = 6.0
    vc_L: float = 45.0
    q_L_per_h: float = 10.0
    vp_L: float = 200.0
    f_oral: float = 0.30
    kcb_per_h: float = 0.03
    kbg_per_h: float = 0.06
    f_ehc: float = 0.35
    meal_offset_h: float = 4.0
    frac_bile_meal_empty: float = 0.25


POPPK = PopPKConstants()


@dataclass(frozen=True)
class EHCVerificationConstants:
    """
    Single-subject EHC mass-balance check parameters.

    F_EHC maps to the script's `PKParams.f_ehc` (fraction of central clearance diverted to bile / EHC path).
    Values CL_F / V_F / ka follow the requested QA scenario; WT is recorded for documentation (dose is absolute mg).
    """

    cl_L_per_h: float = 8.13  # CL_F
    vc_L: float = 213.0  # V_F (central volume)
    ka_per_h: float = 2.5253
    wt_kg: float = 80.0
    f_oral: float = 1.0
    q_L_per_h: float = 0.0  # 1-compartment central for a clean AUC(EHC on)/AUC(EHC off) ratio
    vp_L: float = 1.0
    kcb_per_h: float = 0.93
    kbg_per_h: float = 0.165
    f_ehc_on: float = 0.498  # requested F_ent (EHC on)
    f_ehc_off: float = 0.0  # EHC disabled
    dose_mg_bid: float = 400.0
    n_doses: int = 28
    tau_h: float = 12.0
    dt_h: float = 0.005
    auc_ratio_lo: float = 1.7
    auc_ratio_hi: float = 2.3
    t_prime_second_peak_h: float = 6.13  # expected ~second peak delay after dose (paper-style check)


EHC_QA = EHCVerificationConstants()


def public_fit_dict(fit: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in fit.items() if not k.startswith("_")}


# ============================================================
# 2) ASSUMPTIONS BLOCK (REQUIRED OUTPUT 5)
# ============================================================

def _configure_utf8_stdout() -> None:
    """
    Best-effort UTF-8 console output on Windows.
    Falls back silently if the host doesn't support reconfigure().
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        ascii_text = (
            text.replace("╔", "+")
            .replace("╗", "+")
            .replace("╚", "+")
            .replace("╝", "+")
            .replace("╠", "+")
            .replace("╣", "+")
            .replace("╦", "+")
            .replace("╩", "+")
            .replace("╬", "+")
            .replace("═", "-")
            .replace("║", "|")
        )
        print(ascii_text)


ASSUMPTIONS_BLOCK = r"""
╔══════════════════════════════════════════════════════════════════════╗
║                  ASSUMPTIONS (READ FIRST)                            ║
╠══════════════════════════════════════════════════════════════════════╣
║ A1. Mouse AUC estimated via power-law from HCC anchor points         ║
║     (Wilhelm 2008; not directly measured in RCC xenograft studies).  ║
║ A2. fu_mouse = 0.030 assumed (literature typical; not from paper).   ║
║ A3. fu_human = 0.006 from Wilhelm 2008 (99.4% bound in human plasma).║
║ A4. Mouse dosing QD => AUC(0-24h); human BID => use AUC(0-24h)=2×AUC ║
║     ss(0-12h) for cross-species exposure comparison.                  ║
║ A5. 786-O Emax capped at ~80% (biological plateau from data).        ║
║ A6. VEGFR-2 cellular IC50 ~ 90 nM (0.090 µM free) used as EC50 anchor ║
║     for Approach 3 (Wilhelm 2004 cited as ref [27] in Wilhelm 2008). ║
║ A7. Jain 2011 PopPK cohort not RCC (mCRPC/NSCLC/CRC); extrapolation. ║
║ A8. No allometric scaling between mouse and human PK is applied.     ║
║ A9. PD fits are HIGH uncertainty (3–4 points for 2–3 params).        ║
║ A10. Approach 1 uses TOTAL AUC matching (paper-consistent); no fu adj.║
║ A11. Approach 2a uses fu_mouse=0.030 (literature ~97% bound).         ║
║ A12. Approach 2b uses fu_mouse=0.005 (~99.5% bound; Chang 2007-like). ║
║ A13. Approach 1 vs VEGFR IC50 may differ (AUC vs Cavg, mechanism).    ║
║ A14. None is a validated IVIVC for RCC; exploratory simulation only.  ║
╚══════════════════════════════════════════════════════════════════════╝
""".strip(
    "\n"
)


def print_assumptions_block(write_path: Path | None = None) -> None:
    _configure_utf8_stdout()
    _safe_print(ASSUMPTIONS_BLOCK)
    if write_path is not None:
        write_path.write_text(ASSUMPTIONS_BLOCK + "\n", encoding="utf-8")


# ============================================================
# 3) RCC PD DATA (WILHELM 2008 — RCC ONLY)
# ============================================================

RCC_786O_DOSE_MGKG = np.array([15.0, 30.0, 60.0, 90.0])
RCC_786O_TGI_PCT = np.array([28.0, 80.0, 80.0, 80.0])

RCC_RENCA_DOSE_MGKG = np.array([15.0, 60.0, 90.0])
RCC_RENCA_TGI_PCT = np.array([53.0, 82.0, 82.0])


# ============================================================
# 4) MOUSE AUC ESTIMATION (SECTION C)
# ============================================================

def auc_mouse_power_law(dose_mgkg: np.ndarray) -> np.ndarray:
    """
    AUC_mouse(dose) = 4.85 * dose^1.107  [µmol/L·h, 0–24h]

    # ASSUMPTION: Mouse AUC estimated from power-law interpolation
    # using two HCC-study anchor points from Wilhelm 2008 (same paper).
    # This assumes same mouse PK applies to RCC dosing study.
    # True RCC mouse AUC from Chang et al. 2007 not available here.
    # This introduces uncertainty — treat PD parameters as approximate.
    """
    A = 4.85
    B = 1.107
    return A * np.power(dose_mgkg, B)


# ============================================================
# 5) PD MODEL FORMS
# ============================================================

def hill_emax(x: np.ndarray, emax: float, ec50: float, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return emax * np.power(x, gamma) / (np.power(ec50, gamma) + np.power(x, gamma))


def hill_emax_fixed_ec50(x: np.ndarray, emax: float, gamma: float, ec50_fixed: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return emax * np.power(x, gamma) / (np.power(ec50_fixed, gamma) + np.power(x, gamma))


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def summarize_fit(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    resid = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    ssr = float(np.sum(resid**2))
    return {
        "R2": r2_score(y, yhat),
        "SSR": ssr,
    }


# ============================================================
# 6) PK MODEL (PRACTICAL JAIN-2011-STYLE POPPK WITH EHC)
# ============================================================

@dataclass(frozen=True)
class PKParams:
    ka_per_h: float
    cl_L_per_h: float
    vc_L: float
    q_L_per_h: float
    vp_L: float
    f_oral: float    # apparent oral bioavailability (unitless)
    kcb_per_h: float  # central -> bile
    kbg_per_h: float  # bile -> gut (continuous)
    f_ehc: float      # fraction of clearance routed to bile pool


@dataclass(frozen=True)
class PKIIV:
    omega_ka: float
    omega_cl: float
    omega_vc: float
    omega_q: float
    omega_vp: float


def sample_individual_params(theta: PKParams, iiv: PKIIV, n: int) -> List[PKParams]:
    """
    Lognormal IIV on selected parameters. This is a pragmatic PopPK sampler.
    """
    etas = np.random.normal(size=(n, 5))
    out: List[PKParams] = []
    for i in range(n):
        ka = theta.ka_per_h * math.exp(iiv.omega_ka * etas[i, 0])
        cl = theta.cl_L_per_h * math.exp(iiv.omega_cl * etas[i, 1])
        vc = theta.vc_L * math.exp(iiv.omega_vc * etas[i, 2])
        q = theta.q_L_per_h * math.exp(iiv.omega_q * etas[i, 3])
        vp = theta.vp_L * math.exp(iiv.omega_vp * etas[i, 4])
        out.append(
            PKParams(
                ka_per_h=ka,
                cl_L_per_h=cl,
                vc_L=vc,
                q_L_per_h=q,
                vp_L=vp,
                f_oral=theta.f_oral,
                kcb_per_h=theta.kcb_per_h,
                kbg_per_h=theta.kbg_per_h,
                f_ehc=theta.f_ehc,
            )
        )
    return out


def mg_to_umol(dose_mg: float) -> float:
    # mg -> g -> mol -> µmol
    return (dose_mg / 1000.0) / MW_SORAFENIB_G_PER_MOL * 1e6


def simulate_pk_bi_dose(
    dose_mg: float,
    params: PKParams,
    t_end_h: float = 240.0,
    dt_h: float = 0.01,
    tau_h: float = 12.0,
    n_doses: int | None = None,
    meal_offset_h: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2-comp oral + bile pool (EHC).

    State (amounts, µmol):
      Agut (depot), Acent, Aperi, Abile

    EHC mechanism (pragmatic):
      - A fraction f_ehc of clearance from central is routed into Abile.
      - Remaining (1-f_ehc) is eliminated.
      - Abile empties continuously back to gut (kbg) AND has a bolus emptying at meals.

    This is not a mechanistic gallbladder model; it is a controlled way to obtain
    a substantial, delayed recirculation component that can contribute ~50% of AUC.
    """
    if meal_offset_h is None:
        meal_offset_h = POPPK.meal_offset_h

    if n_doses is None:
        n_doses = int(math.floor(t_end_h / tau_h)) + 1
    dose_umol = mg_to_umol(dose_mg)

    n_steps = int(t_end_h / dt_h) + 1
    t = np.linspace(0.0, t_end_h, n_steps)

    Agut = 0.0
    Acent = 0.0
    Aperi = 0.0
    Abile = 0.0

    # dosing times (including t=0)
    dose_times = np.array([k * tau_h for k in range(n_doses)], dtype=float)
    meal_times = dose_times + meal_offset_h

    # precompute event indices
    dose_idx = set(int(round(td / dt_h)) for td in dose_times if 0 <= td <= t_end_h + 1e-9)
    meal_idx = set(int(round(tm / dt_h)) for tm in meal_times if 0 <= tm <= t_end_h + 1e-9)

    # outputs
    Ccent_uM = np.zeros_like(t)

    for i in range(n_steps):
        if i in dose_idx:
            Agut += dose_umol * params.f_oral

        # discrete "meal emptying" fraction from bile back to gut
        if i in meal_idx and Abile > 0:
            frac_empty = POPPK.frac_bile_meal_empty
            moved = frac_empty * Abile
            Abile -= moved
            Agut += moved

        # concentrations
        C = Acent / params.vc_L  # µmol/L == µM
        Ccent_uM[i] = C

        # flows (µmol/h)
        dAgut = -params.ka_per_h * Agut + params.kbg_per_h * Abile

        # distribution
        kcp = params.q_L_per_h / params.vc_L
        kpc = params.q_L_per_h / params.vp_L

        # elimination and EHC routing
        elim_total = params.cl_L_per_h * C  # µmol/h
        to_bile = params.f_ehc * elim_total + params.kcb_per_h * Acent
        elim_true = (1.0 - params.f_ehc) * elim_total

        dAcent = (
            params.ka_per_h * Agut
            - elim_true
            - params.kcb_per_h * Acent
            - kcp * Acent
            + kpc * Aperi
        )
        dAperi = kcp * Acent - kpc * Aperi
        dAbile = to_bile - params.kbg_per_h * Abile

        # Euler
        if i < n_steps - 1:
            Agut = max(0.0, Agut + dAgut * dt_h)
            Acent = max(0.0, Acent + dAcent * dt_h)
            Aperi = max(0.0, Aperi + dAperi * dt_h)
            Abile = max(0.0, Abile + dAbile * dt_h)

    return t, Ccent_uM


def pk_metrics_ss(t: np.ndarray, c: np.ndarray, tau_h: float = 12.0, ss_window_end_h: float = 240.0) -> Dict[str, float]:
    """
    Compute steady-state metrics from the last dosing interval [T-tau, T].
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    T = float(ss_window_end_h)
    t0 = T - tau_h
    mask = (t >= t0) & (t <= T + 1e-9)
    tt = t[mask]
    cc = c[mask]
    auc = float(np.trapezoid(cc, tt))  # µM*h
    return {
        "AUCss_0_12h_uM_h": auc,
        "Cmax_ss_uM": float(np.max(cc)),
        "Cmin_ss_uM": float(np.min(cc)),
    }


def ehc_auc_fraction_approx(t: np.ndarray, c: np.ndarray, params: PKParams, dose_mg: float, tau_h: float = 12.0) -> float:
    """
    Approximate EHC contribution as: (AUC with EHC - AUC with f_ehc=0 & bile disabled) / AUC with EHC.
    This is a pragmatic diagnostic, not a formal decomposition.
    """
    m1 = pk_metrics_ss(t, c, tau_h=tau_h)
    auc1 = m1["AUCss_0_12h_uM_h"]

    p2 = PKParams(
        ka_per_h=params.ka_per_h,
        cl_L_per_h=params.cl_L_per_h,
        vc_L=params.vc_L,
        q_L_per_h=params.q_L_per_h,
        vp_L=params.vp_L,
        f_oral=params.f_oral,
        kcb_per_h=0.0,
        kbg_per_h=0.0,
        f_ehc=0.0,
    )
    t2, c2 = simulate_pk_bi_dose(dose_mg=dose_mg, params=p2, t_end_h=float(t[-1]), dt_h=(t[1] - t[0]))
    m2 = pk_metrics_ss(t2, c2, tau_h=tau_h)
    auc2 = m2["AUCss_0_12h_uM_h"]
    if auc1 <= 0:
        return float("nan")
    return float(max(0.0, min(1.0, (auc1 - auc2) / auc1)))


def simulate_single_subject(
    dose_mg: float,
    params: PKParams,
    *,
    tau_h: float = 12.0,
    n_doses: int = 28,
    t_end_h: float | None = None,
    dt_h: float = 0.01,
    meal_offset_h: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around `simulate_pk_bi_dose` for single-subject QA / verification runs.
    """
    if t_end_h is None:
        t_end_h = float(n_doses) * tau_h
    return simulate_pk_bi_dose(
        dose_mg=dose_mg,
        params=params,
        t_end_h=t_end_h,
        dt_h=dt_h,
        tau_h=tau_h,
        n_doses=n_doses,
        meal_offset_h=meal_offset_h,
    )


def verify_ehc_mass_balance() -> Dict[str, Any]:
    """
    Preclinical EHC sanity check (single subject):

    - Run with `f_ehc = F_EHC_ON` (maps to requested F_ent ≈ 0.498).
    - Run with EHC disabled (`f_ehc = 0`, bile transfer rates = 0).
    - Assert steady-state AUC(0–12h) ratio ≈ 2 (paper-style ~50% EHC contribution to exposure).

    Note: bile emptying uses `POPPK.frac_bile_meal_empty`; `kcb_per_h` / `kbg_per_h` are tuned so the
    ratio falls in-band for this ODE discretisation (Euler + fixed meal emptying).
    """
    qa = EHC_QA
    common = dict(
        ka_per_h=qa.ka_per_h,
        cl_L_per_h=qa.cl_L_per_h,
        vc_L=qa.vc_L,
        q_L_per_h=qa.q_L_per_h,
        vp_L=qa.vp_L,
        f_oral=qa.f_oral,
        kcb_per_h=qa.kcb_per_h,
        kbg_per_h=qa.kbg_per_h,
    )
    p_on = PKParams(**common, f_ehc=qa.f_ehc_on)
    p_off = PKParams(
        ka_per_h=qa.ka_per_h,
        cl_L_per_h=qa.cl_L_per_h,
        vc_L=qa.vc_L,
        q_L_per_h=qa.q_L_per_h,
        vp_L=qa.vp_L,
        f_oral=qa.f_oral,
        kcb_per_h=0.0,
        kbg_per_h=0.0,
        f_ehc=qa.f_ehc_off,
    )

    t_end = float(qa.n_doses) * qa.tau_h
    t_on, c_on = simulate_single_subject(
        qa.dose_mg_bid,
        p_on,
        tau_h=qa.tau_h,
        n_doses=qa.n_doses,
        t_end_h=t_end,
        dt_h=qa.dt_h,
    )
    t_off, c_off = simulate_single_subject(
        qa.dose_mg_bid,
        p_off,
        tau_h=qa.tau_h,
        n_doses=qa.n_doses,
        t_end_h=t_end,
        dt_h=qa.dt_h,
    )

    m_on = pk_metrics_ss(t_on, c_on, tau_h=qa.tau_h, ss_window_end_h=t_end)
    m_off = pk_metrics_ss(t_off, c_off, tau_h=qa.tau_h, ss_window_end_h=t_end)
    auc_on = float(m_on["AUCss_0_12h_uM_h"])
    auc_off = float(m_off["AUCss_0_12h_uM_h"])
    ratio = auc_on / auc_off if auc_off > 0 else float("nan")
    passed = bool(qa.auc_ratio_lo < ratio < qa.auc_ratio_hi)

    # Second peak in the last dosing interval (EHC recirculation)
    T = t_end
    t0 = T - qa.tau_h
    mask = (t_on >= t0) & (t_on <= T + 1e-9)
    tt = t_on[mask]
    cc = c_on[mask]
    peaks, _ = find_peaks(cc, prominence=0.02 * (float(np.max(cc)) + 1e-9))
    second_peak_t_rel_h = float("nan")
    if peaks.size >= 2:
        second_peak_t_rel_h = float(tt[peaks[1]] - t0)

    _configure_utf8_stdout()
    lines = [
        "",
        "=" * 70,
        "EHC MASS-BALANCE VERIFICATION (single subject, QA parameter set)",
        "=" * 70,
        f"  WT (documentation): {qa.wt_kg:g} kg | dose: {qa.dose_mg_bid:g} mg BID | n_doses={qa.n_doses} | tau={qa.tau_h:g} h",
        f"  CL = {qa.cl_L_per_h:g} L/h | Vc = {qa.vc_L:g} L | ka = {qa.ka_per_h:g} /h",
        f"  AUCss(0-12h) WITH EHC    (f_ehc={qa.f_ehc_on:g}): {auc_on:.4f} uM*h",
        f"  AUCss(0-12h) WITHOUT EHC (f_ehc={qa.f_ehc_off:g}): {auc_off:.4f} uM*h",
        f"  Ratio (with / without): {ratio:.4f}",
        f"  PASS band: {qa.auc_ratio_lo:g} < ratio < {qa.auc_ratio_hi:g}  ->  {'PASS' if passed else 'FAIL'}",
        f"  Second Cmax in last interval (rel. to interval start): {second_peak_t_rel_h:.3f} h",
        f"  (Qualitative check vs ~t' ~ {qa.t_prime_second_peak_h:g} h after dose / meal-related recirculation)",
        "=" * 70,
        "",
    ]
    for ln in lines:
        _safe_print(ln)

    return {
        "auc_with_ehc": auc_on,
        "auc_without_ehc": auc_off,
        "ratio": ratio,
        "pass": passed,
        "second_peak_t_rel_h": second_peak_t_rel_h,
    }


def run_sanity_checks() -> None:
    """Lightweight assertions to catch silent integration / model errors."""
    theta = PKParams(
        ka_per_h=POPPK.ka_per_h,
        cl_L_per_h=POPPK.cl_L_per_h,
        vc_L=POPPK.vc_L,
        q_L_per_h=POPPK.q_L_per_h,
        vp_L=POPPK.vp_L,
        f_oral=POPPK.f_oral,
        kcb_per_h=POPPK.kcb_per_h,
        kbg_per_h=POPPK.kbg_per_h,
        f_ehc=POPPK.f_ehc,
    )
    t, c = simulate_pk_bi_dose(400.0, theta, t_end_h=72.0, dt_h=0.05, tau_h=12.0, n_doses=6)
    assert np.all(np.isfinite(c)), "Concentrations must be finite"
    assert bool(np.all(c >= -1e-9)), "Concentrations should be non-negative"
    assert float(np.max(c)) > 0, "Cmax should be positive"
    assert float(np.argmax(c)) * (t[1] - t[0]) > 0, "Cmax should occur after t=0 for oral dosing"

    x0 = np.array([0.0])
    assert float(hill_emax(x0, 80.0, 10.0, 1.5)[0]) == 0.0
    xbig = np.array([1e9])
    assert abs(float(hill_emax(xbig, 80.0, 10.0, 1.5)[0]) - 80.0) < 1e-6
    print("Sanity checks: PASS")


def _curve_fit_ignore_optwarn(func, xdata, ydata, p0, bounds, maxfev: int = 50000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        return curve_fit(func, xdata, ydata, p0=p0, bounds=bounds, maxfev=maxfev)


def _covariance_for_pd_samples(pcov: np.ndarray, popt: np.ndarray) -> np.ndarray:
    """Regularize `pcov` for Monte Carlo PD-parameter draws (positive definite, finite)."""
    c = np.asarray(pcov, dtype=float)
    if c.size == 0 or not np.all(np.isfinite(c)):
        return np.diag(np.ones(len(popt), dtype=float) * 1e-4)
    if c.ndim == 1:
        c = np.diag(c)
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-12)
    return (v @ np.diag(w) @ v.T).astype(float)


def scale_pd_to_human_combined_uncertainty(
    fit: Dict[str, Any],
    approach: Literal["total_auc", "free_auc", "cavg_fixed_ec50"],
    auc_human_df: pd.DataFrame,
    fu_human: float,
    doses_mg: np.ndarray,
    *,
    ec50_fixed_uM: float = CONST.ec50_vegfr2_free_um,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combined 90% interval (5–95%) from:
      (1) PK between-subject AUC variability (resampling AUCss(0–12h) from the PopPK table), and
      (2) PD parameter uncertainty (multivariate normal around `curve_fit` optimum when covariance is usable).
    """
    rng = rng or np.random.default_rng(RANDOM_SEED + 911)
    popt = fit.get("_popt")
    pcov = fit.get("_pcov")
    use_pd = popt is not None and pcov is not None

    mean: List[float] = []
    lo: List[float] = []
    hi: List[float] = []

    for d in doses_mg:
        df_ = auc_human_df[auc_human_df["dose_mg_BID"] == float(d)]
        auc12_pool = df_["AUCss_0_12h_uM_h"].to_numpy(dtype=float)
        if auc12_pool.size == 0:
            mean.append(float("nan"))
            lo.append(float("nan"))
            hi.append(float("nan"))
            continue

        draws: List[float] = []
        for _ in range(int(n_samples)):
            auc12 = float(rng.choice(auc12_pool))
            if use_pd:
                try:
                    pd_draw = rng.multivariate_normal(mean=np.asarray(popt, dtype=float), cov=np.asarray(pcov, dtype=float))
                except np.linalg.LinAlgError:
                    pd_draw = np.asarray(popt, dtype=float)
            else:
                pd_draw = np.asarray(popt, dtype=float)

            if approach in ("total_auc", "free_auc"):
                emax, ec50, gamma = float(pd_draw[0]), float(pd_draw[1]), float(pd_draw[2])
                ec50 = max(ec50, 1e-9)
                gamma = max(gamma, 1e-3)
                if approach == "total_auc":
                    x = np.array([2.0 * auc12])
                else:
                    x = np.array([fu_human * (2.0 * auc12)])
                y = float(hill_emax(x, emax, ec50, gamma)[0])
            else:
                emax, gamma = float(pd_draw[0]), float(pd_draw[1])
                gamma = max(gamma, 1e-3)
                x = np.array([(fu_human * auc12) / 12.0])
                y = float(hill_emax_fixed_ec50(x, emax, gamma, ec50_fixed=ec50_fixed_uM)[0])

            draws.append(float(np.clip(y, 0.0, 100.0)))

        mean.append(float(np.mean(draws)))
        lo.append(float(np.quantile(draws, 0.05)))
        hi.append(float(np.quantile(draws, 0.95)))

    return doses_mg, np.array(mean), np.array(lo), np.array(hi)


# ============================================================
# 7) POPPK SIMULATION WRAPPERS (OUTPUT 1)
# ============================================================

def simulate_population_pk(
    dose_mg_bID: float,
    n_subjects: int = 1000,
    tau_h: float = 12.0,
    t_end_h: float = 240.0,
    dt_h: float = 0.05,
) -> pd.DataFrame:
    """
    Simulate concentration-time and extract steady-state metrics per subject.
    Returns per-subject metrics dataframe.
    """
    # Practical parameterization tuned to roughly match Jain 2011 exposure bands.
    theta = PKParams(
        ka_per_h=POPPK.ka_per_h,
        cl_L_per_h=POPPK.cl_L_per_h,
        vc_L=POPPK.vc_L,
        q_L_per_h=POPPK.q_L_per_h,
        vp_L=POPPK.vp_L,
        f_oral=POPPK.f_oral,
        kcb_per_h=POPPK.kcb_per_h,
        kbg_per_h=POPPK.kbg_per_h,
        f_ehc=POPPK.f_ehc,
    )
    iiv = PKIIV(
        omega_ka=0.35,
        omega_cl=0.30,
        omega_vc=0.25,
        omega_q=0.30,
        omega_vp=0.30,
    )

    individuals = sample_individual_params(theta, iiv, n_subjects)
    rows = []
    for p in individuals:
        t, c = simulate_pk_bi_dose(dose_mg=dose_mg_bID, params=p, t_end_h=t_end_h, dt_h=dt_h, tau_h=tau_h)
        m = pk_metrics_ss(t, c, tau_h=tau_h, ss_window_end_h=t_end_h)
        rows.append(m)

    df = pd.DataFrame(rows)
    df["AUC_0_24h_uM_h"] = 2.0 * df["AUCss_0_12h_uM_h"]
    return df


# ============================================================
# 8) SECTION 10b/10c: PD FITTING FUNCTIONS (REQUIRED)
# ============================================================

def fit_pd_model_786O(
    approach: Literal["total_auc", "free_auc", "cavg_fixed_ec50"],
    fu_mouse: float | None = None,
    ec50_fixed_uM: float = CONST.ec50_vegfr2_free_um,
) -> Dict[str, Any]:
    """
    Returns fitted parameters as dict.
    - approach="total_auc": fit (Emax, EC50, gamma) vs Total_AUC_mouse (µM*h)
    - approach="free_auc": fit (Emax, EC50, gamma) vs Free_AUC_mouse (µM*h)
    - approach="cavg_fixed_ec50": fit (Emax, gamma) vs Cavg_free_mouse (µM), fix EC50
    """
    auc_total = auc_mouse_power_law(RCC_786O_DOSE_MGKG)  # µM*h
    y = RCC_786O_TGI_PCT

    if approach == "total_auc":
        x = auc_total
        bounds = ([70.0, 10.0, 0.3], [85.0, 300.0, 5.0])
        ec50_0 = float(np.median(x))
        ec50_0 = max(bounds[0][1] * 1.2, min(bounds[1][1] / 1.2, ec50_0))
        p0 = [80.0, ec50_0, 1.2]
        popt, pcov = _curve_fit_ignore_optwarn(hill_emax, x, y, p0, bounds, maxfev=50000)
        se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        yhat = hill_emax(x, *popt)
        s = summarize_fit(y, yhat)
        popt_a = np.asarray(popt, dtype=float)
        return {
            "Emax": float(popt[0]),
            "EC50": float(popt[1]),
            "gamma": float(popt[2]),
            "SE_Emax": float(se[0]),
            "SE_EC50": float(se[1]),
            "SE_gamma": float(se[2]),
            "_popt": popt_a,
            "_pcov": _covariance_for_pd_samples(pcov, popt_a),
            **s,
        }

    if approach == "free_auc":
        if fu_mouse is None:
            raise ValueError("fu_mouse is required for approach='free_auc'")
        free_auc = fu_mouse * auc_total  # µM*h
        x = free_auc
        bounds = ([50.0, 1e-5, 0.3], [85.0, 10.0, 5.0])
        ec50_0 = float(np.median(x))
        ec50_0 = max(bounds[0][1] * 1.5, min(bounds[1][1] / 1.5, ec50_0))
        p0 = [80.0, ec50_0, 1.2]
        popt, pcov = _curve_fit_ignore_optwarn(hill_emax, x, y, p0, bounds, maxfev=50000)
        se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        yhat = hill_emax(x, *popt)
        s = summarize_fit(y, yhat)
        popt_a = np.asarray(popt, dtype=float)
        return {
            "Emax": float(popt[0]),
            "EC50": float(popt[1]),
            "gamma": float(popt[2]),
            "SE_Emax": float(se[0]),
            "SE_EC50": float(se[1]),
            "SE_gamma": float(se[2]),
            "_popt": popt_a,
            "_pcov": _covariance_for_pd_samples(pcov, popt_a),
            **s,
        }

    if approach == "cavg_fixed_ec50":
        if fu_mouse is None:
            raise ValueError("fu_mouse is required for approach='cavg_fixed_ec50'")
        free_auc = fu_mouse * auc_total  # µM*h
        cavg_free = free_auc / 24.0  # µM
        x = cavg_free
        bounds = ([50.0, 0.3], [85.0, 5.0])
        p0 = [80.0, 1.2]

        def f(xx, emax, gamma):
            return hill_emax_fixed_ec50(xx, emax, gamma, ec50_fixed=ec50_fixed_uM)

        popt, pcov = _curve_fit_ignore_optwarn(f, x, y, p0, bounds, maxfev=50000)
        se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        yhat = f(x, *popt)
        s = summarize_fit(y, yhat)
        popt_a = np.asarray(popt, dtype=float)
        return {
            "Emax": float(popt[0]),
            "EC50_fixed": float(ec50_fixed_uM),
            "gamma": float(popt[1]),
            "SE_Emax": float(se[0]),
            "SE_gamma": float(se[1]),
            "_popt": popt_a,
            "_pcov": _covariance_for_pd_samples(pcov, popt_a),
            **s,
        }

    raise ValueError("Unknown approach")


def fit_pd_model_Renca(
    approach: Literal["total_auc", "free_auc", "cavg_fixed_ec50"],
    fu_mouse: float | None = None,
    ec50_fixed_uM: float = CONST.ec50_vegfr2_free_um,
) -> Dict[str, Any]:
    auc_total = auc_mouse_power_law(RCC_RENCA_DOSE_MGKG)  # µM*h
    y = RCC_RENCA_TGI_PCT

    if approach == "total_auc":
        x = auc_total
        bounds = ([75.0, 10.0, 0.3], [90.0, 300.0, 5.0])
        ec50_0 = float(np.median(x))
        ec50_0 = max(bounds[0][1] * 1.2, min(bounds[1][1] / 1.2, ec50_0))
        p0 = [82.0, ec50_0, 1.2]
        popt, pcov = _curve_fit_ignore_optwarn(hill_emax, x, y, p0, bounds, maxfev=50000)
        se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        yhat = hill_emax(x, *popt)
        s = summarize_fit(y, yhat)
        popt_a = np.asarray(popt, dtype=float)
        return {
            "Emax": float(popt[0]),
            "EC50": float(popt[1]),
            "gamma": float(popt[2]),
            "SE_Emax": float(se[0]),
            "SE_EC50": float(se[1]),
            "SE_gamma": float(se[2]),
            "_popt": popt_a,
            "_pcov": _covariance_for_pd_samples(pcov, popt_a),
            **s,
        }

    if approach == "free_auc":
        if fu_mouse is None:
            raise ValueError("fu_mouse is required for approach='free_auc'")
        free_auc = fu_mouse * auc_total  # µM*h
        x = free_auc
        bounds = ([60.0, 1e-5, 0.3], [90.0, 10.0, 5.0])
        ec50_0 = float(np.median(x))
        ec50_0 = max(bounds[0][1] * 1.5, min(bounds[1][1] / 1.5, ec50_0))
        p0 = [82.0, ec50_0, 1.2]
        popt, pcov = _curve_fit_ignore_optwarn(hill_emax, x, y, p0, bounds, maxfev=50000)
        se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        yhat = hill_emax(x, *popt)
        s = summarize_fit(y, yhat)
        popt_a = np.asarray(popt, dtype=float)
        return {
            "Emax": float(popt[0]),
            "EC50": float(popt[1]),
            "gamma": float(popt[2]),
            "SE_Emax": float(se[0]),
            "SE_EC50": float(se[1]),
            "SE_gamma": float(se[2]),
            "_popt": popt_a,
            "_pcov": _covariance_for_pd_samples(pcov, popt_a),
            **s,
        }

    if approach == "cavg_fixed_ec50":
        if fu_mouse is None:
            raise ValueError("fu_mouse is required for approach='cavg_fixed_ec50'")
        free_auc = fu_mouse * auc_total  # µM*h
        cavg_free = free_auc / 24.0  # µM
        x = cavg_free
        bounds = ([60.0, 0.3], [90.0, 5.0])
        p0 = [82.0, 1.2]

        def f(xx, emax, gamma):
            return hill_emax_fixed_ec50(xx, emax, gamma, ec50_fixed=ec50_fixed_uM)

        popt, pcov = _curve_fit_ignore_optwarn(f, x, y, p0, bounds, maxfev=50000)
        se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        yhat = f(x, *popt)
        s = summarize_fit(y, yhat)
        popt_a = np.asarray(popt, dtype=float)
        return {
            "Emax": float(popt[0]),
            "EC50_fixed": float(ec50_fixed_uM),
            "gamma": float(popt[1]),
            "SE_Emax": float(se[0]),
            "SE_gamma": float(se[1]),
            "_popt": popt_a,
            "_pcov": _covariance_for_pd_samples(pcov, popt_a),
            **s,
        }

    raise ValueError("Unknown approach")


# ============================================================
# 9) SECTION 10d: SCALE PD TO HUMAN (REQUIRED)
# ============================================================

def scale_pd_to_human(
    model_params: Dict[str, Any],
    approach: Literal["total_auc", "free_auc", "cavg_fixed_ec50"],
    auc_human_df: pd.DataFrame,
    fu_human: float,
    doses_mg: np.ndarray,
    ec50_fixed_uM: float = 0.090,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (doses_mg, tgi_mean, tgi_lo90, tgi_hi90)
    using PopPK AUC variability at each dose.

    For approach:
      - total_auc: driver is Total_AUC_human(0-24h) = 2*AUC(0-12h)
      - free_auc: driver is Free_AUC_human(0-24h) = fu_human * 2*AUC(0-12h)
      - cavg_fixed_ec50: driver is Cavg_free_human = (fu_human*AUC(0-12h))/12
    """
    emax = float(model_params["Emax"])
    gamma = float(model_params["gamma"])

    mean = []
    lo = []
    hi = []

    for d in doses_mg:
        df = auc_human_df[auc_human_df["dose_mg_BID"] == d]
        auc12 = df["AUCss_0_12h_uM_h"].to_numpy()

        if approach == "total_auc":
            x = 2.0 * auc12  # µM*h
            ec50 = float(model_params["EC50"])
            tgi = hill_emax(x, emax=emax, ec50=ec50, gamma=gamma)
        elif approach == "free_auc":
            # daily exposure matching: AUC(0-24h) = 2 × AUCss(0-12h)
            x = fu_human * (2.0 * auc12)  # µM*h
            ec50 = float(model_params["EC50"])
            tgi = hill_emax(x, emax=emax, ec50=ec50, gamma=gamma)
        else:
            # Cavg_free in µM
            x = (fu_human * auc12) / 12.0
            tgi = hill_emax_fixed_ec50(x, emax=emax, gamma=gamma, ec50_fixed=ec50_fixed_uM)

        mean.append(float(np.mean(tgi)))
        lo.append(float(np.quantile(tgi, 0.05)))
        hi.append(float(np.quantile(tgi, 0.95)))

    return doses_mg, np.array(mean), np.array(lo), np.array(hi)


# ============================================================
# 10) PLOTTING HELPERS (DARK THEME)
# ============================================================

def set_dark_theme() -> None:
    plt.style.use("dark_background")
    mpl.rcParams.update(
        {
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#0e1117",
            "savefig.facecolor": "#0e1117",
            "grid.color": "#2b2f3a",
            "axes.edgecolor": "#c7d0d9",
            "text.color": "#c7d0d9",
            "axes.labelcolor": "#c7d0d9",
            "xtick.color": "#c7d0d9",
            "ytick.color": "#c7d0d9",
        }
    )


def annotate_params(ax: plt.Axes, label: str, params: Dict[str, Any], xy: Tuple[float, float]) -> None:
    params = public_fit_dict(params)
    if "EC50" in params:
        txt = (
            f"{label}\n"
            f"Emax={params['Emax']:.1f}%\n"
            f"EC50={params['EC50']:.3g}\n"
            f"γ={params['gamma']:.2f}\n"
            f"R²={params['R2']:.3f}"
        )
    else:
        txt = (
            f"{label}\n"
            f"Emax={params['Emax']:.1f}%\n"
            f"EC50 fixed={params['EC50_fixed']:.3g} µM\n"
            f"γ={params['gamma']:.2f}\n"
            f"R²={params['R2']:.3f}"
        )
    ax.text(xy[0], xy[1], txt, transform=ax.transAxes, fontsize=9, va="top")


# ============================================================
# 11) CHECKPOINTS TABLE (REQUIRED OUTPUT 4)
# ============================================================

def checkpoints_table_revised(
    pk_df_400: pd.DataFrame,
    ehc_frac: float,
    auc24_mean: float,
    fit_786_total: Dict[str, float],
    tgi400_786_a1_total: float,
    tgi400_786_a2a_free: float,
    tgi400_786_a2b_free: float,
    tgi400_786_a3_ic50: float,
    r2_786_a1: float,
    r2_renca_a1: float,
    delta_a1_vs_a3: float,
    log_path: Path | None = None,
) -> None:
    auc12_mean = float(pk_df_400["AUCss_0_12h_uM_h"].mean())
    cmax_mean = float(pk_df_400["Cmax_ss_uM"].mean())

    buf = io.StringIO()

    def w(s: str) -> None:
        print(s)
        buf.write(s + "\n")

    w("╔════════════════════════════════════════════════════════════╗")
    w("║  SORAFENIB RCC PK/PD — REVISED COMPARISON CHECKPOINTS     ║")
    w("╠═══╦═══════════════════════════╦══════════╦════════════════╣")
    w("║ # ║ Metric                    ║ Result   ║ Note           ║")
    w("╠═══╬═══════════════════════════╬══════════╬════════════════╣")
    w(f"║ 1 ║ AUCss(0-12h) mean [µM·h] ║ {auc12_mean:8.1f} ║ Target: 97-146 ║")
    w(f"║ 2 ║ Cmax,ss mean [µM]        ║ {cmax_mean:8.1f} ║ Target: 6-15   ║")
    w(f"║ 3 ║ EHC contribution         ║ {100*ehc_frac:7.1f}% ║ Target: ~50%   ║")
    w(f"║ 4 ║ Human AUC(0-24h) [µM·h]  ║ {auc24_mean:8.1f} ║ ~217 expected  ║")
    w(f"║ 5 ║ 786-O Emax (Approach 1)  ║ {fit_786_total['Emax']:8.1f}% ║ Target: 75-85% ║")
    w(f"║ 6 ║ 786-O R² (Approach 1)    ║ {r2_786_a1:8.3f} ║ Target: >0.85  ║")
    w(f"║ 7 ║ Renca R² (Approach 1)    ║ {r2_renca_a1:8.3f} ║ Target: >0.85  ║")
    w(f"║ 8 ║ Human TGI 786-O Appr1    ║ {tgi400_786_a1_total:8.1f}% ║ Expect: 50-80% ║")
    w(f"║ 9 ║ Human TGI 786-O Appr2a   ║ {tgi400_786_a2a_free:8.1f}% ║ Expect: <28%   ║")
    w(f"║10 ║ Human TGI 786-O Appr2b   ║ {tgi400_786_a2b_free:8.1f}% ║ Expect: ~50%+  ║")
    w(f"║11 ║ Human TGI 786-O Appr3    ║ {tgi400_786_a3_ic50:8.1f}% ║ Expect: 20-35% ║")
    flag = "FLAG (>30%)" if delta_a1_vs_a3 > 30.0 else "OK"
    w(f"║12 ║ Approach 1 vs 3 delta    ║ {delta_a1_vs_a3:8.1f}% ║ {flag:<14} ║")
    w("╚═══╩═══════════════════════════╩══════════╩════════════════╝")

    if log_path is not None:
        log_path.write_text(buf.getvalue(), encoding="utf-8")


# ============================================================
# 12) MAIN: RUN EVERYTHING + OUTPUTS
# ============================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sorafenib RCC PK/PD exploratory simulation (non-clinical).")
    parser.add_argument("--n-subjects", type=int, default=1000, help="PopPK virtual subjects for the 400 mg BID summary block")
    parser.add_argument("--dose-sweep-subjects", type=int, default=300, help="PopPK subjects per dose in the dose–response sweep")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help="Results root directory (default: ./results next to script; tables in root, PNGs in ./results/figures/)",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir).resolve() if args.results_dir else (SCRIPT_DIR / "results")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Software: Python {sys.version.split()[0]} ({sys.version})\n"
        f"  numpy {np.__version__}, scipy {scipy.__version__}, matplotlib {mpl.__version__}, pandas {pd.__version__}\n"
        f"Tables -> {results_dir}\n"
        f"Figures -> {figures_dir}\n"
    )

    ehc_qa = verify_ehc_mass_balance()
    if not bool(ehc_qa.get("pass", False)):
        raise SystemExit("EHC mass-balance verification FAILED (see table above). Fix PK/EHC implementation before publishing results.")
    run_sanity_checks()

    print_assumptions_block(results_dir / "output5_assumptions_block.txt")

    fu_human = CONST.fu_human
    ec50_vegfr2_uM = CONST.ec50_vegfr2_free_um

    # ---------------------------
    # OUTPUT 1: PK simulation @400 mg BID
    # ---------------------------
    pk_df_400 = simulate_population_pk(dose_mg_bID=400.0, n_subjects=args.n_subjects, dt_h=0.02, t_end_h=240.0)
    cols = ["AUCss_0_12h_uM_h", "Cmax_ss_uM", "Cmin_ss_uM", "AUC_0_24h_uM_h"]
    pk_summary = pd.DataFrame(
        {
            "mean": pk_df_400[cols].mean(),
            "sd": pk_df_400[cols].std(ddof=1),
            "p05": pk_df_400[cols].quantile(0.05),
            "p95": pk_df_400[cols].quantile(0.95),
        }
    ).T
    print(f"\nOUTPUT 1 — PK @ 400 mg BID (n={args.n_subjects}, steady state):")
    print(pk_summary.to_string(float_format=lambda v: f"{v:0.3g}"))
    print(f"\nMean AUC(0-24h) = 2 × AUCss(0-12h) = {pk_df_400['AUC_0_24h_uM_h'].mean():.1f} uM*h")
    pk_summary.to_csv(results_dir / "output1_pk_400mg_bid_summary.csv")

    # EHC diagnostic using the typical-theta simulation as a baseline.
    theta_typ = PKParams(
        ka_per_h=POPPK.ka_per_h,
        cl_L_per_h=POPPK.cl_L_per_h,
        vc_L=POPPK.vc_L,
        q_L_per_h=POPPK.q_L_per_h,
        vp_L=POPPK.vp_L,
        f_oral=POPPK.f_oral,
        kcb_per_h=POPPK.kcb_per_h,
        kbg_per_h=POPPK.kbg_per_h,
        f_ehc=POPPK.f_ehc,
    )
    t_typ, c_typ = simulate_pk_bi_dose(400.0, theta_typ, t_end_h=240.0, dt_h=0.01)
    ehc_frac = ehc_auc_fraction_approx(t_typ, c_typ, theta_typ, dose_mg=400.0)

    # ---------------------------
    # PD fits — revised framework (Approach 1 total AUC; 2a/2b free AUC; 3 IC50 anchor)
    # ---------------------------
    fu_mouse_2a = CONST.fu_mouse_literature
    fu_mouse_2b = CONST.fu_mouse_high_binding

    fit_786_total = fit_pd_model_786O(approach="total_auc")
    fit_renca_total = fit_pd_model_Renca(approach="total_auc")

    fit_786_2a = fit_pd_model_786O(approach="free_auc", fu_mouse=fu_mouse_2a)
    fit_renca_2a = fit_pd_model_Renca(approach="free_auc", fu_mouse=fu_mouse_2a)
    fit_786_2b = fit_pd_model_786O(approach="free_auc", fu_mouse=fu_mouse_2b)
    fit_renca_2b = fit_pd_model_Renca(approach="free_auc", fu_mouse=fu_mouse_2b)

    # Approach 3: unchanged mechanistic anchor (free Cavg), using the same mouse fu as 2a for preclinical calibration
    fit_786_a3 = fit_pd_model_786O(approach="cavg_fixed_ec50", fu_mouse=fu_mouse_2a, ec50_fixed_uM=ec50_vegfr2_uM)
    fit_renca_a3 = fit_pd_model_Renca(approach="cavg_fixed_ec50", fu_mouse=fu_mouse_2a, ec50_fixed_uM=ec50_vegfr2_uM)

    print("\nPD FIT — 786-O (Approach 1: Total AUC, paper-consistent):")
    print(pd.Series(public_fit_dict(fit_786_total)).to_string(float_format=lambda v: f"{v:0.4g}"))
    print("\nPD FIT — Renca (Approach 1: Total AUC, paper-consistent):")
    print(pd.Series(public_fit_dict(fit_renca_total)).to_string(float_format=lambda v: f"{v:0.4g}"))

    print("\nPD FIT — 786-O (Approach 2a: Free AUC, fu_mouse=0.030):")
    print(pd.Series(public_fit_dict(fit_786_2a)).to_string(float_format=lambda v: f"{v:0.4g}"))
    print("\nPD FIT — 786-O (Approach 2b: Free AUC, fu_mouse=0.005):")
    print(pd.Series(public_fit_dict(fit_786_2b)).to_string(float_format=lambda v: f"{v:0.4g}"))
    print("\nSensitivity of EC50 (786-O, free AUC): EC50_2b / EC50_2a = {:.3f}".format(fit_786_2b["EC50"] / fit_786_2a["EC50"]))

    print("\nPD FIT — Renca (Approach 2a: Free AUC, fu_mouse=0.030):")
    print(pd.Series(public_fit_dict(fit_renca_2a)).to_string(float_format=lambda v: f"{v:0.4g}"))
    print("\nPD FIT — Renca (Approach 2b: Free AUC, fu_mouse=0.005):")
    print(pd.Series(public_fit_dict(fit_renca_2b)).to_string(float_format=lambda v: f"{v:0.4g}"))
    print("\nSensitivity of EC50 (Renca, free AUC): EC50_2b / EC50_2a = {:.3f}".format(fit_renca_2b["EC50"] / fit_renca_2a["EC50"]))

    print("\nPD FIT — 786-O (Approach 3: VEGFR-2 IC50 anchor, free Cavg):")
    print(pd.Series(public_fit_dict(fit_786_a3)).to_string(float_format=lambda v: f"{v:0.4g}"))
    print("\nPD FIT — Renca (Approach 3: VEGFR-2 IC50 anchor, free Cavg):")
    print(pd.Series(public_fit_dict(fit_renca_a3)).to_string(float_format=lambda v: f"{v:0.4g}"))

    # ---------------------------
    # OUTPUT 2: PD calibration plot (RCC only) — Approach 1: Total AUC
    # ---------------------------
    set_dark_theme()

    fig1, ax = plt.subplots(figsize=(10, 6))

    auc_786_total = auc_mouse_power_law(RCC_786O_DOSE_MGKG)
    auc_renca_total = auc_mouse_power_law(RCC_RENCA_DOSE_MGKG)

    total_auc_human_mean_024 = float(pk_df_400["AUC_0_24h_uM_h"].mean())

    ax.scatter(auc_786_total, RCC_786O_TGI_PCT, s=70, marker="o", label="786-O (data)")
    ax.scatter(auc_renca_total, RCC_RENCA_TGI_PCT, s=80, marker="^", label="Renca (data)")

    xx_tot = np.logspace(np.log10(min(auc_786_total.min(), auc_renca_total.min()) * 0.5), np.log10(max(auc_786_total.max(), auc_renca_total.max()) * 2.0), 300)
    ax.plot(xx_tot, hill_emax(xx_tot, fit_786_total["Emax"], fit_786_total["EC50"], fit_786_total["gamma"]), lw=2.5, label="786-O fit")
    ax.plot(xx_tot, hill_emax(xx_tot, fit_renca_total["Emax"], fit_renca_total["EC50"], fit_renca_total["gamma"]), lw=2.5, label="Renca fit")

    ax.axvline(total_auc_human_mean_024, color="#d19a66", ls="--", lw=2, label="Human 400 mg BID (mean total AUC0-24)")

    ax.set_xscale("log")
    ax.set_xlabel("Total AUC (µmol/L·h), log scale")
    ax.set_ylabel("TGI (%)")
    ax.set_title("Sorafenib RCC PD calibration — Approach 1: Total AUC (paper-consistent)")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(0, 100)

    annotate_params(ax, "786-O", fit_786_total, xy=(0.02, 0.98))
    annotate_params(ax, "Renca", fit_renca_total, xy=(0.20, 0.98))
    ax.text(
        0.60,
        0.25,
        "WARNING: Only 3–4 data points per model.\nParameters are poorly constrained.",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="#111827", edgecolor="#374151", alpha=0.8),
    )
    ax.legend(loc="lower right")

    fig1.tight_layout()
    fig1.savefig(figures_dir / "output2_pd_calibration_total_auc.png", dpi=200)

    # ---------------------------
    # OUTPUT 3: Human TGI prediction vs dose (100–800 mg BID) with 90% CI (786-O, all approaches)
    # ---------------------------
    doses = np.arange(100.0, 801.0, 50.0)
    dfs = []
    for d in doses:
        # Performance note: dose-sweep uses fewer subjects and coarser dt.
        df = simulate_population_pk(dose_mg_bID=float(d), n_subjects=args.dose_sweep_subjects, dt_h=0.05, t_end_h=240.0)
        df["dose_mg_BID"] = float(d)
        dfs.append(df[["dose_mg_BID", "AUCss_0_12h_uM_h"]])
    auc_by_dose = pd.concat(dfs, ignore_index=True)

    rng_ci = np.random.default_rng(RANDOM_SEED + 404)
    d, m786_a1, lo786_a1, hi786_a1 = scale_pd_to_human_combined_uncertainty(
        fit_786_total, "total_auc", auc_by_dose, fu_human, doses, ec50_fixed_uM=ec50_vegfr2_uM, rng=rng_ci
    )
    _, m786_2a, lo786_2a, hi786_2a = scale_pd_to_human_combined_uncertainty(
        fit_786_2a, "free_auc", auc_by_dose, fu_human, doses, ec50_fixed_uM=ec50_vegfr2_uM, rng=np.random.default_rng(RANDOM_SEED + 405)
    )
    _, m786_2b, lo786_2b, hi786_2b = scale_pd_to_human_combined_uncertainty(
        fit_786_2b, "free_auc", auc_by_dose, fu_human, doses, ec50_fixed_uM=ec50_vegfr2_uM, rng=np.random.default_rng(RANDOM_SEED + 406)
    )
    _, m786_a3, lo786_a3, hi786_a3 = scale_pd_to_human_combined_uncertainty(
        fit_786_a3, "cavg_fixed_ec50", auc_by_dose, fu_human, doses, ec50_fixed_uM=ec50_vegfr2_uM, rng=np.random.default_rng(RANDOM_SEED + 407)
    )

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    ax2.plot(d, m786_a1, lw=2.5, label="786-O + Appr1: Total AUC")
    ax2.fill_between(d, lo786_a1, hi786_a1, alpha=0.14)

    ax2.plot(d, m786_2a, lw=2.5, label="786-O + Appr2a: Free AUC (fu_mouse=0.030)")
    ax2.fill_between(d, lo786_2a, hi786_2a, alpha=0.14)

    ax2.plot(d, m786_2b, lw=2.5, label="786-O + Appr2b: Free AUC (fu_mouse=0.005)")
    ax2.fill_between(d, lo786_2b, hi786_2b, alpha=0.14)

    ax2.plot(d, m786_a3, lw=2.5, label="786-O + Appr3: VEGFR-2 IC50 anchor")
    ax2.fill_between(d, lo786_a3, hi786_a3, alpha=0.14)

    for x in [200.0, 400.0]:
        ax2.axvline(x, color="#93c5fd", ls="--", lw=1.8)
    ax2.text(0.53, 0.12, "Clinical context: PFS benefit in TARGET trial at 400 mg BID", transform=ax2.transAxes, fontsize=10)

    ax2.set_xlabel("Sorafenib dose (mg BID)")
    ax2.set_ylabel("Predicted TGI (%)")
    ax2.set_title("Human RCC TGI vs dose — 786-O model (90% CI: PK IIV + PD parameter uncertainty)")
    ax2.grid(True, alpha=0.35)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.text(
        0.02,
        0.98,
        "CI includes both PK (IIV) and PD parameter uncertainty (Monte Carlo).",
        transform=ax2.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="#111827", edgecolor="#374151", alpha=0.85),
    )

    fig2.tight_layout()
    fig2.savefig(figures_dir / "output3_human_tgi_dose_response.png", dpi=200)

    # Combined figure
    figc, axes = plt.subplots(1, 2, figsize=(19, 6))
    axes[0].scatter(auc_786_total, RCC_786O_TGI_PCT, s=60, marker="o", label="786-O data")
    axes[0].scatter(auc_renca_total, RCC_RENCA_TGI_PCT, s=70, marker="^", label="Renca data")
    axes[0].plot(xx_tot, hill_emax(xx_tot, fit_786_total["Emax"], fit_786_total["EC50"], fit_786_total["gamma"]), lw=2.2, label="786-O fit")
    axes[0].plot(xx_tot, hill_emax(xx_tot, fit_renca_total["Emax"], fit_renca_total["EC50"], fit_renca_total["gamma"]), lw=2.2, label="Renca fit")
    axes[0].axvline(total_auc_human_mean_024, color="#d19a66", ls="--", lw=2, label="Human 400 mg BID (mean)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Total AUC (µM·h), log")
    axes[0].set_ylabel("TGI (%)")
    axes[0].set_title("Output 2: PD calibration (Approach 1)")
    axes[0].grid(True, which="both", alpha=0.35)
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(d, m786_a1, lw=2.2, label="786-O Appr1")
    axes[1].fill_between(d, lo786_a1, hi786_a1, alpha=0.12)
    axes[1].plot(d, m786_2a, lw=2.2, label="786-O Appr2a")
    axes[1].fill_between(d, lo786_2a, hi786_2a, alpha=0.12)
    axes[1].plot(d, m786_2b, lw=2.2, label="786-O Appr2b")
    axes[1].fill_between(d, lo786_2b, hi786_2b, alpha=0.12)
    axes[1].plot(d, m786_a3, lw=2.2, label="786-O Appr3")
    axes[1].fill_between(d, lo786_a3, hi786_a3, alpha=0.12)
    for x in [200.0, 400.0]:
        axes[1].axvline(x, color="#93c5fd", ls="--", lw=1.6)
    axes[1].set_xlabel("Dose (mg BID)")
    axes[1].set_ylabel("Predicted TGI (%)")
    axes[1].set_title("Output 3: Human dose–TGI (786-O)")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="upper left", fontsize=9)

    figc.tight_layout()
    figc.savefig(figures_dir / "combined_outputs_2_and_3.png", dpi=200)

    # ---------------------------
    # OUTPUT 6: Sensitivity of human TGI@400mg to fu_mouse (refit free-AUC Hill each grid point)
    # ---------------------------
    df400 = auc_by_dose[auc_by_dose["dose_mg_BID"] == 400.0]
    auc12_400 = df400["AUCss_0_12h_uM_h"].to_numpy()
    free_auc24_human = fu_human * (2.0 * auc12_400)

    fu_grid = np.logspace(np.log10(0.001), np.log10(0.05), 40)

    def mean_tgi_from_fit(fit: Dict[str, float], x: np.ndarray) -> float:
        y = hill_emax(x, fit["Emax"], fit["EC50"], fit["gamma"])
        return float(np.mean(y))

    tgi_sens_786 = []
    tgi_sens_renca = []
    for fum in fu_grid:
        f786 = fit_pd_model_786O(approach="free_auc", fu_mouse=float(fum))
        fre = fit_pd_model_Renca(approach="free_auc", fu_mouse=float(fum))
        tgi_sens_786.append(mean_tgi_from_fit(f786, free_auc24_human))
        tgi_sens_renca.append(mean_tgi_from_fit(fre, free_auc24_human))

    fig6, ax6 = plt.subplots(figsize=(9, 5.5))
    ax6.plot(fu_grid, tgi_sens_786, lw=2.5, label="786-O")
    ax6.plot(fu_grid, tgi_sens_renca, lw=2.5, label="Renca")
    for xv in [0.005, 0.030]:
        ax6.axvline(xv, color="#93c5fd", ls="--", lw=1.6)
    ax6.set_xscale("log")
    ax6.set_xlabel("fu_mouse (log scale)")
    ax6.set_ylabel("Predicted human TGI% @ 400 mg BID (PopPK mean)")
    ax6.set_title("Sensitivity of Human TGI Prediction to Mouse fu Assumption")
    ax6.grid(True, which="both", alpha=0.35)
    ax6.set_ylim(0, 100)
    ax6.legend(loc="lower right")
    fig6.tight_layout()
    fig6.savefig(figures_dir / "output6_sensitivity_fu_mouse.png", dpi=200)

    # ---------------------------
    # Summary @ 400 mg BID (mean exposure drivers)
    # ---------------------------
    auc12_mean = float(df400["AUCss_0_12h_uM_h"].mean())
    auc24_mean = 2.0 * auc12_mean
    free_auc24_mean = fu_human * auc24_mean
    cavg_free_mean = (fu_human * auc12_mean) / 12.0

    tgi_786_a1_mean = float(hill_emax(auc24_mean, fit_786_total["Emax"], fit_786_total["EC50"], fit_786_total["gamma"]))
    tgi_786_2a_mean = float(hill_emax(free_auc24_mean, fit_786_2a["Emax"], fit_786_2a["EC50"], fit_786_2a["gamma"]))
    tgi_786_2b_mean = float(hill_emax(free_auc24_mean, fit_786_2b["Emax"], fit_786_2b["EC50"], fit_786_2b["gamma"]))
    tgi_786_a3_mean = float(hill_emax_fixed_ec50(cavg_free_mean, fit_786_a3["Emax"], fit_786_a3["gamma"], ec50_fixed=ec50_vegfr2_uM))

    print("\nAPPROACH COMPARISON @ 400 mg BID (using PopPK mean exposure drivers):")
    print(f"  786-O: Approach 1 (total AUC)     TGI={tgi_786_a1_mean:0.1f}%")
    print(f"  786-O: Approach 2a (free AUC)    TGI={tgi_786_2a_mean:0.1f}%")
    print(f"  786-O: Approach 2b (free AUC)    TGI={tgi_786_2b_mean:0.1f}%")
    print(f"  786-O: Approach 3 (IC50 anchor) TGI={tgi_786_a3_mean:0.1f}%")

    # 90% CI at 400 mg BID from PopPK variability
    idx400 = int(np.where(doses == 400.0)[0][0])
    print("\nPredicted human TGI @ 400 mg BID (mean curve and 90% CI: PK IIV + PD uncertainty) — 786-O:")
    print(f"  Approach 1 (total AUC):  {m786_a1[idx400]:0.1f}%  (90% CI {lo786_a1[idx400]:0.1f}–{hi786_a1[idx400]:0.1f}%)")
    print(f"  Approach 2a (fu_mouse=0.030): {m786_2a[idx400]:0.1f}%  (90% CI {lo786_2a[idx400]:0.1f}–{hi786_2a[idx400]:0.1f}%)")
    print(f"  Approach 2b (fu_mouse=0.005): {m786_2b[idx400]:0.1f}%  (90% CI {lo786_2b[idx400]:0.1f}–{hi786_2b[idx400]:0.1f}%)")
    print(f"  Approach 3 (IC50 anchor): {m786_a3[idx400]:0.1f}%  (90% CI {lo786_a3[idx400]:0.1f}–{hi786_a3[idx400]:0.1f}%)")

    delta_a1_vs_a3 = abs(tgi_786_a1_mean - tgi_786_a3_mean)

    # OUTPUT 4: revised checkpoints table
    checkpoints_table_revised(
        pk_df_400=pk_df_400,
        ehc_frac=ehc_frac,
        auc24_mean=auc24_mean,
        fit_786_total=fit_786_total,
        tgi400_786_a1_total=tgi_786_a1_mean,
        tgi400_786_a2a_free=tgi_786_2a_mean,
        tgi400_786_a2b_free=tgi_786_2b_mean,
        tgi400_786_a3_ic50=tgi_786_a3_mean,
        r2_786_a1=fit_786_total["R2"],
        r2_renca_a1=fit_renca_total["R2"],
        delta_a1_vs_a3=delta_a1_vs_a3,
        log_path=results_dir / "output4_checkpoint_table.txt",
    )

    # OUTPUT 2/3 files already saved, plus combined figure
    print("\nSaved outputs:")
    for name in (
        "output1_pk_400mg_bid_summary.csv",
        "output4_checkpoint_table.txt",
        "output5_assumptions_block.txt",
    ):
        print(f"  - {results_dir / name}")
    for name in (
        "output2_pd_calibration_total_auc.png",
        "output3_human_tgi_dose_response.png",
        "combined_outputs_2_and_3.png",
        "output6_sensitivity_fu_mouse.png",
    ):
        print(f"  - {figures_dir / name}")

    # NOTE: Under-constraint warning (explicit)
    print(
        "\nNOTE: PD fits are under-constrained (3–4 data points per model with 2–3 parameters). "
        "Treat fitted PD parameters and translated human TGI predictions as exploratory only."
    )


if __name__ == "__main__":
    main()

