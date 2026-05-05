"""Pazopanib PK/PD simulator for MIPD use-cases (Python-only).

Model blocks included:

- PK: one-compartment oral model with first-order absorption and dose-dependent F1.
- Toxicity: Gompertz time-to-event hazard linked to Cmin,ss threshold.
- Efficacy: semimechanistic SLD dynamics for mRCC and STS with resistance terms.

This module is designed for simulation and dose comparison workflows; it is not a
NONMEM replacement. Parameters can be refined as final estimates become available.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

Indication = Literal["mRCC", "STS"]


@dataclass
class PazopanibPKParams:
    ka_h_inv: float = 0.976
    cl_f_L_h: float = 0.497
    v_f_L: float = 46.1
    f1_ref: float = 1.0
    f1_ref_dose_mg: float = 200.0
    f1_power: float = 0.42


@dataclass
class ResidualError:
    proportional_cv: float = 0.24
    additive_mg_L: float = 4.71


@dataclass
class IIVParams:
    cl_cv: float = 0.0
    v_cv: float = 0.88
    f1_cv: float = 0.36


@dataclass
class QDRegimen:
    dose_mg: float = 600.0
    interval_h: float = 24.0


@dataclass
class ToxicityParams:
    threshold_cmin_mg_L: float = 34.0
    hazard_ratio_above_threshold: float = 3.35
    gompertz_h0_day_inv: float = 0.0021
    gompertz_gamma_day_inv: float = -0.012


@dataclass
class TumorDynamicsParams:
    kg_day_inv: float
    kd_day_inv: float
    lambda_day_inv: float
    primary_resistance_fraction: float
    efficacy_threshold_cmin_mg_L: float = 20.5
    low_exposure_kd_scale: float = 0.75


@dataclass
class PazopanibPKPDConfig:
    pk: PazopanibPKParams = field(default_factory=PazopanibPKParams)
    iiv: IIVParams = field(default_factory=IIVParams)
    residual: ResidualError = field(default_factory=ResidualError)
    regimen: QDRegimen = field(default_factory=QDRegimen)
    toxicity: ToxicityParams = field(default_factory=ToxicityParams)
    tumor_mrcc: TumorDynamicsParams = field(
        default_factory=lambda: TumorDynamicsParams(
            kg_day_inv=0.0005,
            kd_day_inv=0.0040,
            lambda_day_inv=0.0080,
            primary_resistance_fraction=0.27,
        )
    )
    tumor_sts: TumorDynamicsParams = field(
        default_factory=lambda: TumorDynamicsParams(
            kg_day_inv=0.0086,
            kd_day_inv=0.0080,
            lambda_day_inv=0.0003,
            primary_resistance_fraction=0.13,
        )
    )


def _lognormal_sample(mean: float, cv: float, rng: np.random.Generator) -> float:
    if cv <= 0.0:
        return float(mean)
    sigma2 = np.log(1.0 + cv * cv)
    sigma = np.sqrt(sigma2)
    mu = np.log(max(mean, 1e-12)) - 0.5 * sigma2
    return float(rng.lognormal(mean=mu, sigma=sigma))


def sample_individual_config(base: PazopanibPKPDConfig, rng: np.random.Generator) -> PazopanibPKPDConfig:
    """Sample one individual PK profile with IIV on V/F and F1 scalar."""
    cl = base.pk.cl_f_L_h
    v = _lognormal_sample(base.pk.v_f_L, base.iiv.v_cv, rng)
    f1_scalar = _lognormal_sample(base.pk.f1_ref, base.iiv.f1_cv, rng)
    f1_scalar = max(f1_scalar, 1e-6)
    pk = replace(base.pk, cl_f_L_h=cl, v_f_L=v, f1_ref=f1_scalar)
    return replace(base, pk=pk)


def dose_dependent_f1(dose_mg: float, pk: PazopanibPKParams) -> float:
    """NONMEM-like FCOV: F1 = F1_ref * (dose_ref / dose)^power."""
    dose = max(dose_mg, 1e-9)
    ref_ratio = max(pk.f1_ref_dose_mg, 1e-9) / dose
    return float(max(pk.f1_ref * (ref_ratio**pk.f1_power), 1e-8))


def build_qd_dose_times(t_end_h: float, interval_h: float = 24.0) -> list[float]:
    n_doses = int(np.floor(max(t_end_h, 0.0) / interval_h)) + 1
    return [float(k * interval_h) for k in range(n_doses)]


def _pk_rhs(_: float, y: np.ndarray, ka_h_inv: float, ke_h_inv: float) -> np.ndarray:
    a_gut, a_cent = y
    return np.array([-ka_h_inv * a_gut, ka_h_inv * a_gut - ke_h_inv * a_cent], dtype=float)


def simulate_pazopanib_pk(
    cfg: PazopanibPKPDConfig,
    *,
    t_end_days: float = 365.0,
    n_eval: int = 2001,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Simulate oral one-compartment PK under QD dosing."""
    t_end_h = float(t_end_days * 24.0)
    t_grid = np.linspace(0.0, t_end_h, int(n_eval))
    ke = cfg.pk.cl_f_L_h / max(cfg.pk.v_f_L, 1e-12)
    dts = build_qd_dose_times(t_end_h, cfg.regimen.interval_h)
    knots = sorted({0.0, t_end_h, *dts})
    y = np.zeros(2, dtype=float)
    sol_t: list[float] = []
    sol_y: list[np.ndarray] = []
    eps = 1e-10
    f1 = dose_dependent_f1(cfg.regimen.dose_mg, cfg.pk)
    dose_to_gut = cfg.regimen.dose_mg * f1

    def rhs(tt: float, yy: np.ndarray) -> np.ndarray:
        return _pk_rhs(tt, yy, cfg.pk.ka_h_inv, ke)

    for i in range(len(knots) - 1):
        t0, t1 = float(knots[i]), float(knots[i + 1])
        if t1 <= t0 + eps:
            continue
        for td in dts:
            if abs(td - t0) <= eps:
                y[0] += dose_to_gut

        sub_t = t_grid[(t_grid >= t0 - 1e-11) & (t_grid <= t1 + 1e-11)]
        if sub_t.size > 0:
            sub_t = sub_t[(sub_t >= t0) & (sub_t <= t1)]
        if sol_t and sub_t.size > 0 and abs(float(sub_t[0]) - sol_t[-1]) < 1e-8:
            sub_t = sub_t[1:]

        if sub_t.size == 0:
            iv = solve_ivp(rhs, (t0, t1), y, rtol=rtol, atol=atol, method="RK45")
            if not iv.success:
                raise RuntimeError(iv.message)
            y = iv.y[:, -1].copy()
            continue

        iv = solve_ivp(rhs, (t0, t1), y, t_eval=sub_t, rtol=rtol, atol=atol, method="RK45")
        if not iv.success:
            raise RuntimeError(iv.message)
        for ti, yi in zip(iv.t, iv.y.T, strict=True):
            sol_t.append(float(ti))
            sol_y.append(np.asarray(yi, dtype=float))
        y = iv.y[:, -1].copy()

    if not sol_y:
        raise RuntimeError("No samples generated. Increase t_end_days or n_eval.")
    Y = np.vstack(sol_y)
    t_h = np.asarray(sol_t, dtype=float)
    a_gut = Y[:, 0]
    a_cent = Y[:, 1]
    c_mg_L = a_cent / max(cfg.pk.v_f_L, 1e-12)
    return {
        "t_h": t_h,
        "A_gut_mg": a_gut,
        "A_cent_mg": a_cent,
        "C_mg_L": c_mg_L,
    }


def daily_cmin_from_profile(t_h: np.ndarray, c_mg_L: np.ndarray, *, horizon_days: int) -> np.ndarray:
    """Compute daily trough-like Cmin using the last 24h window for each day."""
    t_h = np.asarray(t_h, dtype=float)
    c_mg_L = np.asarray(c_mg_L, dtype=float)
    out = np.zeros(int(horizon_days), dtype=float)
    for d in range(1, horizon_days + 1):
        lo, hi = (d - 1) * 24.0, d * 24.0
        m = (t_h >= lo) & (t_h <= hi + 1e-9)
        out[d - 1] = float(np.min(c_mg_L[m])) if np.any(m) else float("nan")
    return out


def estimate_cmin_ss(cmin_daily: np.ndarray, window_days: int = 14) -> float:
    x = np.asarray(cmin_daily, dtype=float)
    n = min(int(window_days), x.size)
    if n <= 0:
        return float("nan")
    return float(np.nanmedian(x[-n:]))


def liver_toxicity_probability(cmin_daily: np.ndarray, tox: ToxicityParams) -> np.ndarray:
    """Cumulative probability of CTCAE>=2 liver toxicity by day."""
    cmin_daily = np.asarray(cmin_daily, dtype=float)
    hr_log = np.log(max(tox.hazard_ratio_above_threshold, 1e-8))
    cum_hazard = 0.0
    probs = np.zeros_like(cmin_daily, dtype=float)
    for i, c in enumerate(cmin_daily, start=1):
        day = float(i)
        above = 1.0 if c > tox.threshold_cmin_mg_L else 0.0
        hz = tox.gompertz_h0_day_inv * np.exp(tox.gompertz_gamma_day_inv * day + hr_log * above)
        cum_hazard += hz
        probs[i - 1] = 1.0 - np.exp(-cum_hazard)
    return probs


def simulate_tumor_sld(
    cmin_daily: np.ndarray,
    tumor: TumorDynamicsParams,
    *,
    baseline_sld_mm: float = 100.0,
) -> np.ndarray:
    """Daily SLD semimechanistic trajectory with acquired and primary resistance."""
    cmin_daily = np.asarray(cmin_daily, dtype=float)
    sld = np.zeros_like(cmin_daily, dtype=float)
    s = float(max(baseline_sld_mm, 1e-9))
    for i, c in enumerate(cmin_daily, start=1):
        t = float(i)
        exposure_effect = 1.0 if c >= tumor.efficacy_threshold_cmin_mg_L else tumor.low_exposure_kd_scale
        kd_eff = tumor.kd_day_inv * np.exp(-tumor.lambda_day_inv * t) * (1.0 - tumor.primary_resistance_fraction) * exposure_effect
        growth = tumor.kg_day_inv * s
        shrink = kd_eff * s
        s = max(s + growth - shrink, 1e-9)
        sld[i - 1] = s
    return sld


def simulate_pazopanib_pkpd(
    cfg: PazopanibPKPDConfig,
    *,
    t_end_days: int = 365,
    n_eval: int = 3001,
    baseline_sld_mm: float = 100.0,
) -> dict[str, np.ndarray]:
    """Run PK + toxicity risk + mRCC and STS tumor dynamics for one virtual patient."""
    pk = simulate_pazopanib_pk(cfg, t_end_days=float(t_end_days), n_eval=n_eval)
    cmin_daily = daily_cmin_from_profile(pk["t_h"], pk["C_mg_L"], horizon_days=int(t_end_days))
    tox_p = liver_toxicity_probability(cmin_daily, cfg.toxicity)
    sld_mrcc = simulate_tumor_sld(cmin_daily, cfg.tumor_mrcc, baseline_sld_mm=baseline_sld_mm)
    sld_sts = simulate_tumor_sld(cmin_daily, cfg.tumor_sts, baseline_sld_mm=baseline_sld_mm)
    return {
        **pk,
        "day": np.arange(1, int(t_end_days) + 1, dtype=int),
        "Cmin_daily_mg_L": cmin_daily,
        "Cmin_ss_mg_L": np.full(int(t_end_days), estimate_cmin_ss(cmin_daily), dtype=float),
        "toxicity_prob_day": tox_p,
        "SLD_mRCC_mm": sld_mrcc,
        "SLD_STS_mm": sld_sts,
    }


def simulate_population_dose_scenario(
    base_cfg: PazopanibPKPDConfig,
    *,
    dose_mg: float,
    n_patients: int = 200,
    t_end_days: int = 365,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Population simulation for one dose with IIV."""
    rng = np.random.default_rng(seed)
    cmin_curves: list[np.ndarray] = []
    tox_curves: list[np.ndarray] = []
    sld_mrcc_curves: list[np.ndarray] = []
    sld_sts_curves: list[np.ndarray] = []
    for _ in range(int(n_patients)):
        ind = sample_individual_config(replace(base_cfg, regimen=replace(base_cfg.regimen, dose_mg=float(dose_mg))), rng)
        out = simulate_pazopanib_pkpd(ind, t_end_days=t_end_days)
        cmin_curves.append(out["Cmin_daily_mg_L"])
        tox_curves.append(out["toxicity_prob_day"])
        sld_mrcc_curves.append(out["SLD_mRCC_mm"])
        sld_sts_curves.append(out["SLD_STS_mm"])

    C = np.vstack(cmin_curves)
    T = np.vstack(tox_curves)
    M = np.vstack(sld_mrcc_curves)
    S = np.vstack(sld_sts_curves)
    return {
        "dose_mg": np.array([dose_mg], dtype=float),
        "day": np.arange(1, t_end_days + 1, dtype=int),
        "cmin_p05": np.percentile(C, 5, axis=0),
        "cmin_p50": np.percentile(C, 50, axis=0),
        "cmin_p95": np.percentile(C, 95, axis=0),
        "tox_p50": np.percentile(T, 50, axis=0),
        "sld_mrcc_p50": np.percentile(M, 50, axis=0),
        "sld_sts_p50": np.percentile(S, 50, axis=0),
        "pct_cmin_ge_20_5": np.array([100.0 * np.mean(C[:, -1] >= 20.5)], dtype=float),
        "pct_cmin_gt_34": np.array([100.0 * np.mean(C[:, -1] > 34.0)], dtype=float),
    }
