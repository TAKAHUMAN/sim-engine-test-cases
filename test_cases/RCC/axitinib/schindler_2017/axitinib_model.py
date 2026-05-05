"""Axitinib PK → AUCdaily → biomarkers / dBP / SLD (literature-style mRCC model).

Two-pass integration: (1) oral 2C PK with absorption lag and BID dosing;
(2) rolling 24 h AUC in µg·h/L; (3) coupled biomarker + dBP + tumor ODEs driven by AUC(t).

Units: PK amounts mg, volumes L, time h; concentrations mg/L; AUCdaily µg·h/L;
biomarkers pg/mL; dBP mmHg; SLD mm.

Equations follow the agent specification (Rini 2013 PK; Schindler 2017 PD/TGI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_ivp


def _trapz_yx(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

FoodState = Literal["fed", "fasted"]
Formulation = Literal["IV_form", "XLI"]


@dataclass
class AxitinibPKParams:
    CL: float = 14.6  # L/h
    Vc: float = 47.3  # L
    Q: float = 4.0  # L/h
    Vp: float = 393.0  # L
    ka_fed: float = 0.482  # 1/h
    ka_fasted: float = 1.43  # 1/h
    tlag: float = 0.454  # h
    F_fed_IV: float = 0.457
    F_fasted_IV: float = 0.608
    F_XLI: float = 0.402


@dataclass
class AxitinibCovariates:
    """Multiplicative factors on population CL (linear %-change approx where noted)."""

    age_over_60: bool = False
    japanese: bool = False
    smoker: bool = False
    weight_kg: float = 74.0
    weight_ref_kg: float = 74.0
    Vc_weight_exponent: float = 0.778


@dataclass
class AxitinibRegimen:
    dose_mg: float = 5.0
    interval_h: float = 12.0
    fed_state: FoodState = "fasted"
    formulation: Formulation = "IV_form"


@dataclass
class BiomarkerBlock:
    baseline_pg_ml: float
    MRT_days: float
    AUC50: float
    Imax: float = 1.0
    gamma: float = 1.0


@dataclass
class AxitinibBiomarkerParams:
    VEGF: BiomarkerBlock = field(
        default_factory=lambda: BiomarkerBlock(65.0, 0.722, 354.0, 1.0, 1.0)
    )
    sVEGFR1: BiomarkerBlock = field(
        default_factory=lambda: BiomarkerBlock(83.5, 0.624, 1380.0, 1.0, 1.0)
    )
    sVEGFR2: BiomarkerBlock = field(
        default_factory=lambda: BiomarkerBlock(8850.0, 19.7, 717.0, 1.0, 0.733)
    )
    sVEGFR3: BiomarkerBlock = field(
        default_factory=lambda: BiomarkerBlock(19500.0, 5.76, 717.0, 1.0, 1.0)
    )
    VEGF_disease_progression_per_year: float = 0.65


@dataclass
class DBPParams:
    baseline_mmHg: float = 78.9
    MRT_days: float = 4.92
    Emax: float = 0.197
    S0: float = 0.00127


@dataclass
class SLDParams:
    KG_per_week: float = 0.00361
    ksVEGFR3_per_week: float = -0.174
    lambda_per_week: float = 0.101
    baseline_mm: float = 65.0


@dataclass
class AxitinibChainConfig:
    pk: AxitinibPKParams = field(default_factory=AxitinibPKParams)
    covariates: AxitinibCovariates = field(default_factory=AxitinibCovariates)
    regimen: AxitinibRegimen = field(default_factory=AxitinibRegimen)
    biomarkers: AxitinibBiomarkerParams = field(default_factory=AxitinibBiomarkerParams)
    dbp: DBPParams = field(default_factory=DBPParams)
    sld: SLDParams = field(default_factory=SLDParams)


def _effective_cl_vc(pk: AxitinibPKParams, cov: AxitinibCovariates) -> tuple[float, float]:
    cl = pk.CL
    if cov.age_over_60:
        cl *= 0.787
    if cov.japanese:
        cl *= 0.751
    if cov.smoker:
        cl *= 2.02
    vc = pk.Vc * (cov.weight_kg / max(cov.weight_ref_kg, 1e-9)) ** cov.Vc_weight_exponent
    return float(cl), float(vc)


def _ka_F(pk: AxitinibPKParams, reg: AxitinibRegimen) -> tuple[float, float]:
    if reg.formulation == "XLI":
        return pk.ka_fed, pk.F_XLI
    if reg.fed_state == "fasted":
        return pk.ka_fasted, pk.F_fasted_IV
    return pk.ka_fed, pk.F_fed_IV


def build_bid_dose_times(t_end_h: float, dose_mg: float, interval_h: float) -> tuple[list[float], list[float]]:
    times: list[float] = []
    doses: list[float] = []
    t = 0.0
    while t < t_end_h - 1e-12:
        times.append(float(t))
        doses.append(float(dose_mg))
        t += interval_h
    return times, doses


def pk_rhs(
    _: float,
    y: np.ndarray,
    *,
    ka: float,
    CL: float,
    Vc: float,
    Q: float,
    Vp: float,
) -> np.ndarray:
    G, A1, A2 = y
    dG = -ka * G
    dA1 = ka * G + Q * (A2 / Vp - A1 / Vc) - CL * A1 / Vc
    dA2 = Q * (A1 / Vc - A2 / Vp)
    return np.array([dG, dA1, dA2], dtype=float)


def simulate_axitinib_pk(
    cfg: AxitinibChainConfig,
    *,
    t_end_h: float = 24.0 * 56,
    n_eval: int = 2001,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Oral 2C PK with lag; BID dosing from ``cfg.regimen``."""
    pk = cfg.pk
    reg = cfg.regimen
    CL, Vc = _effective_cl_vc(pk, cfg.covariates)
    ka, bioF = _ka_F(pk, reg)
    dose_times, dose_amounts = build_bid_dose_times(t_end_h, reg.dose_mg, reg.interval_h)
    eff_times = [float(td + pk.tlag) for td in dose_times]

    knots = sorted({0.0, t_end_h, *[t for t in eff_times if t <= t_end_h]})
    t_grid = np.linspace(0.0, float(t_end_h), int(n_eval))
    sol_t: list[float] = []
    sol_y: list[np.ndarray] = []

    y = np.zeros(3, dtype=float)
    eps = 1e-10

    def _rhs(tt: float, yy: np.ndarray) -> np.ndarray:
        return pk_rhs(tt, yy, ka=ka, CL=CL, Vc=Vc, Q=pk.Q, Vp=pk.Vp)

    for i in range(len(knots) - 1):
        t0, t1 = float(knots[i]), float(knots[i + 1])
        if t1 <= t0 + 1e-15:
            continue
        for td, dm in zip(eff_times, dose_amounts, strict=True):
            if abs(td - t0) <= eps:
                y[0] += bioF * dm

        sub_t = t_grid[(t_grid >= t0 - 1e-11) & (t_grid <= t1 + 1e-11)]
        if sol_t and sub_t.size > 0 and abs(float(sub_t[0]) - sol_t[-1]) < 1e-7:
            sub_t = sub_t[1:]
        if sub_t.size == 0:
            iv = solve_ivp(_rhs, (t0, t1), y, rtol=rtol, atol=atol, method="RK45")
            y = iv.y[:, -1].copy()
            continue
        iv = solve_ivp(_rhs, (t0, t1), y, t_eval=sub_t, rtol=rtol, atol=atol, method="RK45")
        if not iv.success:
            raise RuntimeError(iv.message)
        for ti, yi in zip(iv.t, iv.y.T, strict=True):
            sol_t.append(float(ti))
            sol_y.append(np.asarray(yi, dtype=float))
        y = iv.y[:, -1].copy()

    if not sol_y:
        raise RuntimeError("PK trajectory empty.")
    Y = np.vstack(sol_y)
    t_vec = np.asarray(sol_t, dtype=float)
    G, A1, A2 = Y[:, 0], Y[:, 1], Y[:, 2]
    C_mg_L = A1 / Vc
    return {"t_h": t_vec, "G_mg": G, "A1_mg": A1, "A2_mg": A2, "C_mg_L": C_mg_L, "Vc_L": np.full_like(t_vec, Vc)}


def compute_auc_daily_backward_24h(t_h: np.ndarray, C_mg_L: np.ndarray) -> np.ndarray:
    """Trapezoidal ∫_{t-24h}^{t} C dt, converted to µg·h/L (PK concentration as mg/L)."""
    t = np.asarray(t_h, dtype=float)
    c = np.asarray(C_mg_L, dtype=float)
    n = t.size
    out = np.zeros(n, dtype=float)
    win = 24.0
    for i in range(n):
        t_lo = t[i] - win
        mask = (t >= t_lo) & (t <= t[i])
        tt = t[mask]
        cc = c[mask]
        if tt.size < 2:
            out[i] = float(cc[-1] * min(win, t[i] - 0.0)) if tt.size else 0.0
        else:
            auc_mg_L_h = _trapz_yx(cc, tt)
            out[i] = auc_mg_L_h * 1000.0
    return out


def hill_auc(auc: float, auc50: float, gamma: float, imax: float = 1.0) -> float:
    if auc50 <= 0 or not np.isfinite(auc):
        return 0.0
    a = max(float(auc), 0.0)
    return float(imax * (a**gamma) / (auc50**gamma + a**gamma))


def simulate_axitinib_pkpd_chain(
    cfg: AxitinibChainConfig,
    *,
    t_end_h: float = 24.0 * 56,
    n_eval: int = 2001,
) -> dict[str, np.ndarray]:
    """PK → AUCdaily → VEGF, sVEGFR1/2/3, dBP, SLD."""
    pk_out = simulate_axitinib_pk(cfg, t_end_h=t_end_h, n_eval=n_eval)
    t = pk_out["t_h"]
    auc = compute_auc_daily_backward_24h(t, pk_out["C_mg_L"])

    bm = cfg.biomarkers
    dbp = cfg.dbp
    tg = cfg.sld

    k_v = 1.0 / max(bm.VEGF.MRT_days * 24.0, 1e-12)
    k1 = 1.0 / max(bm.sVEGFR1.MRT_days * 24.0, 1e-12)
    k2 = 1.0 / max(bm.sVEGFR2.MRT_days * 24.0, 1e-12)
    k3 = 1.0 / max(bm.sVEGFR3.MRT_days * 24.0, 1e-12)
    k_bp = 1.0 / max(dbp.MRT_days * 24.0, 1e-12)

    Rin_v = k_v * bm.VEGF.baseline_pg_ml
    Rin_1 = k1 * bm.sVEGFR1.baseline_pg_ml
    Rin_2 = k2 * bm.sVEGFR2.baseline_pg_ml
    Rin_3 = k3 * bm.sVEGFR3.baseline_pg_ml
    Rin_bp = k_bp * dbp.baseline_mmHg

    s30 = bm.sVEGFR3.baseline_pg_ml
    KG_h = tg.KG_per_week / (24.0 * 7.0)
    ks_h = tg.ksVEGFR3_per_week / (24.0 * 7.0)
    lam = tg.lambda_per_week

    y0 = np.array(
        [
            bm.VEGF.baseline_pg_ml,
            bm.sVEGFR1.baseline_pg_ml,
            bm.sVEGFR2.baseline_pg_ml,
            s30,
            dbp.baseline_mmHg,
            tg.baseline_mm,
        ],
        dtype=float,
    )

    def bm_rhs(tt: float, y: np.ndarray) -> np.ndarray:
        av = float(np.interp(tt, t, auc))
        t_year = tt / (365.25 * 24.0)

        V, s1, s2, s3, bp, sld = y

        h_v = hill_auc(av, bm.VEGF.AUC50, bm.VEGF.gamma, bm.VEGF.Imax)
        h1 = hill_auc(av, bm.sVEGFR1.AUC50, bm.sVEGFR1.gamma, bm.sVEGFR1.Imax)
        h2 = hill_auc(av, bm.sVEGFR2.AUC50, bm.sVEGFR2.gamma, bm.sVEGFR2.Imax)
        h3 = hill_auc(av, bm.sVEGFR3.AUC50, bm.sVEGFR3.gamma, bm.sVEGFR3.Imax)

        dV = Rin_v * (1.0 + bm.VEGF_disease_progression_per_year * t_year) - k_v * (1.0 - h_v) * V
        d1 = Rin_1 * (1.0 - h1) - k1 * s1
        d2 = Rin_2 * (1.0 - h2) - k2 * s2
        d3 = Rin_3 * (1.0 - h3) - k3 * s3

        stim = dbp.Emax * dbp.S0 * av / max(dbp.Emax + dbp.S0 * av, 1e-18)
        dbp_dt = Rin_bp * (1.0 + stim) - k_bp * bp

        t_week = tt / (24.0 * 7.0)
        s3_rel = (s3 - s30) / max(s30, 1e-18)
        dsld = KG_h * sld - ks_h * s3_rel * np.exp(-lam * t_week) * sld

        return np.array([dV, d1, d2, d3, dbp_dt, dsld], dtype=float)

    iv = solve_ivp(bm_rhs, (0.0, float(t[-1])), y0, t_eval=t, method="RK45", rtol=1e-8, atol=1e-10)
    if not iv.success:
        raise RuntimeError(iv.message)

    Z = iv.y.T
    out = dict(pk_out)
    out["AUCdaily_ug_h_L"] = auc
    out["VEGF_pg_ml"] = Z[:, 0]
    out["sVEGFR1_pg_ml"] = Z[:, 1]
    out["sVEGFR2_pg_ml"] = Z[:, 2]
    out["sVEGFR3_pg_ml"] = Z[:, 3]
    out["dBP_mmHg"] = Z[:, 4]
    out["SLD_mm"] = Z[:, 5]
    return out


def os_hazard_log_logistic(
    t_days: float,
    sld_mm: float,
    *,
    beta0: float = 7.09,
    c: float = 0.298,
    beta_sld: float = 0.0115,
) -> float:
    """Log-logistic hazard with time-varying SLD covariate (Eqs. 5-6, Schindler 2017).

    h(t) = psi * c * (psi*t)^(c-1) / [1 + (psi*t)^c] * exp(beta_sld * SLD(t))
    psi  = exp(-beta0)
    """
    if t_days <= 0.0:
        return 0.0
    psi = float(np.exp(-beta0))
    psi_t = psi * t_days
    numerator = psi * c * (psi_t ** (c - 1.0))
    denominator = 1.0 + (psi_t ** c)
    baseline_hazard = numerator / denominator
    return float(baseline_hazard * np.exp(beta_sld * sld_mm))


def load_axitinib_defaults_yaml(path: str | Path | None = None) -> AxitinibChainConfig:
    """Load ``literature_defaults.yaml`` into ``AxitinibChainConfig``."""
    if yaml is None:
        raise RuntimeError("PyYAML required.")
    p = Path(path) if path is not None else Path(__file__).resolve().parent / "literature_defaults.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    pk = raw["pharmacokinetics"]
    cov = raw["covariates"]
    bm = raw["biomarkers"]
    cfg = AxitinibChainConfig(
        pk=AxitinibPKParams(
            CL=float(pk["CL_L_per_h"]),
            Vc=float(pk["Vc_L"]),
            Q=float(pk["Q_L_per_h"]),
            Vp=float(pk["Vp_L"]),
            ka_fed=float(pk["ka_fed_per_h"]),
            ka_fasted=float(pk["ka_fasted_per_h"]),
            tlag=float(pk["tlag_h"]),
            F_fed_IV=float(pk["F_fed_IV"]),
            F_fasted_IV=float(pk["F_fasted_IV"]),
            F_XLI=float(pk["F_XLI"]),
        ),
        covariates=AxitinibCovariates(
            Vc_weight_exponent=float(cov["Vc_weight_exponent"]),
            weight_ref_kg=float(cov["Vc_weight_ref_kg"]),
        ),
        regimen=AxitinibRegimen(
            dose_mg=float(raw["regimen"]["dose_mg_per_administration"]),
            interval_h=float(raw["regimen"]["interval_h"]),
            fed_state=str(raw["regimen"]["fed_state"]),
            formulation=str(raw["regimen"]["formulation"]),
        ),
        biomarkers=AxitinibBiomarkerParams(
            VEGF=BiomarkerBlock(
                float(bm["VEGF"]["baseline_pg_ml"]),
                float(bm["VEGF"]["MRT_days"]),
                float(bm["VEGF"]["AUC50_ug_h_per_L"]),
                float(bm["VEGF"]["Imax"]),
                float(bm["VEGF"]["gamma"]),
            ),
            sVEGFR1=BiomarkerBlock(
                float(bm["sVEGFR1"]["baseline_pg_ml"]),
                float(bm["sVEGFR1"]["MRT_days"]),
                float(bm["sVEGFR1"]["AUC50_ug_h_per_L"]),
                float(bm["sVEGFR1"]["Imax"]),
                float(bm["sVEGFR1"]["gamma"]),
            ),
            sVEGFR2=BiomarkerBlock(
                float(bm["sVEGFR2"]["baseline_pg_ml"]),
                float(bm["sVEGFR2"]["MRT_days"]),
                float(bm["sVEGFR2"]["AUC50_ug_h_per_L"]),
                float(bm["sVEGFR2"]["Imax"]),
                float(bm["sVEGFR2"]["gamma"]),
            ),
            sVEGFR3=BiomarkerBlock(
                float(bm["sVEGFR3"]["baseline_pg_ml"]),
                float(bm["sVEGFR3"]["MRT_days"]),
                float(bm["sVEGFR3"]["AUC50_ug_h_per_L"]),
                float(bm["sVEGFR3"]["Imax"]),
                float(bm["sVEGFR3"]["gamma"]),
            ),
            VEGF_disease_progression_per_year=float(bm["VEGF"]["disease_progression_per_year"]),
        ),
        dbp=DBPParams(
            baseline_mmHg=float(raw["diastolic_bp"]["baseline_mmHg"]),
            MRT_days=float(raw["diastolic_bp"]["MRT_days"]),
            Emax=float(raw["diastolic_bp"]["Emax"]),
            S0=float(raw["diastolic_bp"]["S0"]),
        ),
        sld=SLDParams(
            KG_per_week=float(raw["tumor_sld"]["KG_per_week"]),
            ksVEGFR3_per_week=float(raw["tumor_sld"]["ksVEGFR3_per_week"]),
            lambda_per_week=float(raw["tumor_sld"]["lambda_per_week"]),
            baseline_mm=float(raw["tumor_sld"]["baseline_SLD_mm"]),
        ),
    )
    return cfg
