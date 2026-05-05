"""Sunitinib + SU12662 PK and dual sVEGFR biomarker PD (linked via ACu).

- **PK**: Two-compartment parent (oral ka) with linear clearance and inter-compartmental
  transfer; metabolite SU12662 as a second two-compartment chain fed by a fraction
  ``fm`` of parent clearance from the central compartment.

- **PD**: Two indirect-response *inhibition of production* pathways (same ODE as
  ``term:pd.TurnoverInhibProduction``):

  ``dR/dt = kin * (1 - Imax * C / (IC50 + C)) - kout * R``

  with shared driver ``ACu`` (unbound active concentration, ng/mL). Paper-style
  ``╬▒┬ÀACu/(Kd + ╬▒┬ÀACu)`` with ``Imax=1`` uses ``IC50 = Kd / ╬▒``.

Amounts **mg**, volumes **L**, time **h**; PD driver **ng/mL** (mg/L ├ù 1000).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger
from scipy.integrate import solve_ivp

Cohort = Literal["mRCC", "mCRC"]


@dataclass
class ParentPKParams:
    ka: float = 0.133  # hÔü╗┬╣
    CL: float = 33.9  # L/h
    V1: float = 1820.0  # L
    V2: float = 588.0  # L
    Q: float = 0.371  # L/h
    QH: float = 80.0  # L/h, hepatic blood flow linking the enzyme compartment
    fm: float = 0.21  # fraction of parent clearance that forms metabolite


@dataclass
class MetabolitePKParams:
    CLm: float = 16.5  # L/h
    V1m: float = 730.0  # L
    V2m: float = 592.0  # L
    Qm: float = 2.75  # L/h


@dataclass
class ProteinBinding:
    """Bound fraction ÔåÆ fu = 1 - f_bound."""

    fu_parent: float = 0.05  # 95% bound sunitinib
    fu_metabolite: float = 0.10  # 90% bound SU12662
    metabolite_unbound_weight: float = 1.0  # ACu = Cu_p + Cu_m per Diekstra et al. 2017


@dataclass
class BiomarkerPDParams:
    """Turnover inhibition of production for one biomarker."""

    baseline_ug_L: float
    kout: float  # hÔü╗┬╣
    kin: float | None = None  # ┬Ág/L/h; default baseline * kout
    Imax: float = 1.0
    IC50_ng_ml: float | None = None  # if None, derived from Kd/alpha
    Kd_ng_ml: float = 4.0
    alpha_intrinsic: float = 1.0  # used only when IC50_ng_ml is None

    def __post_init__(self) -> None:
        if self.kin is None:
            object.__setattr__(self, "kin", float(self.baseline_ug_L * self.kout))
        if self.IC50_ng_ml is None:
            object.__setattr__(self, "IC50_ng_ml", float(self.Kd_ng_ml / max(self.alpha_intrinsic, 1e-9)))


@dataclass
class CovariatePDModifiers:
    """Optional cohort / genotype scaling on PD."""

    cohort: Cohort = "mRCC"
    vegfr3_rs6877011_g_allele: bool = False  # sVEGFR-2 ╬▒ ÔêÆ56.5%
    abcb1_rs2032582_t_allele: bool = False  # sVEGFR-2 ╬▒ ÔêÆ31.1%
    tumor_mcrc_reduce_vegfr2_alpha: float = 0.328
    tumor_mcrc_reduce_vegfr3_baseline: float = 0.642


@dataclass
class IntermittentRegimen:
    """4 weeks daily dosing ON, 2 weeks OFF (repeat)."""

    dose_mg: float = 50.0
    interval_h: float = 24.0
    on_weeks: int = 4
    off_weeks: int = 2


@dataclass
class SunitinibPKPDConfig:
    parent: ParentPKParams = field(default_factory=ParentPKParams)
    metabolite: MetabolitePKParams = field(default_factory=MetabolitePKParams)
    binding: ProteinBinding = field(default_factory=ProteinBinding)
    vegfr2: BiomarkerPDParams = field(
        default_factory=lambda: BiomarkerPDParams(
            baseline_ug_L=9.0,
            kout=0.0043,
            alpha_intrinsic=2.31,
            Kd_ng_ml=4.0,
        )
    )
    vegfr3: BiomarkerPDParams = field(
        default_factory=lambda: BiomarkerPDParams(
            baseline_ug_L=63.5,
            kout=0.0053,
            alpha_intrinsic=1.74,
            Kd_ng_ml=4.0,
        )
    )
    covariates: CovariatePDModifiers = field(default_factory=CovariatePDModifiers)
    regimen: IntermittentRegimen = field(default_factory=IntermittentRegimen)


def _apply_covariates_to_pd(cfg: SunitinibPKPDConfig) -> tuple[BiomarkerPDParams, BiomarkerPDParams]:
    """Return (vegfr2, vegfr3) params with cohort/genotype scaling."""
    src2 = cfg.vegfr2
    src3 = cfg.vegfr3
    m = cfg.covariates

    alpha2 = src2.alpha_intrinsic
    if m.cohort == "mCRC":
        alpha2 *= 1.0 - m.tumor_mcrc_reduce_vegfr2_alpha
    if m.vegfr3_rs6877011_g_allele:
        alpha2 *= 1.0 - 0.565
    if m.abcb1_rs2032582_t_allele:
        alpha2 *= 1.0 - 0.311

    v2 = BiomarkerPDParams(
        baseline_ug_L=src2.baseline_ug_L,
        kout=src2.kout,
        kin=src2.kin,
        Imax=src2.Imax,
        Kd_ng_ml=src2.Kd_ng_ml,
        alpha_intrinsic=max(alpha2, 1e-9),
        IC50_ng_ml=None,
    )

    base3 = src3.baseline_ug_L
    if m.cohort == "mCRC":
        base3 *= 1.0 - m.tumor_mcrc_reduce_vegfr3_baseline
    v3 = BiomarkerPDParams(
        baseline_ug_L=base3,
        kout=src3.kout,
        kin=None,
        Imax=src3.Imax,
        Kd_ng_ml=src3.Kd_ng_ml,
        alpha_intrinsic=src3.alpha_intrinsic,
        IC50_ng_ml=None,
    )
    return v2, v3


def build_intermittent_dose_times(t_end_h: float, reg: IntermittentRegimen) -> list[float]:
    """Absolute dose times (h) for repeating 4-on / 2-off weeks, one dose per interval_h while ON."""
    cycle_h = (reg.on_weeks + reg.off_weeks) * 7.0 * 24.0
    on_h = reg.on_weeks * 7.0 * 24.0
    times: list[float] = []
    cyc = 0
    while True:
        t0 = cyc * cycle_h
        if t0 >= t_end_h:
            break
        for k in range(int(on_h / reg.interval_h)):
            td = t0 + k * reg.interval_h
            if td < t_end_h:
                times.append(float(td))
        cyc += 1
    return sorted(set(times))


def _mg_L_to_ng_ml(c_mg_per_L: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(c_mg_per_L, dtype=float) * 1000.0
    if arr.ndim == 0:
        return float(arr)
    return arr


def active_unbound_ng_ml(
    A1_mg: float,
    Am1_mg: float,
    V1: float,
    V1m: float,
    binding: ProteinBinding,
) -> float:
    """ACu driver (ng/mL) from central amounts."""
    c_p = (A1_mg / V1) if V1 > 0 else 0.0
    c_m = (Am1_mg / V1m) if V1m > 0 else 0.0
    ng_p = _mg_L_to_ng_ml(c_p)
    ng_m = _mg_L_to_ng_ml(c_m)
    cu_p = binding.fu_parent * float(ng_p)
    cu_m = binding.fu_metabolite * float(ng_m)
    return float(cu_p + binding.metabolite_unbound_weight * cu_m)


def pk_pd_rhs(
    t: float,
    y: np.ndarray,
    pk_p: ParentPKParams,
    pk_m: MetabolitePKParams,
    pd2: BiomarkerPDParams,
    pd3: BiomarkerPDParams,
    binding: ProteinBinding,
) -> np.ndarray:
    """Combined PKÔÇôPD right-hand side (no dose terms)."""
    Ap, A1, A2, Ahe, Am1, Am2, R2, R3 = y
    ka, CL, V1, V2, Q, QH = pk_p.ka, pk_p.CL, pk_p.V1, pk_p.V2, pk_p.Q, pk_p.QH
    fm = pk_p.fm
    CLm, V1m, V2m, Qm = pk_m.CLm, pk_m.V1m, pk_m.V2m, pk_m.Qm

    dAp = -ka * Ap
    dA1 = ka * Ap + QH * (Ahe / V1 - A1 / V1) + Q * (A2 / V2 - A1 / V1)
    dA2 = Q * (A1 / V1 - A2 / V2)
    dAhe = QH * (A1 / V1 - Ahe / V1) - CL * Ahe / V1
    dAm1 = fm * CL * Ahe / V1 + Qm * (Am2 / V2m - Am1 / V1m) - CLm * Am1 / V1m
    dAm2 = Qm * (Am1 / V1m - Am2 / V2m)

    acu = active_unbound_ng_ml(A1, Am1, V1, V1m, binding)
    inh2 = acu / max(pd2.Kd_ng_ml + acu, 1e-18)
    prod2 = 1.0 / (1.0 + pd2.alpha_intrinsic * inh2)
    dR2 = float(pd2.kin) * prod2 - pd2.kout * R2

    inh3 = acu / max(pd3.Kd_ng_ml + acu, 1e-18)
    prod3 = 1.0 / (1.0 + pd3.alpha_intrinsic * inh3)
    dR3 = float(pd3.kin) * prod3 - pd3.kout * R3

    return np.array([dAp, dA1, dA2, dAhe, dAm1, dAm2, dR2, dR3], dtype=float)


def simulate_sunitinib_pkpd(
    cfg: SunitinibPKPDConfig,
    *,
    t_end_h: float = 24.0 * 7.0 * 24.0,  # 24 weeks
    n_eval: int = 2001,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Integrate PK + dual biomarker PD with oral intermittent dosing.

    Returns ``t_h``, compartment amounts, ``C_parent_ng_ml``, ``C_metabolite_ng_ml``,
    ``ACu_ng_ml``, ``sVEGFR2_ug_L``, ``sVEGFR3_ug_L``, and effective ``IC50_*`` arrays.
    """
    dose_times = sorted(set(build_intermittent_dose_times(t_end_h, cfg.regimen)))
    knots = sorted({0.0, t_end_h, *[d for d in dose_times if d <= t_end_h]})
    pd2, pd3 = _apply_covariates_to_pd(cfg)

    y = np.zeros(8, dtype=float)
    y[6] = pd2.baseline_ug_L
    y[7] = pd3.baseline_ug_L

    t_grid = np.linspace(0.0, float(t_end_h), int(n_eval))
    sol_t: list[float] = []
    sol_y: list[np.ndarray] = []

    def _rhs(tt: float, yy: np.ndarray) -> np.ndarray:
        return pk_pd_rhs(tt, yy, cfg.parent, cfg.metabolite, pd2, pd3, cfg.binding)

    eps = 1e-9
    for i in range(len(knots) - 1):
        t0, t1 = float(knots[i]), float(knots[i + 1])
        if t1 <= t0 + 1e-15:
            continue
        for td in dose_times:
            if abs(td - t0) <= eps:
                y[0] += cfg.regimen.dose_mg

        sub_t = t_grid[(t_grid >= t0 - 1e-11) & (t_grid <= t1 + 1e-11)]
        if sol_t and sub_t.size > 0 and abs(float(sub_t[0]) - sol_t[-1]) < 1e-7:
            sub_t = sub_t[1:]

        if sub_t.size == 0:
            iv = solve_ivp(_rhs, (t0, t1), y, rtol=rtol, atol=atol, method="RK45")
            if not iv.success:
                logger.error("solve_ivp failed: {}", iv.message)
                raise RuntimeError(iv.message)
            y = iv.y[:, -1].copy()
            continue

        iv = solve_ivp(
            _rhs,
            (t0, t1),
            y,
            t_eval=sub_t,
            rtol=rtol,
            atol=atol,
            method="RK45",
        )
        if not iv.success:
            logger.error("solve_ivp failed: {}", iv.message)
            raise RuntimeError(iv.message)

        for ti, yi in zip(iv.t, iv.y.T, strict=True):
            sol_t.append(float(ti))
            sol_y.append(np.asarray(yi, dtype=float))
        y = iv.y[:, -1].copy()

    if not sol_y:
        raise RuntimeError("No trajectory samples produced; increase t_end_h or n_eval.")
    Y = np.vstack(sol_y)
    t_vec = np.asarray(sol_t, dtype=float)
    Ap, A1, A2, Ahe, Am1, Am2, R2, R3 = [Y[:, i] for i in range(8)]

    c_p_ng = _mg_L_to_ng_ml(A1 / cfg.parent.V1)
    c_m_ng = _mg_L_to_ng_ml(Am1 / cfg.metabolite.V1m)
    acu = np.array(
        [active_unbound_ng_ml(a1, am1, cfg.parent.V1, cfg.metabolite.V1m, cfg.binding) for a1, am1 in zip(A1, Am1, strict=True)],
        dtype=float,
    )

    return {
        "t_h": t_vec,
        "Ap_mg": Ap,
        "A1_mg": A1,
        "A2_mg": A2,
        "Ahep_mg": Ahe,
        "Am1_mg": Am1,
        "Am2_mg": Am2,
        "C_parent_ng_ml": c_p_ng,
        "C_metabolite_ng_ml": c_m_ng,
        "ACu_ng_ml": acu,
        "sVEGFR2_ug_L": R2,
        "sVEGFR3_ug_L": R3,
        "IC50_vegfr2_ng_ml": np.full_like(t_vec, pd2.IC50_ng_ml),
        "IC50_vegfr3_ng_ml": np.full_like(t_vec, pd3.IC50_ng_ml),
    }
