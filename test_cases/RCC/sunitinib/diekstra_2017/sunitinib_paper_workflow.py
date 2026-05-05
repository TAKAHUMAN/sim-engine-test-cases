"""Diekstra et al. 2017 mRCC paper: load Table 2/3 parameters, VPC, GOF, light fitting.

Loads ``microservice/tests/sunitinib_mRCC_params.{yaml,json}`` (same estimates),
runs baseline PKPD simulation, Monte Carlo simulation-based VPC with IIV
(log-normal on reported CV%), optional proportional residual on log scale for PK
and for PD outputs, compares user-supplied observations to simulated percentiles,
and can adjust **ka**, **CL**, **Kd/╬▒** (via ``Kd_ng_ml`` scaling), **kout** with
``scipy.optimize.minimize``.

**Plots:** ``plot_baseline_pkpd_figure``, ``plot_vpc_figure``, and
``save_sunitinib_diagnostic_pngs`` write PNGs (matplotlib ``Agg``). Run
``python microservice/sunitinib_paper_workflow.py <output_dir>`` from the repo
root (script bootstraps ``sys.path``), or call ``save_sunitinib_diagnostic_pngs(...)`` from code.

**Note:** Full *prediction-corrected* VPC (NONMEM-style) needs empirical Bayes
individual corrections; this module implements a **simulation VPC** (replicate
individuals from population + residual) which matches the usual Figure 2-style
visual predictive check when observations are provided.

**IIV:** Independent log-normal random effects with SD Ôëê (IIV% / 100) on the natural
log scale (common approximation for moderate CV%). Table 2 correlations are not
applied in this version (documented limitation).
"""

from __future__ import annotations

# Repo root on sys.path so ``python microservice/sunitinib_paper_workflow.py`` works.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from loguru import logger
from scipy.optimize import minimize

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from microservice.sunitinib_pkpd import (
    BiomarkerPDParams,
    IntermittentRegimen,
    MetabolitePKParams,
    ParentPKParams,
    ProteinBinding,
    SunitinibPKPDConfig,
    simulate_sunitinib_pkpd,
)

OutcomeKey = Literal["C_parent_ng_ml", "C_metabolite_ng_ml", "sVEGFR2_ug_L", "sVEGFR3_ug_L", "ACu_ng_ml"]


def _est(block: Any, default: float | None = None) -> float | None:
    if block is None:
        return default
    if isinstance(block, (int, float)):
        return float(block)
    if isinstance(block, dict) and "estimate" in block:
        return float(block["estimate"])
    return default


def _iiv(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    for k in ("IIV_percent", "interindividual_variability_percent"):
        if k in block and block[k] is not None:
            return float(block[k])
    return None


def _sigma_from_residual(block: Any) -> float:
    """Use |estimate| as log-scale proportional SD for PK; positive estimate for PD."""
    if not isinstance(block, dict):
        return 0.0
    e = block.get("estimate")
    if e is None:
        return 0.0
    return abs(float(e))


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load .yaml parameter files.")
        out = yaml.safe_load(text)
        if not isinstance(out, dict):
            raise ValueError(f"Expected mapping at root of {path}")
        return out
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported parameter file suffix: {path.suffix}")


def _pk_parent_blocks(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pk = raw.get("pharmacokinetics") or {}
    parent = pk.get("sunitinib_parent") or pk.get("sunitinib") or {}
    met = pk.get("su12662_metabolite") or pk.get("su12662") or {}
    return parent, met


def _pd_blocks(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pd = raw.get("pharmacodynamics") or {}
    v2 = pd.get("sVEGFR2") or pd.get("sVEGFR-2") or {}
    v3 = pd.get("sVEGFR3") or pd.get("sVEGFR-3") or {}
    return v2, v3


def _dosing_from_raw(raw: dict[str, Any]) -> IntermittentRegimen:
    d = raw.get("dosing") or {}
    dose = float(d.get("dose_mg") or d.get("dose") or 50.0)
    sched = d.get("schedule") or {}
    on_w = int(sched.get("on_weeks") or (sched.get("on_phase_days", 28) // 7) or 4)
    off_w = int(sched.get("off_weeks") or (sched.get("off_phase_days", 14) // 7) or 2)
    interval = float(sched.get("interval_hours") or 24.0)
    return IntermittentRegimen(dose_mg=dose, interval_h=interval, on_weeks=on_w, off_weeks=off_w)


def load_sunitinib_paper_config(path: str | Path) -> tuple[SunitinibPKPDConfig, "PaperVPCSpec"]:
    """Build ``SunitinibPKPDConfig`` + VPC noise/IIV metadata from YAML or JSON."""
    p = Path(path)
    raw = _load_raw(p)
    parent, met = _pk_parent_blocks(raw)
    v2, v3 = _pd_blocks(raw)
    bind = (raw.get("pharmacokinetics") or {}).get("protein_binding") or {}
    fu_p = float(bind.get("sunitinib_unbound_fraction") or 0.05)
    fu_m = float(bind.get("su12662_unbound_fraction") or 0.10)
    fp = (raw.get("pharmacokinetics") or {}).get("fixed_parameters") or {}

    pk_parent = ParentPKParams(
        ka=float(_est(parent.get("ka"), 0.133) or 0.133),
        CL=float(_est(parent.get("CL"), 33.9) or 33.9),
        V1=float(_est(parent.get("V1"), 1820.0) or 1820.0),
        V2=float(_est(parent.get("V2"), 588.0) or 588.0),
        Q=float(_est(parent.get("Q"), 0.371) or 0.371),
        QH=float(_est(fp.get("QH_liver_blood_flow"), 80.0) or 80.0),
        fm=float(_est(parent.get("fm"), 0.21) or 0.21),
    )
    pk_met = MetabolitePKParams(
        CLm=float(_est(met.get("CLm"), 16.5) or 16.5),
        V1m=float(_est(met.get("V1m"), 730.0) or 730.0),
        V2m=float(_est(met.get("V2m"), 592.0) or 592.0),
        Qm=float(_est(met.get("Qm"), 2.75) or 2.75),
    )
    pd_sh = (raw.get("pharmacodynamics") or {}).get("shared_parameters") or {}
    kd_raw = _est(pd_sh.get("Kd")) or _est(fp.get("Kd_dissociation_constant"))
    kd_shared = float(kd_raw if kd_raw is not None else 4.0)

    vegfr2 = BiomarkerPDParams(
        baseline_ug_L=float(_est(v2.get("baseline"), 9.0) or 9.0),
        kout=float(_est(v2.get("kout"), 0.0043) or 0.0043),
        Kd_ng_ml=kd_shared,
        alpha_intrinsic=float(_est(v2.get("alpha"), 2.31) or 2.31),
        IC50_ng_ml=None,
    )
    vegfr3 = BiomarkerPDParams(
        baseline_ug_L=float(_est(v3.get("baseline"), 63.5) or 63.5),
        kout=float(_est(v3.get("kout"), 0.0053) or 0.0053),
        Kd_ng_ml=kd_shared,
        alpha_intrinsic=float(_est(v3.get("alpha"), 1.74) or 1.74),
        IC50_ng_ml=None,
    )
    cfg = SunitinibPKPDConfig(
        parent=pk_parent,
        metabolite=pk_met,
        binding=ProteinBinding(fu_parent=fu_p, fu_metabolite=fu_m),
        vegfr2=vegfr2,
        vegfr3=vegfr3,
        regimen=_dosing_from_raw(raw),
    )

    spec = PaperVPCSpec.from_raw_blocks(parent, met, v2, v3)
    return cfg, spec


@dataclass
class PaperVPCSpec:
    """IIV (CV %) and residual sigmas distilled from Tables 2ÔÇô3."""

    iiv_CL_percent: float = 30.3
    iiv_V1_percent: float = 25.3
    iiv_V1m_percent: float = 42.9
    iiv_fm_percent: float = 34.6
    iiv_ka_percent: float = 0.0  # not estimated in Table 2 (dash); optional perturbation
    iiv_baseline_vegfr2_percent: float = 19.9
    iiv_baseline_vegfr3_percent: float = 42.6
    iiv_alpha_vegfr3_percent: float = 54.3
    sigma_pk_parent_prop: float = 0.367
    sigma_pk_met_prop: float = 0.281
    sigma_pd_vegfr2_prop: float = 0.124
    sigma_pd_vegfr3_prop: float = 0.15

    @staticmethod
    def from_raw_blocks(parent: dict, met: dict, v2: dict, v3: dict) -> "PaperVPCSpec":
        return PaperVPCSpec(
            iiv_CL_percent=float(_iiv(parent.get("CL")) or 30.3),
            iiv_V1_percent=float(_iiv(parent.get("V1")) or 25.3),
            iiv_V1m_percent=float(_iiv(met.get("V1m")) or 42.9),
            iiv_fm_percent=float(_iiv(parent.get("fm")) or 34.6),
            iiv_baseline_vegfr2_percent=float(_iiv(v2.get("baseline")) or 19.9),
            iiv_baseline_vegfr3_percent=float(_iiv(v3.get("baseline")) or 42.6),
            iiv_alpha_vegfr3_percent=float(_iiv(v3.get("alpha")) or 54.3),
            sigma_pk_parent_prop=_sigma_from_residual(parent.get("residual_error")),
            sigma_pk_met_prop=_sigma_from_residual(met.get("residual_error")),
            sigma_pd_vegfr2_prop=_sigma_from_residual(v2.get("residual_error")),
            sigma_pd_vegfr3_prop=_sigma_from_residual(v3.get("residual_error")),
        )


def _lniiv(typical: float, cv_percent: float, rng: np.random.Generator) -> float:
    if cv_percent <= 0.0:
        return float(typical)
    sd = cv_percent / 100.0
    return float(typical * np.exp(sd * rng.standard_normal()))


def sample_individual_config(base: SunitinibPKPDConfig, spec: PaperVPCSpec, rng: np.random.Generator) -> SunitinibPKPDConfig:
    """One Monte Carlo individual: PK + PD random effects (independent)."""
    ka = _lniiv(base.parent.ka, spec.iiv_ka_percent, rng) if spec.iiv_ka_percent > 0 else base.parent.ka
    CL = _lniiv(base.parent.CL, spec.iiv_CL_percent, rng)
    V1 = _lniiv(base.parent.V1, spec.iiv_V1_percent, rng)
    fm = _lniiv(base.parent.fm, spec.iiv_fm_percent, rng)
    fm = float(np.clip(fm, 0.01, 0.99))
    V1m = _lniiv(base.metabolite.V1m, spec.iiv_V1m_percent, rng)
    b2 = _lniiv(base.vegfr2.baseline_ug_L, spec.iiv_baseline_vegfr2_percent, rng)
    b3 = _lniiv(base.vegfr3.baseline_ug_L, spec.iiv_baseline_vegfr3_percent, rng)
    a3 = _lniiv(base.vegfr3.alpha_intrinsic, spec.iiv_alpha_vegfr3_percent, rng)
    a3 = max(a3, 1e-6)

    parent = replace(base.parent, ka=ka, CL=CL, V1=V1, fm=fm)
    met = replace(base.metabolite, V1m=V1m)
    veg2 = BiomarkerPDParams(
        baseline_ug_L=b2,
        kout=base.vegfr2.kout,
        kin=None,
        Imax=base.vegfr2.Imax,
        Kd_ng_ml=base.vegfr2.Kd_ng_ml,
        alpha_intrinsic=base.vegfr2.alpha_intrinsic,
        IC50_ng_ml=None,
    )
    veg3 = BiomarkerPDParams(
        baseline_ug_L=b3,
        kout=base.vegfr3.kout,
        kin=None,
        Imax=base.vegfr3.Imax,
        Kd_ng_ml=base.vegfr3.Kd_ng_ml,
        alpha_intrinsic=a3,
        IC50_ng_ml=None,
    )
    return replace(base, parent=parent, metabolite=met, vegfr2=veg2, vegfr3=veg3)


def _apply_residual(pred: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0.0:
        return pred
    noise = rng.standard_normal(pred.shape)
    return pred * np.exp(sigma * noise)


def run_baseline_simulation(cfg: SunitinibPKPDConfig, **kwargs: Any) -> dict[str, np.ndarray]:
    """Table 2/3 point estimates only (no IIV)."""
    return simulate_sunitinib_pkpd(cfg, **kwargs)


def run_simulation_vpc(
    base_cfg: SunitinibPKPDConfig,
    spec: PaperVPCSpec,
    *,
    n_sim: int = 1000,
    outcome_key: OutcomeKey = "sVEGFR2_ug_L",
    t_end_h: float = 24.0 * 7.0 * 12.0,
    n_eval: int = 501,
    seed: int | None = None,
    add_residual: bool = True,
) -> dict[str, Any]:
    """Monte Carlo VPC: ``n_sim`` individuals, return 5/50/95% envelopes + sim matrix."""
    rng = np.random.default_rng(seed)
    sims: list[np.ndarray] = []
    t_ref: np.ndarray | None = None
    for _ in range(int(n_sim)):
        ind = sample_individual_config(base_cfg, spec, rng)
        out = simulate_sunitinib_pkpd(ind, t_end_h=t_end_h, n_eval=n_eval)
        t_ref = out["t_h"]
        y = np.asarray(out[outcome_key], dtype=float).copy()
        if add_residual:
            if outcome_key in ("C_parent_ng_ml",):
                y = _apply_residual(y, spec.sigma_pk_parent_prop, rng)
            elif outcome_key in ("C_metabolite_ng_ml",):
                y = _apply_residual(y, spec.sigma_pk_met_prop, rng)
            elif outcome_key == "sVEGFR2_ug_L":
                y = _apply_residual(y, spec.sigma_pd_vegfr2_prop, rng)
            elif outcome_key == "sVEGFR3_ug_L":
                y = _apply_residual(y, spec.sigma_pd_vegfr3_prop, rng)
            elif outcome_key == "ACu_ng_ml":
                y = _apply_residual(y, max(spec.sigma_pk_parent_prop, spec.sigma_pk_met_prop), rng)
        sims.append(y)
    if t_ref is None:
        raise RuntimeError("VPC produced no simulations.")
    Y = np.vstack(sims)
    p5, p50, p95 = np.percentile(Y, [5.0, 50.0, 95.0], axis=0)
    return {
        "t_h": t_ref,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "sim_matrix": Y,
        "outcome_key": outcome_key,
        "n_sim": int(n_sim),
    }


def interpolate_envelope(t_obs: np.ndarray, t_grid: np.ndarray, lo: np.ndarray, mid: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo_i = np.interp(t_obs, t_grid, lo)
    mid_i = np.interp(t_obs, t_grid, mid)
    hi_i = np.interp(t_obs, t_grid, hi)
    return lo_i, mid_i, hi_i


def vpc_compare_observed(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    vpc: dict[str, Any],
) -> dict[str, Any]:
    """Count how many observations fall inside simulated 90% interval (p5ÔÇôp95)."""
    t_g = np.asarray(vpc["t_h"], dtype=float)
    p5, p50, p95 = vpc["p5"], vpc["p50"], vpc["p95"]
    lo, _, hi = interpolate_envelope(np.asarray(t_obs, dtype=float), t_g, p5, p50, p95)
    y = np.asarray(y_obs, dtype=float)
    inside = (y >= lo) & (y <= hi)
    below = y < lo
    above = y > hi
    return {
        "n_obs": int(y.size),
        "fraction_inside_90": float(np.mean(inside)),
        "fraction_below_p5": float(np.mean(below)),
        "fraction_above_p95": float(np.mean(above)),
        "expected_fraction_inside_90_nominal": 0.9,
    }


def absolute_average_folding_error(pred: np.ndarray, obs: np.ndarray) -> float:
    """AAFE = mean(|log10(pred/obs)|); pred and obs strictly positive finite."""
    p = np.asarray(pred, dtype=float).ravel()
    o = np.asarray(obs, dtype=float).ravel()
    m = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(np.log10(p[m] / o[m]))))


def folding_error(pred: np.ndarray, obs: np.ndarray) -> float:
    """FE = 10^AAFE (symmetric fold error scale)."""
    aafe = absolute_average_folding_error(pred, obs)
    if not np.isfinite(aafe):
        return float("nan")
    return float(10.0**aafe)


def root_mean_square_error(pred: np.ndarray, obs: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float).ravel()
    o = np.asarray(obs, dtype=float).ravel()
    n = min(p.size, o.size)
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p[:n] - o[:n]) ** 2)))


def fit_key_pkpd_params(
    base_cfg: SunitinibPKPDConfig,
    spec: PaperVPCSpec,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    *,
    outcome_key: OutcomeKey = "C_parent_ng_ml",
    t_end_h: float = 24.0 * 7.0 * 8.0,
    n_eval: int = 301,
    n_sim_mean: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit **ka**, **CL**, **Kd** (scales both biomarkers), **vegfr2 kout** by minimizing AAFE.

    Uses a **small** Monte Carlo inner loop (``n_sim_mean``) for smooth objective;
    increase for production fits.
    """
    rng = np.random.default_rng(seed)
    t_obs = np.asarray(t_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    def _objective(x: np.ndarray) -> float:
        log_ka, log_CL, log_Kd, log_kout2 = x
        ka = float(np.exp(log_ka))
        CL = float(np.exp(log_CL))
        kd = float(np.exp(log_Kd))
        k2 = float(np.exp(log_kout2))
        parent = replace(base_cfg.parent, ka=ka, CL=CL)
        veg2 = BiomarkerPDParams(
            baseline_ug_L=base_cfg.vegfr2.baseline_ug_L,
            kout=k2,
            kin=None,
            Imax=base_cfg.vegfr2.Imax,
            Kd_ng_ml=kd,
            alpha_intrinsic=base_cfg.vegfr2.alpha_intrinsic,
            IC50_ng_ml=None,
        )
        veg3 = BiomarkerPDParams(
            baseline_ug_L=base_cfg.vegfr3.baseline_ug_L,
            kout=base_cfg.vegfr3.kout,
            kin=None,
            Imax=base_cfg.vegfr3.Imax,
            Kd_ng_ml=kd,
            alpha_intrinsic=base_cfg.vegfr3.alpha_intrinsic,
            IC50_ng_ml=None,
        )
        trial = replace(base_cfg, parent=parent, vegfr2=veg2, vegfr3=veg3)
        preds: list[np.ndarray] = []
        for _ in range(int(n_sim_mean)):
            ind = sample_individual_config(trial, spec, rng)
            out = simulate_sunitinib_pkpd(ind, t_end_h=t_end_h, n_eval=n_eval)
            y = np.asarray(out[outcome_key], dtype=float)
            preds.append(np.interp(t_obs, out["t_h"], y))
        mean_pred = np.mean(np.stack(preds, axis=0), axis=0)
        aafe = absolute_average_folding_error(mean_pred, y_obs)
        if not np.isfinite(aafe):
            return 1e6
        logger.trace("fit obj aafe={}", aafe)
        return aafe

    x0 = np.log(
        [
            base_cfg.parent.ka,
            base_cfg.parent.CL,
            base_cfg.vegfr2.Kd_ng_ml,
            base_cfg.vegfr2.kout,
        ]
    )
    res = minimize(_objective, x0, method="Nelder-Mead", options={"maxiter": 120, "xatol": 0.02, "fatol": 0.02})
    opt = res.x
    fitted_cfg = replace(
        base_cfg,
        parent=replace(
            base_cfg.parent,
            ka=float(np.exp(opt[0])),
            CL=float(np.exp(opt[1])),
        ),
        vegfr2=BiomarkerPDParams(
            baseline_ug_L=base_cfg.vegfr2.baseline_ug_L,
            kout=float(np.exp(opt[3])),
            kin=None,
            Imax=base_cfg.vegfr2.Imax,
            Kd_ng_ml=float(np.exp(opt[2])),
            alpha_intrinsic=base_cfg.vegfr2.alpha_intrinsic,
            IC50_ng_ml=None,
        ),
        vegfr3=BiomarkerPDParams(
            baseline_ug_L=base_cfg.vegfr3.baseline_ug_L,
            kout=base_cfg.vegfr3.kout,
            kin=None,
            Imax=base_cfg.vegfr3.Imax,
            Kd_ng_ml=float(np.exp(opt[2])),
            alpha_intrinsic=base_cfg.vegfr3.alpha_intrinsic,
            IC50_ng_ml=None,
        ),
    )
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "fitted_config": fitted_cfg,
        "final_aafe": float(res.fun),
        "x_log": opt.tolist(),
    }


def default_paper_param_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[0]
    tests = root / "tests"
    return tests / "sunitinib_mRCC_params.yaml", tests / "sunitinib_mRCC_params.json"


# ÔöÇÔöÇ Matplotlib diagnostics (VPC + baseline) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_baseline_pkpd_figure(out: dict[str, Any], *, title: str = "Sunitinib PKPD (baseline)") -> Any:
    """Time courses: parent / metabolite concentration, ACu, biomarkers."""
    plt = _plt()
    t = np.asarray(out["t_h"], dtype=float) / 24.0
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=12)

    ax = axes[0, 0]
    ax.plot(t, out["C_parent_ng_ml"], color="C0", lw=1.5, label="Sunitinib (central)")
    ax.set_ylabel("ng/mL")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, out["C_metabolite_ng_ml"], color="C1", lw=1.5, label="SU12662 (central)")
    ax.set_ylabel("ng/mL")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, out["ACu_ng_ml"], color="C2", lw=1.5)
    ax.set_ylabel("ACu (ng/mL)")
    ax.set_xlabel("Time (days)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, out["sVEGFR2_ug_L"], color="C3", lw=1.5, label="sVEGFR-2")
    ax.plot(t, out["sVEGFR3_ug_L"], color="C4", lw=1.5, label="sVEGFR-3")
    ax.set_ylabel("ug/L")
    ax.set_xlabel("Time (days)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_vpc_figure(
    vpc: dict[str, Any],
    *,
    t_obs: np.ndarray | None = None,
    y_obs: np.ndarray | None = None,
    y_label: str = "",
    title: str = "Simulation VPC (5ÔÇô95% from Monte Carlo)",
) -> Any:
    """Shaded 90% prediction interval + median; optional observed scatter."""
    plt = _plt()
    t = np.asarray(vpc["t_h"], dtype=float) / 24.0
    p5, p50, p95 = vpc["p5"], vpc["p50"], vpc["p95"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(t, p5, p95, color="C0", alpha=0.25, label="90% sim interval")
    ax.plot(t, p50, color="C0", lw=2.0, label="Sim median")
    ax.plot(t, p5, color="C0", lw=1.0, ls="--", alpha=0.7)
    ax.plot(t, p95, color="C0", lw=1.0, ls="--", alpha=0.7)
    if t_obs is not None and y_obs is not None:
        to = np.asarray(t_obs, dtype=float) / 24.0
        ax.scatter(to, y_obs, c="k", s=12, zorder=5, label="Observed")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(y_label or str(vpc.get("outcome_key", "outcome")))
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_sunitinib_diagnostic_pngs(
    output_dir: str | Path,
    *,
    param_path: str | Path | None = None,
    t_end_weeks: float = 12.0,
    n_eval: int = 301,
    n_sim_vpc: int = 300,
    vpc_outcome: OutcomeKey = "sVEGFR2_ug_L",
    seed: int = 1,
) -> list[Path]:
    """Load paper params, run baseline + VPC, write ``baseline_pkpd.png`` and ``vpc_<outcome>.png``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppath = Path(param_path) if param_path is not None else default_paper_param_paths()[0]
    cfg, spec = load_sunitinib_paper_config(ppath)
    t_end_h = float(t_end_weeks * 7.0 * 24.0)

    base = run_baseline_simulation(cfg, t_end_h=t_end_h, n_eval=n_eval)
    fig_b = plot_baseline_pkpd_figure(base, title=f"Baseline PKPD ({ppath.name})")
    p_baseline = out_dir / "baseline_pkpd.png"
    fig_b.savefig(p_baseline, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig_b)

    vpc = run_simulation_vpc(
        cfg,
        spec,
        n_sim=int(n_sim_vpc),
        outcome_key=vpc_outcome,
        t_end_h=t_end_h,
        n_eval=n_eval,
        seed=seed,
        add_residual=True,
    )
    labels = {
        "sVEGFR2_ug_L": "sVEGFR-2 (ug/L)",
        "sVEGFR3_ug_L": "sVEGFR-3 (ug/L)",
        "C_parent_ng_ml": "Sunitinib central (ng/mL)",
        "C_metabolite_ng_ml": "SU12662 central (ng/mL)",
        "ACu_ng_ml": "ACu (ng/mL)",
    }
    fig_v = plot_vpc_figure(
        vpc,
        y_label=labels.get(vpc_outcome, vpc_outcome),
        title=f"VPC ÔÇö {vpc_outcome} (n={n_sim_vpc})",
    )
    p_vpc = out_dir / f"vpc_{vpc_outcome}.png"
    fig_v.savefig(p_vpc, dpi=150, bbox_inches="tight")
    plt.close(fig_v)

    logger.info("Wrote diagnostic plots to {}", out_dir)
    return [p_baseline, p_vpc]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sunitinib_plots")
    paths = save_sunitinib_diagnostic_pngs(out, n_sim_vpc=200)
    for p in paths:
        print(p.resolve())
