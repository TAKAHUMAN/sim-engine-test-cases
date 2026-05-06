"""
Lenvatinib RCC-adapted integrated PK / biomarker PD / TGI model (Python)

Primary reference: Majid et al. 2024 — integrated PopPK, biomarker indirect-response, and TGI
(RR-DTC development; RCC CL/F covariate 0.851).

PK: 3-compartment; simultaneous ZO + FO from one absorption depot (mass-conserving step cap).
Biomarkers: cumulative AUC (integral of C_central) drives Hill terms.
TGI: interval AUC over trailing window (default 8 weeks, RECIST tumor assessment cadence
    during randomization per Majid et al.) drives tumor Hill; uses biomarker
     trajectories from PK/PD pass (no tumor→biomarker feedback).

Not for clinical decision-making.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PKParams:
    cl_ref_L_per_h: float = 6.28
    v1_ref_L: float = 46.0
    v2_ref_L: float = 28.3
    v3_ref_L: float = 30.9
    q1_ref_L_per_h: float = 3.57
    q2_ref_L_per_h: float = 0.688
    ka_per_h: float = 0.803
    d1_h: float = 1.27
    f1: float = 0.882
    bw_ref_kg: float = 73.2
    rcc_cl_multiplier: float = 0.851


def scale_pk(bw_kg: float, rcc: bool) -> Tuple[float, float, float, float, float, float]:
    bw = float(bw_kg)
    ref = PKParams.bw_ref_kg
    allo = (bw / ref) ** 0.75
    vscale = bw / ref
    cl = PKParams.cl_ref_L_per_h * allo * (PKParams.rcc_cl_multiplier if rcc else 1.0)
    v1 = PKParams.v1_ref_L * vscale
    v2 = PKParams.v2_ref_L * vscale
    v3 = PKParams.v3_ref_L * vscale
    q1 = PKParams.q1_ref_L_per_h * allo
    q2 = PKParams.q2_ref_L_per_h * allo
    return cl, v1, v2, v3, q1, q2


@dataclass(frozen=True)
class BiomarkerSpec:
    name: str
    bm0: float
    mrt_h: float
    gamma: float
    mode: str


BIOMARKERS: List[BiomarkerSpec] = [
    BiomarkerSpec("VEGF", 0.42, 58.3, 1.0, "kout"),
    BiomarkerSpec("Tie-2", 15.1, 354.0, 0.313, "kin"),
    BiomarkerSpec("Ang-2", 3.21, 173.0, 4.27, "kin"),
    BiomarkerSpec("FGF-23", 0.100, 265.0, 1.0, "kout"),
]

SHARED_PD = dict(Emax=0.344, EC50=930.0, DPslope_per_h=2.93e-6)


@dataclass(frozen=True)
class TGIParams:
    kg_per_week: float = 0.00252
    emax_per_week: float = 0.0755
    ec50_auc: float = 1420.0
    lambda_per_week: float = 0.259
    k_tie2_per_week: float = -0.0112
    k_ang2_per_week: float = -0.0144


H_PER_WEEK = 168.0


def conc_ng_ml(a1_mg: float, v1_l: float) -> float:
    return (a1_mg / max(v1_l, 1e-12)) * 1000.0


def _pk_depot_fluxes(
    *,
    a_abs: float,
    t_h: float,
    tau_h: float,
    dose_mg: float,
    f1: float,
    d1_h: float,
    ka: float,
    dt_h: float,
) -> Tuple[float, float]:
    t_rel = t_h - np.floor((t_h + 1e-9) / tau_h) * tau_h
    r_zo = (f1 * dose_mg / d1_h) if (0.0 <= t_rel < d1_h - 1e-12) else 0.0
    r_fo = ka * max(0.0, a_abs)
    r_tot = r_zo + r_fo
    if r_tot <= 0.0:
        return 0.0, 0.0
    max_rate = max(0.0, a_abs) / max(dt_h, 1e-9)
    scale = 1.0 if r_tot <= max_rate else max_rate / r_tot
    return r_zo * scale, r_fo * scale


def euler_pk_biomarker(
    *,
    t0: float,
    t1: float,
    h: float,
    dose_mg: float,
    tau_h: float,
    f1: float,
    d1_h: float,
    ka: float,
    cl: float,
    v1: float,
    v2: float,
    v3: float,
    q1: float,
    q2: float,
    bm0: np.ndarray,
    kout: np.ndarray,
    kin: np.ndarray,
    gamma: np.ndarray,
    mode_kin_mask: np.ndarray,
    emax: float,
    ec50: float,
    dpslope: float,
    legacy_split: bool,
    f_legacy_zo: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit Euler for PK + biomarkers (no tumor). Returns ts, y array [a1,a2,a3,a_abs,auc,b0..b3]."""
    n = int(np.ceil((t1 - t0) / h))
    ts = t0 + h * np.arange(n + 1)
    ys = np.zeros((n + 1, 9))
    a1 = a2 = a3 = 0.0
    a_abs = 0.0
    auc_cum = 0.0
    b = bm0.astype(float).copy()
    last_bucket = -1
    f_zo = float(np.clip(f_legacy_zo, 0.0, 1.0)) if legacy_split else 0.0
    f_fo = max(0.0, 1.0 - f_zo) if legacy_split else 1.0

    for i in range(n + 1):
        t = ts[i]
        bucket = int(np.floor((t + 1e-6) / tau_h))
        if bucket != last_bucket:
            if legacy_split:
                a_abs += f1 * f_fo * dose_mg
            else:
                a_abs += f1 * dose_mg
            last_bucket = bucket

        c_ng = conc_ng_ml(a1, v1)
        ys[i, :] = np.array([a1, a2, a3, a_abs, auc_cum, b[0], b[1], b[2], b[3]], dtype=float)

        if i == n:
            break

        if legacy_split:
            t_rel = t - np.floor((t + 1e-9) / tau_h) * tau_h
            r_zo = (f1 * f_zo * dose_mg / d1_h) if (0.0 <= t_rel < d1_h - 1e-12) else 0.0
            r_fo = ka * max(0.0, a_abs)
            r_tot = r_zo + r_fo
            max_rate = max(0.0, a_abs) / max(h, 1e-9)
            scale = 1.0 if r_tot <= max_rate else max_rate / max(r_tot, 1e-18)
            flux = r_tot * scale * h
        else:
            rz, rf = _pk_depot_fluxes(
                a_abs=a_abs, t_h=t, tau_h=tau_h, dose_mg=dose_mg, f1=f1, d1_h=d1_h, ka=ka, dt_h=h
            )
            flux = (rz + rf) * h

        a_abs -= flux
        a_abs = max(0.0, a_abs)

        da1 = (
            flux
            - (cl / v1) * a1
            - (q1 / v1) * a1
            + (q1 / v2) * a2
            - (q2 / v1) * a1
            + (q2 / v3) * a3
        )
        da2 = (q1 / v1) * a1 - (q1 / v2) * a2
        da3 = (q2 / v1) * a1 - (q2 / v3) * a3
        a1 += da1 * h
        a2 += da2 * h
        a3 += da3 * h

        auc_cum += c_ng * h

        auc_bm = max(0.0, auc_cum)
        db = np.zeros(4)
        for j in range(4):
            g = gamma[j]
            num = emax * (auc_bm**g)
            den = (ec50**g) + (auc_bm**g)
            hill = num / max(den, 1e-18)
            if mode_kin_mask[j]:
                db[j] = kin[j] * (1.0 - hill) - kout[j] * b[j] + dpslope * t * b[j]
            else:
                db[j] = kin[j] - kout[j] * (1.0 - hill) * b[j] + dpslope * t * b[j]

        b = b + db * h
        b = np.maximum(b, 1e-15)

    return ts, ys


def interval_auc_from_cumulative(auc_cum: np.ndarray, nwin: int) -> np.ndarray:
    """Trailing window: AUC_int[i] = auc_cum[i] - auc_cum[max(0,i-nwin)]."""
    out = np.zeros_like(auc_cum)
    for i in range(len(auc_cum)):
        j0 = max(0, i - nwin)
        out[i] = max(0.0, float(auc_cum[i] - auc_cum[j0]))
    return out


def euler_tumor(
    *,
    ts: np.ndarray,
    h: float,
    auc_interval: np.ndarray,
    b_traj: np.ndarray,
    bm0_tie2: float,
    bm0_ang2: float,
    y0_tumor_mm: float,
    tgi: TGIParams,
) -> np.ndarray:
    """Second pass: tumor ODE driven by precomputed interval AUC and biomarkers."""
    n = len(ts) - 1
    y_t = float(y0_tumor_mm)
    out = np.zeros(n + 1)
    out[0] = y_t
    kg = tgi.kg_per_week / H_PER_WEEK
    lam = tgi.lambda_per_week / H_PER_WEEK
    em = tgi.emax_per_week / H_PER_WEEK
    kt = tgi.k_tie2_per_week / H_PER_WEEK
    k_ang = tgi.k_ang2_per_week / H_PER_WEEK

    for i in range(n):
        t = ts[i]
        auc_tv = max(0.0, float(auc_interval[i]))
        hill_tumor = auc_tv / (tgi.ec50_auc + auc_tv)
        b = b_traj[i]
        r_tie = np.log(max(bm0_tie2 / max(b[1], 1e-15), 1e-12))
        r_ang = np.log(max(bm0_ang2 / max(b[2], 1e-15), 1e-12))
        dy = (
            kg * y_t
            - em * hill_tumor * y_t * np.exp(-lam * t)
            + kt * r_tie * y_t
            + k_ang * r_ang * y_t
        )
        y_t += dy * h
        y_t = max(y_t, 1e-9)
        out[i + 1] = y_t

    return out


def run_scenario(
    *,
    weeks: float,
    dose_mg: float,
    bw_kg: float,
    rcc: bool,
    dt_h: float,
    y0_tumor_mm: float,
    biomarker_dpslope_per_h: float | None,
    tumor_auc_window_weeks: float,
    legacy_pk_split: bool,
    f_legacy_zo: float,
) -> pd.DataFrame:
    cl, v1, v2, v3, q1, q2 = scale_pk(bw_kg, rcc=rcc)
    f1 = PKParams.f1
    ka = PKParams.ka_per_h
    d1 = PKParams.d1_h
    tau_h = 24.0

    bm0 = np.array([s.bm0 for s in BIOMARKERS], dtype=float)
    mrt = np.array([s.mrt_h for s in BIOMARKERS], dtype=float)
    kout = 1.0 / mrt
    kin = kout * bm0
    gamma = np.array([s.gamma for s in BIOMARKERS], dtype=float)
    mode_kin_mask = np.array([1.0 if s.mode == "kin" else 0.0 for s in BIOMARKERS], dtype=float)

    t_end = weeks * H_PER_WEEK
    dpslope_use = 0.0 if biomarker_dpslope_per_h is None else float(biomarker_dpslope_per_h)
    win_h = max(float(tumor_auc_window_weeks), 1e-6) * H_PER_WEEK
    nwin = max(1, int(np.round(win_h / dt_h)))

    ts, ypk = euler_pk_biomarker(
        t0=0.0,
        t1=t_end,
        h=dt_h,
        dose_mg=dose_mg,
        tau_h=tau_h,
        f1=f1,
        d1_h=d1,
        ka=ka,
        cl=cl,
        v1=v1,
        v2=v2,
        v3=v3,
        q1=q1,
        q2=q2,
        bm0=bm0,
        kout=kout,
        kin=kin,
        gamma=gamma,
        mode_kin_mask=mode_kin_mask,
        emax=SHARED_PD["Emax"],
        ec50=SHARED_PD["EC50"],
        dpslope=dpslope_use,
        legacy_split=legacy_pk_split,
        f_legacy_zo=f_legacy_zo,
    )

    a1 = ypk[:, 0]
    auc = ypk[:, 4]
    b = ypk[:, 5:9]
    auc_tumor = interval_auc_from_cumulative(auc, nwin)
    y_tum = euler_tumor(
        ts=ts,
        h=dt_h,
        auc_interval=auc_tumor,
        b_traj=b,
        bm0_tie2=float(BIOMARKERS[1].bm0),
        bm0_ang2=float(BIOMARKERS[2].bm0),
        y0_tumor_mm=y0_tumor_mm,
        tgi=TGIParams(),
    )

    c_ng = conc_ng_ml(a1, v1)
    pct_bm = 100.0 * (b / bm0 - 1.0)
    pct_tumor = 100.0 * (y_tum / y0_tumor_mm - 1.0)
    t_weeks = ts / H_PER_WEEK

    return pd.DataFrame(
        {
            "time_h": ts,
            "time_weeks": t_weeks,
            "C_ng_mL": c_ng,
            "AUC_cum_biomarker_ng_h_mL": auc,
            "AUC_interval_tumor_ng_h_mL": auc_tumor,
            "tumor_mm": y_tum,
            "pct_change_tumor": pct_tumor,
            "pct_change_VEGF": pct_bm[:, 0],
            "pct_change_Tie2": pct_bm[:, 1],
            "pct_change_Ang2": pct_bm[:, 2],
            "pct_change_FGF23": pct_bm[:, 3],
        }
    )


def _interp_at_week(df: pd.DataFrame, col: str, week: float) -> float:
    s = df[["time_weeks", col]].dropna()
    return float(np.interp(week, s["time_weeks"].to_numpy(), s[col].to_numpy()))


def validation_summary(df: pd.DataFrame) -> dict:
    return {
        "vegf_pct_change_2w": _interp_at_week(df, "pct_change_VEGF", 2.0),
        "fgf23_pct_change_8w": _interp_at_week(df, "pct_change_FGF23", 8.0),
        "ang2_pct_change_4w": _interp_at_week(df, "pct_change_Ang2", 4.0),
        "tie2_pct_change_8w": _interp_at_week(df, "pct_change_Tie2", 8.0),
        "tumor_pct_change_52w": float(df.iloc[-1]["pct_change_tumor"]),
    }


def _week_first_frac_of_reference(
    df: pd.DataFrame, col: str, ref_week: float, frac: float, *, up: bool
) -> float:
    """
    First week where response reaches `frac` of the interpolated value at `ref_week`.
    For upregulated markers (up=True), first time v >= frac*v_ref.
    For downregulated (up=False), first time v <= frac*v_ref (v_ref negative).
    """
    v_ref = _interp_at_week(df, col, ref_week)
    if np.isnan(v_ref) or abs(v_ref) < 1e-9:
        return float("nan")
    for _, row in df.iterrows():
        w = float(row["time_weeks"])
        if w > ref_week + 0.25:
            break
        v = float(row[col])
        if up and v_ref > 0 and v >= frac * v_ref:
            return w
        if (not up) and v_ref < 0 and v <= frac * v_ref:
            return w
    return float("nan")


def paper_figure4_timing_metrics(df: pd.DataFrame) -> dict:
    """
    Fig. 4 timing proxies (Majid et al.): under monotone cumulative-AUC PD, argmax week often
    sits at a scan-window edge; we report week of first ≥95%% of the response at a late
    reference week (plateau / nadir depth).
    """
    return {
        "vegf_plateau95_week": _week_first_frac_of_reference(
            df, "pct_change_VEGF", ref_week=6.0, frac=0.95, up=True
        ),
        "fgf23_plateau95_week": _week_first_frac_of_reference(
            df, "pct_change_FGF23", ref_week=12.0, frac=0.95, up=True
        ),
        "ang2_nadir95_week": _week_first_frac_of_reference(
            df, "pct_change_Ang2", ref_week=10.0, frac=0.95, up=False
        ),
        "tie2_nadir95_week": _week_first_frac_of_reference(
            df, "pct_change_Tie2", ref_week=12.0, frac=0.95, up=False
        ),
        "vegf_pct_2w": _interp_at_week(df, "pct_change_VEGF", 2.0),
        "fgf23_pct_8w": _interp_at_week(df, "pct_change_FGF23", 8.0),
        "ang2_pct_4w": _interp_at_week(df, "pct_change_Ang2", 4.0),
        "tie2_pct_8w": _interp_at_week(df, "pct_change_Tie2", 8.0),
    }


def plot_figure4_style(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    tw = df["time_weeks"].to_numpy()

    axes[0, 0].plot(tw, df["pct_change_VEGF"], label="VEGF")
    axes[0, 0].plot(tw, df["pct_change_FGF23"], label="FGF-23")
    axes[0, 0].axhline(0.0, color="k", lw=0.5)
    axes[0, 0].set_title("Upregulated biomarkers (% change)")
    axes[0, 0].set_xlabel("Weeks")
    axes[0, 0].set_ylabel("% change from baseline")
    axes[0, 0].legend()

    axes[0, 1].plot(tw, df["pct_change_Ang2"], label="Ang-2")
    axes[0, 1].plot(tw, df["pct_change_Tie2"], label="Tie-2")
    axes[0, 1].axhline(0.0, color="k", lw=0.5)
    axes[0, 1].set_title("Downregulated biomarkers (% change)")
    axes[0, 1].set_xlabel("Weeks")
    axes[0, 1].set_ylabel("% change from baseline")
    axes[0, 1].legend()

    axes[1, 0].plot(tw, df["pct_change_tumor"], color="C3")
    axes[1, 0].axhline(-35.0, color="C3", ls="--", lw=1, label="~35% shrink reference")
    axes[1, 0].axhline(0.0, color="k", lw=0.5)
    axes[1, 0].set_title("Tumor size (sum of diameters, % change)")
    axes[1, 0].set_xlabel("Weeks")
    axes[1, 0].set_ylabel("% change from baseline")
    axes[1, 0].legend()

    axr = axes[1, 1]
    axr.plot(tw, df["C_ng_mL"], color="C2", label="C (ng/mL)")
    axr.set_xlabel("Weeks")
    axr.set_ylabel("ng/mL", color="C2")
    axr.tick_params(axis="y", labelcolor="C2")
    ax2 = axr.twinx()
    ax2.plot(tw, df["AUC_interval_tumor_ng_h_mL"], color="C4", ls="--", alpha=0.85, label="TGI interval AUC")
    ax2.set_ylabel("Interval AUC (ng*h/mL)", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")
    axr.set_title("Concentration + tumor-driver interval AUC")
    lines, labels = axr.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axr.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_assumptions(path: Path) -> None:
    path.write_text(
        """Assumptions (Majid et al. 2024 alignment)
A1. PK Table 1; allometry as published; RCC CL/F * 0.851.
A2. Default PK absorption: single depot per dose (F1*dose), simultaneous nominal ZO (F1*dose/D1
    during D1) and FO (Ka*depot), mass-conserving per Euler step. Legacy: --pk-legacy-split.
A3. Biomarker Hill uses cumulative AUC = integral C_central dt from t=0 on the same time grid.
A4. TGI Hill uses trailing interval AUC (difference in cumulative AUC over
    --tumor-auc-window-weeks; default 8 wk per RECIST schedule in Majid et al.), distinct from
    biomarker cumulative AUC.
A5. Tumor is integrated in a second pass using PK/PD outputs (no tumor→biomarker feedback).
A6. DPslope default 0 for long treated-only runs; optional --biomarker-dpslope.
A7. Euler integration; default --dt-h 0.1 h for publication-style stability (Ang-2 Hill γ=4.27).
""",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Lenvatinib RCC PK/PD/TGI (Majid et al. 2024)")
    p.add_argument("--weeks", type=float, default=52.0)
    p.add_argument("--dose-mg", type=float, default=24.0)
    p.add_argument("--bw-kg", type=float, default=73.2)
    p.add_argument("--no-rcc-cl", action="store_true")
    p.add_argument(
        "--dt-h",
        type=float,
        default=0.1,
        help="Euler step (h); 0.1 recommended (Majid-style stiff biomarker γ); 0.5 often <0.5%% drift",
    )
    p.add_argument("--y0-tumor-mm", type=float, default=70.2, help="Baseline SLD (mm); 70.2 for >35%% shrink caption")
    p.add_argument(
        "--tumor-auc-window-weeks",
        type=float,
        default=8.0,
        help="Trailing AUC window for TGI (weeks); 8 = RECIST assessment interval in randomization phase",
    )
    p.add_argument("--results-dir", type=str, default=str(SCRIPT_DIR / "results"))
    p.add_argument("--biomarker-dpslope", type=float, default=None)
    p.add_argument("--pk-legacy-split", action="store_true")
    p.add_argument("--f-zero-order", type=float, default=0.40, help="Legacy ZO fraction with --pk-legacy-split")
    args = p.parse_args()

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    results = Path(args.results_dir)
    (results / "figures").mkdir(parents=True, exist_ok=True)

    common_kw = dict(
        weeks=float(args.weeks),
        dose_mg=float(args.dose_mg),
        rcc=not args.no_rcc_cl,
        dt_h=float(args.dt_h),
        biomarker_dpslope_per_h=(None if args.biomarker_dpslope is None else float(args.biomarker_dpslope)),
        tumor_auc_window_weeks=float(args.tumor_auc_window_weeks),
        legacy_pk_split=bool(args.pk_legacy_split),
        f_legacy_zo=float(args.f_zero_order),
    )

    df = run_scenario(
        bw_kg=float(args.bw_kg),
        y0_tumor_mm=float(args.y0_tumor_mm),
        **common_kw,
    )

    csv_path = results / "lenvatinib_rcc_timeseries.csv"
    df.to_csv(csv_path, index=False)
    fig_path = results / "figures" / "figure4_style_pk_pd_tgi.png"
    plot_figure4_style(df, fig_path)

    # Fig. 4 biomarker timing: medians from studies 303+211 (Majid et al. caption)
    df_fig4 = run_scenario(bw_kg=75.1, y0_tumor_mm=59.5, **common_kw)
    fig4 = paper_figure4_timing_metrics(df_fig4)

    summ = validation_summary(df)
    lines = [
        "Majid et al. 2024 CPT PSP 2024;13(6):954-969 — validation vs Figure 4 narrative",
        "",
        "A) Primary exported run (CLI: BW, baseline tumor, interval window)",
        f"    BW={args.bw_kg} kg, baseline tumor={args.y0_tumor_mm} mm, TGI AUC window={args.tumor_auc_window_weeks} wk",
        f"    VEGF % change @ 2 wk: {summ['vegf_pct_change_2w']:.2f}%",
        f"    FGF-23 % change @ 8 wk: {summ['fgf23_pct_change_8w']:.2f}%",
        f"    Ang-2 % change @ 4 wk: {summ['ang2_pct_change_4w']:.2f}%",
        f"    Tie-2 % change @ 8 wk: {summ['tie2_pct_change_8w']:.2f}%",
        f"    Tumor % change @ 52 wk: {summ['tumor_pct_change_52w']:.2f}%",
        "",
        "B) Figure 4 biomarker profile (BW=75.1 kg, baseline tumor=59.5 mm; same dose/window/dt)",
        f"    VEGF % @ 2 wk: {fig4['vegf_pct_2w']:.2f}% (paper near-peak ~2 wk)",
        f"    FGF-23 % @ 8 wk: {fig4['fgf23_pct_8w']:.2f}% (paper near-peak ~8 wk)",
        f"    Ang-2 % @ 4 wk: {fig4['ang2_pct_4w']:.2f}% (paper near-nadir ~4 wk)",
        f"    Tie-2 % @ 8 wk: {fig4['tie2_pct_8w']:.2f}% (paper near-nadir ~8 wk)",
        f"    VEGF week of >=95% of wk-6 response: {fig4['vegf_plateau95_week']:.2f} wk (proxy ~2 wk)",
        f"    FGF-23 week of >=95% of wk-12 response: {fig4['fgf23_plateau95_week']:.2f} wk (proxy ~8 wk)",
        f"    Ang-2 week of <=95% depth vs wk-10 nadir: {fig4['ang2_nadir95_week']:.2f} wk (proxy ~4 wk)",
        f"    Tie-2 week of <=95% depth vs wk-12 nadir: {fig4['tie2_nadir95_week']:.2f} wk (proxy ~8 wk)",
        "",
        "C) >35% shrink caption (73.2 kg, 70.2 mm baseline, 24 mg QD) — dedicated check",
    ]
    df_shrink = run_scenario(
        weeks=float(args.weeks),
        dose_mg=24.0,
        bw_kg=73.2,
        y0_tumor_mm=70.2,
        rcc=common_kw["rcc"],
        dt_h=common_kw["dt_h"],
        biomarker_dpslope_per_h=common_kw["biomarker_dpslope_per_h"],
        tumor_auc_window_weeks=common_kw["tumor_auc_window_weeks"],
        legacy_pk_split=common_kw["legacy_pk_split"],
        f_legacy_zo=common_kw["f_legacy_zo"],
    )
    shrink_pct = float(df_shrink.iloc[-1]["pct_change_tumor"])
    lines.append(f"    Tumor % change @ 52 wk: {shrink_pct:.2f}% (paper > -35%)")
    lines.extend(
        [
            "",
            "Pass/fail (informational; primary run):",
            f"  VEGF @ 2 wk in band: {35.0 <= summ['vegf_pct_change_2w'] <= 58.0}",
            f"  FGF-23 @ 8 wk in band: {45.0 <= summ['fgf23_pct_change_8w'] <= 56.0}",
            f"  Ang-2 @ 4 wk in band: {-36.0 <= summ['ang2_pct_change_4w'] <= -30.0}",
            f"  Tie-2 drop by 8 wk: {summ['tie2_pct_change_8w'] <= -22.0}",
            f"  Tumor shrink >= 35% (primary run): {summ['tumor_pct_change_52w'] <= -35.0}",
            "",
            "Figure 4 timing / magnitude (BW 75.1 kg, 59.5 mm; informational):",
            f"  VEGF % @ 2 wk: {35.0 <= fig4['vegf_pct_2w'] <= 58.0}",
            f"  FGF-23 % @ 8 wk: {45.0 <= fig4['fgf23_pct_8w'] <= 56.0}",
            f"  Ang-2 % @ 4 wk: {-36.0 <= fig4['ang2_pct_4w'] <= -30.0}",
            f"  Tie-2 % @ 8 wk: {fig4['tie2_pct_8w'] <= -22.0}",
            f"  VEGF plateau95 week ~2 wk: {1.0 <= fig4['vegf_plateau95_week'] <= 4.5}",
            f"  FGF-23 plateau95 week ~8 wk: {6.0 <= fig4['fgf23_plateau95_week'] <= 11.0}",
            f"  Ang-2 nadir95 week ~4 wk: {2.5 <= fig4['ang2_nadir95_week'] <= 6.5}",
            f"  Tie-2 nadir95 week ~8 wk: {6.0 <= fig4['tie2_nadir95_week'] <= 11.0}",
            f"  Shrink caption (73.2 kg, 70.2 mm): {shrink_pct <= -35.0}",
        ]
    )
    chk = results / "validation_checks.txt"
    chk.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_assumptions(results / "assumptions_block.txt")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {chk}")
    print(f"Wrote: {results / 'assumptions_block.txt'}")


if __name__ == "__main__":
    main()
