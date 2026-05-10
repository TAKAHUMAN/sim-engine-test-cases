"""
Deterministic mean-data PKPD model for sapanisertib (TAK-228).

Source numbers are limited to the values supplied in the prompt from
Voss et al. BJC 2020, Supplementary Table 5 and visual reads of
Supplementary Figure 5. No population PK, random effects, or simulation of
individual patients is used.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.stats import t


BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "results"
FIGDIR = OUTDIR / "figures"
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)


# Fixed PK parameters from the prompt.
KA = 1.2  # 1/h
CL_OVER_F = 19.0  # L/h
VD_OVER_F = 200.0  # L
KE = CL_OVER_F / VD_OVER_F  # 1/h
F_REL = 1.0

TIME_END_H = 672.0
DT_H = 0.1
TIME_GRID = np.round(np.arange(0.0, TIME_END_H + DT_H, DT_H), 10)


PK_C1D1 = pd.DataFrame(
    {
        "dose_mg": [2, 4, 6, 7, 9, 10, 12, 13, 15, 16, 20, 30, 40],
        "tmax_h": [2.0, 2.0, 1.0, 1.5, 1.1, 2.8, 2.0, 2.1, 2.0, 2.1, 2.1, 1.0, 2.4],
        "cmax_ng_ml": [13.5, 19.1, 50.8, 46.7, 75.9, 48.4, 99.5, 93.6, 56.7, 66.7, 154.1, 161.8, 172.4],
        "auc0_24_ng_h_ml": [np.nan, 178.4, 354.3, 327.3, 595.9, 341.8, 730.2, 952.6, 517.8, 688.7, 1262.8, 1076.8, 1639.5],
        "half_life_h": [np.nan, 7.1, 6.8, 6.9, 7.5, 6.4, 7.4, 8.7, 9.4, 7.0, 6.5, 5.9, 7.6],
    }
)

PK_C2D1 = pd.DataFrame(
    {
        "schedule": ["QD", "QD", "QD", "QW", "QW"],
        "dose_mg": [4, 6, 7, 30, 40],
        "auc0_24_c2d1": [281.4, 327.0, 350.6, 1120.9, 2222.1],
        "auc0_24_c1d1": [178.4, 354.3, 327.3, 1076.8, 1639.5],
        "observed_ratio": [1.58, 0.92, 1.07, 1.04, 1.35],
    }
)


PD_DATA = {
    "pS6": {
        "median": ([5, 12.5, 25, 50, 100, 160, 220], [-60, -75, -85, -90, -90, -90, -95]),
        "scatter": (
            [5, 5, 5, 12.5, 12.5, 12.5, 25, 25, 25, 50, 50, 50, 100, 100, 160, 160, 160, 220],
            [-100, -60, 55, -100, -75, 25, -100, -85, 30, -100, -90, 85, -100, -40, -100, -90, 195, -95],
        ),
        "p0": [90, 12, 1.2],
    },
    "p4EBP1": {
        "median": ([5, 15, 40, 170], [-70, -95, -100, -100]),
        "scatter": (
            [5, 5, 5, 15, 15, 15, 40, 40, 100, 170, 240, 170],
            [-100, -70, 60, -100, -95, -90, -100, -100, -100, -100, -100, -5],
        ),
        "p0": [100, 6, 1.8],
    },
    "pNDRG1": {
        "median": ([5, 15, 45, 132.5, 220], [-65, -90, -95, -97, -95]),
        "scatter": (
            [5, 5, 5, 15, 15, 45, 45, 45, 100, 132.5, 165, 165, 220],
            [-100, -65, 20, -85, -95, -100, -90, 100, -95, -100, -95, 145, -95],
        ),
        "p0": [100, 8, 1.5],
    },
    "pPRAS40": {
        "median": ([5, 25, 60, 110, 165, 230], [-55, -82.5, -90, -92.5, -95, -90]),
        "scatter": (
            [5, 5, 5, 25, 25, 25, 60, 60, 110, 110, 120, 165, 165, 230],
            [-100, -55, 30, -90, -82.5, 30, -90, 25, -95, -10, 145, 145, -95, -90],
        ),
        "p0": [90, 15, 1.0],
    },
}


def pk_rhs(_t, y):
    ad, ac = y
    return [-KA * ad, KA * ad - KE * ac]


def concentration_from_amount(ac_mg):
    return ac_mg * 1000.0 / VD_OVER_F


def dose_times(schedule):
    if schedule == "QD":
        days = range(1, 29)
    elif schedule == "QW":
        days = [1, 8, 15, 22]
    elif schedule == "QD3dQW":
        days = [d for wk in [1, 8, 15, 22] for d in range(wk, wk + 3)]
    elif schedule == "QD5dQW":
        days = [d for wk in [1, 8, 15, 22] for d in range(wk, wk + 5)]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return np.array([(day - 1) * 24.0 for day in days], dtype=float)


def simulate_pk(dose_mg, schedule, t_grid=TIME_GRID):
    dosing = dose_times(schedule)
    y = np.array([0.0, 0.0])
    all_t = []
    all_y = []
    current = 0.0

    event_times = list(dosing) + [float(t_grid[-1])]
    for event_time in event_times:
        seg_t = t_grid[(t_grid >= current) & (t_grid <= event_time)]
        if event_time > current and len(seg_t) > 0:
            if seg_t[0] != current:
                seg_t = np.insert(seg_t, 0, current)
            sol = solve_ivp(pk_rhs, (current, event_time), y, t_eval=seg_t, rtol=1e-8, atol=1e-10)
            keep = np.ones(sol.t.shape, dtype=bool)
            if all_t:
                keep[0] = False
            all_t.extend(sol.t[keep])
            all_y.extend(sol.y[:, keep].T)
            y = sol.y[:, -1]
        if event_time in dosing:
            y = y.copy()
            y[0] += dose_mg * F_REL
        current = event_time

    result = pd.DataFrame(all_y, columns=["ad_mg", "ac_mg"])
    result.insert(0, "time_h", np.array(all_t))
    result["concentration_ng_ml"] = concentration_from_amount(result["ac_mg"])
    return result


def simulate_single_dose(dose_mg, duration_h=24.0):
    t_grid = np.round(np.arange(0.0, duration_h + DT_H, DT_H), 10)
    y0 = [dose_mg * F_REL, 0.0]
    sol = solve_ivp(pk_rhs, (0.0, duration_h), y0, t_eval=t_grid, rtol=1e-8, atol=1e-10)
    df = pd.DataFrame({"time_h": sol.t, "ad_mg": sol.y[0], "ac_mg": sol.y[1]})
    df["concentration_ng_ml"] = concentration_from_amount(df["ac_mg"])
    return df


def auc_linear(t_h, y):
    return float(np.trapezoid(y, t_h))


def inhibitory_emax(c, emax, ec50, hill):
    c = np.asarray(c, dtype=float)
    return -emax * np.power(c, hill) / (np.power(ec50, hill) + np.power(c, hill))


def fit_pd_models():
    rows = []
    fits = {}
    for biomarker, spec in PD_DATA.items():
        x, y = [np.array(v, dtype=float) for v in spec["median"]]
        popt, pcov = curve_fit(
            inhibitory_emax,
            x,
            y,
            p0=spec["p0"],
            bounds=([0.0, 0.01, 0.05], [100.0, 250.0, 8.0]),
            maxfev=100000,
        )
        pred = inhibitory_emax(x, *popt)
        residual = y - pred
        rmse = float(np.sqrt(np.mean(residual**2)))
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        dof = max(0, len(x) - len(popt))
        if dof > 0 and np.all(np.isfinite(pcov)):
            tcrit = t.ppf(0.975, dof)
            se = np.sqrt(np.diag(pcov))
            ci_low = popt - tcrit * se
            ci_high = popt + tcrit * se
        else:
            ci_low = np.full_like(popt, np.nan)
            ci_high = np.full_like(popt, np.nan)

        fits[biomarker] = {"params": popt, "pcov": pcov, "r2": r2, "rmse": rmse}
        rows.append(
            {
                "biomarker": biomarker,
                "emax_pct": popt[0],
                "emax_ci_low": ci_low[0],
                "emax_ci_high": ci_high[0],
                "ec50_ng_ml": popt[1],
                "ec50_ci_low": ci_low[1],
                "ec50_ci_high": ci_high[1],
                "hill": popt[2],
                "hill_ci_low": ci_low[2],
                "hill_ci_high": ci_high[2],
                "r2": r2,
                "rmse_pct": rmse,
                "n_median_bins": len(x),
                "fit_note": "R2/RMSE are based on binned median values, not raw scatter; true uncertainty is larger.",
            }
        )
    return fits, pd.DataFrame(rows)


def validate_pk():
    rows = []
    for dose in PK_C1D1["dose_mg"]:
        df = simulate_single_dose(dose)
        c = df["concentration_ng_ml"].to_numpy()
        t_h = df["time_h"].to_numpy()
        pred_auc = auc_linear(t_h, c)
        rows.append(
            {
                "dose_mg": dose,
                "pred_cmax_ng_ml": float(np.max(c)),
                "pred_tmax_h": float(t_h[np.argmax(c)]),
                "pred_auc0_24_ng_h_ml": pred_auc,
                "pred_auc0_inf_ng_h_ml": dose * 1000.0 / CL_OVER_F,
            }
        )
    val = PK_C1D1.merge(pd.DataFrame(rows), on="dose_mg", how="left")
    val["cmax_pct_error"] = 100.0 * (val["pred_cmax_ng_ml"] - val["cmax_ng_ml"]) / val["cmax_ng_ml"]
    val["within_15pct"] = val["cmax_pct_error"].abs() <= 15.0
    val["auc0_24_pct_error"] = 100.0 * (val["pred_auc0_24_ng_h_ml"] - val["auc0_24_ng_h_ml"]) / val["auc0_24_ng_h_ml"]
    val["auc0_inf_pct_error"] = 100.0 * (val["pred_auc0_inf_ng_h_ml"] - val["auc0_24_ng_h_ml"]) / val["auc0_24_ng_h_ml"]
    return val


def end_interval_time(schedule):
    dosing = dose_times(schedule)
    if schedule == "QD":
        return dosing[-1] + 24.0
    if schedule == "QW":
        return dosing[-1] + 168.0
    return TIME_END_H


def concentration_at_time(sim, time_h):
    return float(np.interp(time_h, sim["time_h"], sim["concentration_ng_ml"]))


def accumulation_table():
    rows = []
    for _, row in PK_C2D1.iterrows():
        sim = simulate_pk(row["dose_mg"], row["schedule"])
        c1 = sim[(sim["time_h"] >= 0.0) & (sim["time_h"] <= 24.0)]
        c22 = sim[(sim["time_h"] >= 504.0) & (sim["time_h"] <= 528.0)].copy()
        c22["time_since_dose_h"] = c22["time_h"] - 504.0
        pred_c1_auc = auc_linear(c1["time_h"], c1["concentration_ng_ml"])
        pred_c2_auc = auc_linear(c22["time_since_dose_h"], c22["concentration_ng_ml"])
        rows.append(
            {
                "schedule": row["schedule"],
                "dose_mg": row["dose_mg"],
                "observed_ratio": row["observed_ratio"],
                "pred_auc0_24_c1d1": pred_c1_auc,
                "pred_auc0_24_c2d1": pred_c2_auc,
                "pred_ratio": pred_c2_auc / pred_c1_auc,
            }
        )
    return pd.DataFrame(rows)


def summarize_pkpd(fits):
    scenarios = [
        ("5 mg QD", 5, "QD"),
        ("30 mg QW", 30, "QW"),
        ("30 mg QD3dQW", 30, "QD3dQW"),
        ("30 mg QD5dQW", 30, "QD5dQW"),
    ]
    rows = []
    for label, dose, schedule in scenarios:
        sim = simulate_pk(dose, schedule)
        t_h = sim["time_h"].to_numpy()
        c = sim["concentration_ng_ml"].to_numpy()
        trough_time = end_interval_time(schedule)
        ctrough = concentration_at_time(sim, trough_time)
        for biomarker, fit in fits.items():
            e = inhibitory_emax(c, *fit["params"])
            rows.append(
                {
                    "scenario": label,
                    "dose_mg": dose,
                    "schedule": schedule,
                    "biomarker": biomarker,
                    "cmax_ng_ml": float(np.max(c)),
                    "cavg_ng_ml": auc_linear(t_h, c) / TIME_END_H,
                    "ctrough_ng_ml": ctrough,
                    "trough_time_h": trough_time,
                    "ft_gt_ec50_pct": float(100.0 * np.mean(c > fit["params"][1])),
                    "auec_pct_h": auc_linear(t_h, -e),
                }
            )
    return pd.DataFrame(rows)


def ec50_sensitivity(fits):
    scenarios = [("5 mg QD", 5, "QD"), ("30 mg QW", 30, "QW")]
    ec50_values = [3.0, 6.0, 10.0, 15.0]
    biomarkers = ["p4EBP1", "pNDRG1"]
    rows = []
    for label, dose, schedule in scenarios:
        sim = simulate_pk(dose, schedule)
        t_h = sim["time_h"].to_numpy()
        c = sim["concentration_ng_ml"].to_numpy()
        for biomarker in biomarkers:
            emax, _ec50, hill = fits[biomarker]["params"]
            for ec50 in ec50_values:
                e = inhibitory_emax(c, emax, ec50, hill)
                rows.append(
                    {
                        "scenario": label,
                        "dose_mg": dose,
                        "schedule": schedule,
                        "biomarker": biomarker,
                        "ec50_assumed_ng_ml": ec50,
                        "ft_gt_ec50_pct": float(100.0 * np.mean(c > ec50)),
                        "auec_pct_h": auc_linear(t_h, -e),
                    }
                )
    return pd.DataFrame(rows)


def qd_dose_coverage(fits):
    fitted_ec50 = fits["p4EBP1"]["params"][1]
    rows = []
    for dose in PK_C1D1["dose_mg"].to_numpy(dtype=float):
        sim = simulate_pk(dose, "QD")
        c = sim["concentration_ng_ml"].to_numpy()
        rows.append(
            {
                "dose_mg": dose,
                "ec50_ng_ml": fitted_ec50,
                "cmax_ng_ml": float(np.max(c)),
                "cavg_ng_ml": auc_linear(sim["time_h"], c) / TIME_END_H,
                "ctrough_ng_ml": concentration_at_time(sim, 672.0),
                "ft_gt_ec50_pct": float(100.0 * np.mean(c > fitted_ec50)),
            }
        )
    return pd.DataFrame(rows)


def make_plots(fits, pd_params, pk_validation, pkpd_summary, sensitivity):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    auc_rows = pk_validation.dropna(subset=["auc0_24_ng_h_ml"]).copy()
    axes[0].scatter(auc_rows["auc0_24_ng_h_ml"], auc_rows["pred_auc0_24_ng_h_ml"], s=55, color="#2f6f9f", label="AUC0-24")
    lim = [0, max(auc_rows["auc0_24_ng_h_ml"].max(), auc_rows["pred_auc0_24_ng_h_ml"].max()) * 1.1]
    axes[0].plot(lim, lim, color="black", lw=1)
    axes[0].plot(lim, [0.95 * lim[0], 0.95 * lim[1]], color="#777777", ls="--", lw=1, label="+/-5%")
    axes[0].plot(lim, [1.05 * lim[0], 1.05 * lim[1]], color="#777777", ls="--", lw=1)
    for _, row in auc_rows.iterrows():
        axes[0].annotate(f"{int(row['dose_mg'])} mg", (row["auc0_24_ng_h_ml"], row["pred_auc0_24_ng_h_ml"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    axes[0].set_xlabel("Observed AUC0-24 (ng*h/mL)")
    axes[0].set_ylabel("Predicted AUC0-24 (ng*h/mL)")
    axes[0].set_title("Primary PK validation: AUC")
    axes[0].legend()
    axes[1].scatter(pk_validation["cmax_ng_ml"], pk_validation["pred_cmax_ng_ml"], s=55, color="#c44e52")
    cmax_lim = [0, max(pk_validation["cmax_ng_ml"].max(), pk_validation["pred_cmax_ng_ml"].max()) * 1.1]
    axes[1].plot(cmax_lim, cmax_lim, color="black", lw=1)
    axes[1].plot(cmax_lim, [0.85 * cmax_lim[0], 0.85 * cmax_lim[1]], color="#777777", ls="--", lw=1, label="+/-15%")
    axes[1].plot(cmax_lim, [1.15 * cmax_lim[0], 1.15 * cmax_lim[1]], color="#777777", ls="--", lw=1)
    axes[1].set_xlabel("Observed Cmax (ng/mL)")
    axes[1].set_ylabel("Predicted Cmax (ng/mL)")
    axes[1].set_title("Cmax diagnostic: variable small-n means")
    axes[1].legend()
    fig.suptitle(f"PK validation: Ka={KA:.2f} 1/h, Cl/F={CL_OVER_F:.1f} L/h, Vd/F={VD_OVER_F:.0f} L")
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_pk_auc_primary_cmax_diagnostic.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, dose, schedule, color in [("5 mg QD", 5, "QD", "#2f6f9f"), ("30 mg QW", 30, "QW", "#c44e52")]:
        sim = simulate_pk(dose, schedule)
        ax.plot(sim["time_h"] / 24.0, sim["concentration_ng_ml"], label=label, color=color)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Concentration (ng/mL)")
    ax.set_title(f"28-day PK profiles (Ka={KA:.2f}, Cl/F={CL_OVER_F:.1f} L/h, Vd/F={VD_OVER_F:.0f} L)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_pk_profiles_28d.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for dose in [4, 6, 9, 12, 20, 30, 40]:
        sim = simulate_single_dose(dose)
        ax.plot(sim["time_h"], sim["concentration_ng_ml"], label=f"{dose} mg")
        obs = PK_C1D1.loc[PK_C1D1["dose_mg"] == dose].iloc[0]
        ax.scatter(obs["tmax_h"], obs["cmax_ng_ml"], s=28, marker="x", color=ax.lines[-1].get_color())
    ax.set_xlabel("Cycle 1 Day 1 time (h)")
    ax.set_ylabel("Concentration (ng/mL)")
    ax.set_title("Cycle 1 Day 1 profiles with observed mean Cmax markers")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_c1d1_profiles_observed_cmax.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    x_line = np.linspace(0.01, 240.0, 500)
    for ax, biomarker in zip(axes.flat, ["p4EBP1", "pS6", "pNDRG1", "pPRAS40"]):
        scatter_x, scatter_y = PD_DATA[biomarker]["scatter"]
        med_x, med_y = PD_DATA[biomarker]["median"]
        pars = fits[biomarker]["params"]
        ax.scatter(scatter_x, scatter_y, s=28, color="#999999", alpha=0.75, label="Approx scatter")
        ax.scatter(med_x, med_y, s=42, color="#2f6f9f", label="Median bins")
        ax.plot(x_line, inhibitory_emax(x_line, *pars), color="#c44e52", lw=2, label="Fit")
        ax.axhline(-pars[0] / 2.0, color="#777777", lw=0.8, ls=":")
        ax.set_title(f"{biomarker}: Emax={pars[0]:.1f}%, EC50={pars[1]:.1f} ng/mL, gamma={pars[2]:.2f}\nR2={fits[biomarker]['r2']:.3f}, RMSE={fits[biomarker]['rmse']:.1f}%")
        ax.set_xlim(0, 240)
        ax.set_ylim(-110, 210)
    for ax in axes[-1, :]:
        ax.set_xlabel("Concentration (ng/mL)")
    for ax in axes[:, 0]:
        ax.set_ylabel("% change from baseline")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGDIR / "04_emax_fits_all_biomarkers.png", dpi=200)
    plt.close(fig)

    for biomarker, filename, title in [
        ("p4EBP1", "05_pkpd_p4ebp1_timecourse.png", "TORC1 inhibition time course: p4EBP1"),
        ("pNDRG1", "06_pkpd_pndrg1_timecourse.png", "TORC2 inhibition time course: pNDRG1"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        pars = fits[biomarker]["params"]
        for label, dose, schedule, color in [("5 mg QD", 5, "QD", "#2f6f9f"), ("30 mg QW", 30, "QW", "#c44e52")]:
            sim = simulate_pk(dose, schedule)
            e = inhibitory_emax(sim["concentration_ng_ml"], *pars)
            ax.plot(sim["time_h"] / 24.0, e, label=label, color=color)
        ax.axhline(-pars[0] / 2.0, color="black", ls="--", lw=1, label=f"EC50 effect ({-pars[0] / 2:.1f}%)")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("% change from baseline")
        ax.set_title(f"{title}; Emax={pars[0]:.1f}%, EC50={pars[1]:.1f} ng/mL, gamma={pars[2]:.2f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGDIR / filename, dpi=200)
        plt.close(fig)

    rec = pkpd_summary.copy()
    rec = rec[rec["scenario"].isin(["5 mg QD", "30 mg QW", "30 mg QD3dQW", "30 mg QD5dQW"])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    pivot_auec = rec.pivot(index="scenario", columns="biomarker", values="auec_pct_h")
    pivot_ft = rec.pivot(index="scenario", columns="biomarker", values="ft_gt_ec50_pct")
    pivot_auec.plot(kind="bar", ax=axes[0], width=0.82)
    pivot_ft.plot(kind="bar", ax=axes[1], width=0.82)
    axes[0].set_ylabel("AUEC (% inhibition*h)")
    axes[1].set_ylabel("ft > EC50 (%)")
    axes[1].set_xlabel("Dose/schedule")
    axes[0].set_title("PKPD summary across schedules; dose labels use recommended examples")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)
    axes[1].legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_summary_bar_auec_ft_ec50.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(PK_C1D1["dose_mg"], pk_validation["pred_cmax_ng_ml"], label="Predicted Cmax", color="#2f6f9f")
    axes[0].scatter(PK_C1D1["dose_mg"], PK_C1D1["cmax_ng_ml"], label="Observed Cmax", color="#c44e52", marker="x")
    axes[0].set_xlabel("Dose (mg)")
    axes[0].set_ylabel("Cmax (ng/mL)")
    axes[0].set_title("Dose proportionality: Cmax")
    axes[0].legend()
    axes[1].scatter(PK_C1D1["dose_mg"], pk_validation["pred_auc0_24_ng_h_ml"], label="Predicted AUC0-24", color="#2f6f9f")
    axes[1].scatter(PK_C1D1["dose_mg"], PK_C1D1["auc0_24_ng_h_ml"], label="Observed AUC0-24", color="#c44e52", marker="x")
    axes[1].set_xlabel("Dose (mg)")
    axes[1].set_ylabel("AUC0-24 (ng*h/mL)")
    axes[1].set_title("Dose proportionality: AUC")
    axes[1].legend()
    fig.suptitle(f"Linear PK model dose proportionality, 2-40 mg; Ka={KA:.2f}, Ke={KE:.3f}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "08_dose_proportionality_cmax_auc.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, biomarker, title in [
        (axes[0], "p4EBP1", "Single-interval TORC1 rebound: p4EBP1"),
        (axes[1], "pNDRG1", "Single-interval TORC2 rebound: pNDRG1"),
    ]:
        pars = fits[biomarker]["params"]
        qd = simulate_pk(5, "QD")
        qd_interval = qd[(qd["time_h"] >= 648.0) & (qd["time_h"] <= 672.0)].copy()
        qd_interval["interval_h"] = qd_interval["time_h"] - 648.0
        qw = simulate_pk(30, "QW")
        qw_interval = qw[(qw["time_h"] >= 504.0) & (qw["time_h"] <= 672.0)].copy()
        qw_interval["interval_h"] = qw_interval["time_h"] - 504.0
        ax.plot(qd_interval["interval_h"], inhibitory_emax(qd_interval["concentration_ng_ml"], *pars), color="#2f6f9f", label="5 mg QD, 0-24h")
        ax.plot(qw_interval["interval_h"], inhibitory_emax(qw_interval["concentration_ng_ml"], *pars), color="#c44e52", label="30 mg QW, 0-168h")
        ax.axhline(-pars[0] / 2.0, color="black", ls="--", lw=1, label=f"EC50 effect ({-pars[0] / 2:.1f}%)")
        ax.set_xlabel("Hours after dose")
        ax.set_title(f"{title}\nEmax={pars[0]:.1f}%, EC50={pars[1]:.1f} ng/mL, gamma={pars[2]:.2f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("% change from baseline")
    fig.tight_layout()
    fig.savefig(FIGDIR / "09_single_interval_inhibition_rebound.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, biomarker in zip(axes, ["p4EBP1", "pNDRG1"]):
        sub = sensitivity[sensitivity["biomarker"] == biomarker]
        for scenario, color in [("5 mg QD", "#2f6f9f"), ("30 mg QW", "#c44e52")]:
            row = sub[sub["scenario"] == scenario]
            ax.plot(row["ec50_assumed_ng_ml"], row["ft_gt_ec50_pct"], marker="o", color=color, label=scenario)
        ax.set_xlabel("Assumed EC50 (ng/mL)")
        ax.set_title(f"EC50 sensitivity: {biomarker}")
        ax.set_ylim(0, 105)
        ax.legend()
    axes[0].set_ylabel("ft > EC50 (%)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "10_ec50_sensitivity_ft_ec50.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    auc_rows = pk_validation.dropna(subset=["auc0_24_ng_h_ml"]).copy()
    auc_rows["observed_auc_per_mg"] = auc_rows["auc0_24_ng_h_ml"] / auc_rows["dose_mg"]
    auc_rows["pred_auc0_inf_per_mg"] = auc_rows["pred_auc0_inf_ng_h_ml"] / auc_rows["dose_mg"]
    ax.scatter(auc_rows["dose_mg"], auc_rows["observed_auc_per_mg"], color="#c44e52", s=55, label="Observed AUC0-24/dose")
    ax.axhline(1000.0 / CL_OVER_F, color="#2f6f9f", lw=2, label=f"Linear model AUCinf/dose = {1000.0 / CL_OVER_F:.1f}")
    ax.plot(auc_rows["dose_mg"], auc_rows["pred_auc0_24_ng_h_ml"] / auc_rows["dose_mg"], color="#2f6f9f", ls="--", label="Predicted AUC0-24/dose")
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel("Dose-normalized AUC (ng*h/mL per mg)")
    ax.set_title("Dose-normalized AUC supports approximate linear exposure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "11_dose_normalized_auc.png", dpi=200)
    plt.close(fig)

    qd_doses = PK_C1D1["dose_mg"].to_numpy(dtype=float)
    biomarker = "p4EBP1"
    fitted_ec50 = fits[biomarker]["params"][1]
    dose_rows = []
    for dose in qd_doses:
        sim = simulate_pk(dose, "QD")
        c = sim["concentration_ng_ml"].to_numpy()
        dose_rows.append(
            {
                "dose_mg": dose,
                "ft_gt_fitted_ec50_pct": float(100.0 * np.mean(c > fitted_ec50)),
                "ctrough_ng_ml": concentration_at_time(sim, 672.0),
            }
        )
    dose_df = pd.DataFrame(dose_rows)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(dose_df["dose_mg"], dose_df["ft_gt_fitted_ec50_pct"], marker="o", color="#2f6f9f", label="ft > p4EBP1 EC50")
    ax1.set_xlabel("QD dose (mg)")
    ax1.set_ylabel("ft > EC50 over 28 days (%)", color="#2f6f9f")
    ax1.tick_params(axis="y", labelcolor="#2f6f9f")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.plot(dose_df["dose_mg"], dose_df["ctrough_ng_ml"], marker="s", color="#c44e52", label="Ctrough")
    ax2.axhline(fitted_ec50, color="#333333", ls="--", lw=1.2, label=f"p4EBP1 EC50 = {fitted_ec50:.2f} ng/mL")
    ax2.set_ylabel("QD Ctrough (ng/mL)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    ax1.set_title("QD dose transition to sustained mTOR pathway coverage")
    fig.tight_layout()
    fig.savefig(FIGDIR / "12_qd_ft_ec50_vs_dose.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
    pars = fits["p4EBP1"]["params"]
    for label, dose, schedule, color in [("5 mg QD", 5, "QD", "#2f6f9f"), ("30 mg QW", 30, "QW", "#c44e52")]:
        sim = simulate_pk(dose, schedule)
        e = inhibitory_emax(sim["concentration_ng_ml"], *pars)
        axes[0].plot(sim["time_h"] / 24.0, sim["concentration_ng_ml"], color=color, label=label)
        axes[1].plot(sim["time_h"] / 24.0, e, color=color, label=label)
    axes[0].axhline(pars[1], color="#333333", ls="--", lw=1.1, label=f"p4EBP1 EC50 = {pars[1]:.2f} ng/mL")
    axes[1].axhline(-pars[0] / 2.0, color="#333333", ls="--", lw=1.1, label=f"EC50 effect = {-pars[0] / 2:.1f}%")
    axes[0].set_ylabel("Concentration (ng/mL)")
    axes[1].set_ylabel("p4EBP1 % change")
    axes[1].set_xlabel("Time (days)")
    axes[0].set_title("PK profile drives schedule-dependent p4EBP1 inhibition")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "13_combined_pk_pd_story_p4ebp1.png", dpi=200)
    plt.close(fig)


def main():
    fits, pd_params = fit_pd_models()
    pk_validation = validate_pk()
    acc = accumulation_table()
    pkpd_summary = summarize_pkpd(fits)
    sensitivity = ec50_sensitivity(fits)
    qd_coverage = qd_dose_coverage(fits)

    comparison = pkpd_summary[
        (pkpd_summary["scenario"].isin(["5 mg QD", "30 mg QW"]))
        & (pkpd_summary["biomarker"].isin(["p4EBP1", "pNDRG1"]))
    ].copy()

    pk_validation.to_csv(OUTDIR / "pk_validation.csv", index=False)
    acc.to_csv(OUTDIR / "accumulation_check.csv", index=False)
    pd_params.to_csv(OUTDIR / "pd_fitted_parameters.csv", index=False)
    pkpd_summary.to_csv(OUTDIR / "pkpd_summary.csv", index=False)
    comparison.to_csv(OUTDIR / "comparison_5mg_qd_vs_30mg_qw.csv", index=False)
    sensitivity.to_csv(OUTDIR / "ec50_sensitivity.csv", index=False)
    qd_coverage.to_csv(OUTDIR / "qd_dose_coverage.csv", index=False)

    make_plots(fits, pd_params, pk_validation, pkpd_summary, sensitivity)

    print("\nFixed PK parameters")
    print(pd.DataFrame([{"Ka_1_h": KA, "Cl_F_L_h": CL_OVER_F, "Vd_F_L": VD_OVER_F, "Ke_1_h": KE, "F_relative": F_REL}]).to_string(index=False))
    print("\nPK validation, AUC primary and Cmax diagnostic")
    print(pk_validation[["dose_mg", "auc0_24_ng_h_ml", "pred_auc0_24_ng_h_ml", "auc0_24_pct_error", "cmax_ng_ml", "pred_cmax_ng_ml", "cmax_pct_error"]].round(3).to_string(index=False))
    print("\nAccumulation check")
    print(acc.round(3).to_string(index=False))
    print("\nFitted PD parameters with 95% CI, R2, RMSE")
    print(pd_params.round(4).to_string(index=False))
    print("\nComparison: 5 mg QD vs 30 mg QW")
    print(comparison.round(3).to_string(index=False))
    print("\nEC50 sensitivity")
    print(sensitivity.round(3).to_string(index=False))
    print("\nQD dose coverage using fitted p4EBP1 EC50")
    print(qd_coverage.round(3).to_string(index=False))
    print(f"\nFiles written to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
