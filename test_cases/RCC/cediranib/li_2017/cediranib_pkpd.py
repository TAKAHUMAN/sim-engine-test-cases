"""
Cediranib (AZD2171) PK/PD model.

This script implements:
1. Rat-to-human allometric scaling checks.
2. A 2-compartment oral human PK model with finite zero-order input into
   the absorption depot followed by first-order absorption.
3. VEGFR-2 Emax inhibition from free plasma concentration.
4. An indirect-response sVEGFR-2 biomarker model.
5. Steady-state dose comparisons for 15, 20, and 30 mg qd.

Outputs are written next to this script under ./results and ./results/figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# Constants
MW_CEDIRANIB = 450.5  # g/mol
FU_HUMAN = 0.05

IC50_VEGFR2_NM = 0.5
IC50_VEGFR1_NM = 5.0
IC50_VEGFR3_NM = 3.0

BW_HUMAN = 73.0  # kg
BW_RAT = 0.25  # kg
BW_MOUSE = 0.02  # kg

CL_RAT_PER_KG = 0.8  # L/h/kg
T_HALF_RAT = 9.0  # h
F_RAT = 0.60
FU_RAT = 0.06
VD_RAT_PER_KG = (CL_RAT_PER_KG * T_HALF_RAT) / 0.693

OBS_HUMAN_CL = 15.8  # L/h, CL/F 26.3 L/h corrected by F ~= 0.60
OBS_HUMAN_VD = 421.0  # L, Vss/F 702 L corrected by F ~= 0.60
OBS_HUMAN_T_HALF = 24.0  # h

SIM_DAYS = 14
TAU = 24.0
T_END = SIM_DAYS * TAU
DT = 0.02
TIME_GRID = np.arange(0.0, T_END + DT, DT)

CASE_DIR = Path(__file__).resolve().parent
OUT_DIR = CASE_DIR / "results"
FIG_DIR = OUT_DIR / "figures"


@dataclass(frozen=True)
class PKParams:
    cl: float = 26.3  # L/h, apparent CL/F
    vc: float = 489.0  # L, apparent Vc/F
    vp: float = 213.0  # L, apparent Vp/F
    q: float = 11.8  # L/h, apparent Q/F
    ka: float = 2.70  # 1/h
    d1: float = 1.68  # h
    fu: float = FU_HUMAN
    bw: float = BW_HUMAN
    age: float = 59.0

    def adjusted(self) -> "PKParams":
        cl_adj = self.cl * (self.age / 59.0) ** (-0.409) * (self.bw / 73.0) ** 0.517
        vc_adj = self.vc * (self.bw / 73.0) ** 0.65
        return PKParams(
            cl=cl_adj,
            vc=vc_adj,
            vp=self.vp,
            q=self.q,
            ka=self.ka,
            d1=self.d1,
            fu=self.fu,
            bw=self.bw,
            age=self.age,
        )


def nM_to_mg_per_l(conc_nm: float) -> float:
    return conc_nm * MW_CEDIRANIB * 1e-6


def mg_per_l_to_nM(conc_mg_l: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(conc_mg_l) * 1e6 / MW_CEDIRANIB


def free_mg_per_l_to_nM(conc_free_mg_l: np.ndarray | float) -> np.ndarray | float:
    return mg_per_l_to_nM(conc_free_mg_l)


def total_mg_per_l_to_free_nM(conc_total_mg_l: np.ndarray | float, fu: float = FU_HUMAN) -> np.ndarray | float:
    return mg_per_l_to_nM(np.asarray(conc_total_mg_l) * fu)


def inhibition_from_free_nM(cp_free_nm: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    cp = np.maximum(np.asarray(cp_free_nm), 0.0)
    return 100.0 * cp**gamma / (IC50_VEGFR2_NM**gamma + cp**gamma)


def allometric_scaling() -> pd.DataFrame:
    """Dimensionally consistent scaling from total rat CL and Vd."""
    cl_rat_total = CL_RAT_PER_KG * BW_RAT
    vd_rat_total = VD_RAT_PER_KG * BW_RAT

    bw_ratio = BW_HUMAN / BW_RAT
    cl_pred = cl_rat_total * bw_ratio**0.75
    vd_pred = vd_rat_total * bw_ratio**1.0
    t_half_pred = (vd_pred / cl_pred) * 0.693

    rows = [
        ("CL (L/h)", cl_pred, OBS_HUMAN_CL),
        ("Vd (L)", vd_pred, OBS_HUMAN_VD),
        ("t1/2 (h)", t_half_pred, OBS_HUMAN_T_HALF),
    ]
    return pd.DataFrame(
        {
            "Parameter": [row[0] for row in rows],
            "Rat (scaled)": [row[1] for row in rows],
            "Human (observed)": [row[2] for row in rows],
            "% Error": [100.0 * (row[1] - row[2]) / row[2] for row in rows],
        }
    )


def active_zero_order_input(t: float, dose_mg: float, d1: float) -> float:
    if dose_mg == 0.0:
        return 0.0

    local_t = t % TAU
    if 0.0 <= local_t < d1:
        return dose_mg / d1
    return 0.0


def simulate_pk(dose_mg: float, params: PKParams | None = None) -> pd.DataFrame:
    params = (params or PKParams()).adjusted()

    def rhs(t: float, y: np.ndarray) -> list[float]:
        a_abs, a_central, a_peripheral = y
        zero_in = active_zero_order_input(t, dose_mg, params.d1)
        first_order_in = params.ka * a_abs

        d_a_abs = zero_in - first_order_in
        d_a_central = (
            first_order_in
            - (params.cl / params.vc) * a_central
            - (params.q / params.vc) * a_central
            + (params.q / params.vp) * a_peripheral
        )
        d_a_peripheral = (params.q / params.vc) * a_central - (params.q / params.vp) * a_peripheral
        return [d_a_abs, d_a_central, d_a_peripheral]

    solution = solve_ivp(
        rhs,
        (0.0, T_END),
        [0.0, 0.0, 0.0],
        t_eval=TIME_GRID,
        rtol=1e-7,
        atol=1e-10,
        method="LSODA",
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    cp_total = solution.y[1] / params.vc
    cp_free_mg_l = cp_total * params.fu
    cp_free_nm = total_mg_per_l_to_free_nM(cp_total, params.fu)
    inh_g1 = inhibition_from_free_nM(cp_free_nm, gamma=1.0)
    inh_g2 = inhibition_from_free_nM(cp_free_nm, gamma=2.0)

    return pd.DataFrame(
        {
            "time_h": solution.t,
            "day": solution.t / 24.0,
            "A_abs_mg": solution.y[0],
            "A_central_mg": solution.y[1],
            "A_peripheral_mg": solution.y[2],
            "Cp_total_mg_L": cp_total,
            "Cp_free_mg_L": cp_free_mg_l,
            "Cp_free_nM": cp_free_nm,
            "Inh_gamma1_pct": inh_g1,
            "Inh_gamma2_pct": inh_g2,
        }
    )


def steady_state_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["time_h"] >= T_END - TAU) & (df["time_h"] <= T_END)].copy()


def time_above(values: np.ndarray, threshold: float, times: np.ndarray) -> float:
    return float(np.trapezoid((values > threshold).astype(float), times))


def pk_summary(df: pd.DataFrame) -> dict[str, float]:
    ss = steady_state_window(df)
    t_ss = ss["time_h"].to_numpy() - (T_END - TAU)
    cp_total = ss["Cp_total_mg_L"].to_numpy()
    cp_free_mg_l = ss["Cp_free_mg_L"].to_numpy()
    cp_free_nm = ss["Cp_free_nM"].to_numpy()
    inh = ss["Inh_gamma1_pct"].to_numpy()

    return {
        "AUCss_mg_h_L": float(np.trapezoid(cp_total, t_ss)),
        "AUCss_free_mg_h_L": float(np.trapezoid(cp_free_mg_l, t_ss)),
        "Cmax_ss_mg_L": float(cp_total.max()),
        "Cmin_ss_mg_L": float(cp_total.min()),
        "Cmax_ss_free_mg_L": float(cp_free_mg_l.max()),
        "Cmin_ss_free_mg_L": float(cp_free_mg_l.min()),
        "Cmax_ss_free_nM": float(cp_free_nm.max()),
        "Cmin_ss_free_nM": float(cp_free_nm.min()),
        "Mean_inh_pct": float(inh.mean()),
        "Min_inh_pct": float(inh.min()),
        "Max_inh_pct": float(inh.max()),
        "Time_gt_50_inh_h": time_above(inh, 50.0, t_ss),
        "Time_gt_90_inh_h": time_above(inh, 90.0, t_ss),
        "Time_gt_IC50_h": time_above(cp_free_nm, IC50_VEGFR2_NM, t_ss),
    }


def terminal_half_life(params: PKParams | None = None) -> float:
    params = (params or PKParams()).adjusted()
    k10 = params.cl / params.vc
    k12 = params.q / params.vc
    k21 = params.q / params.vp
    beta = 0.5 * ((k10 + k12 + k21) - np.sqrt((k10 + k12 + k21) ** 2 - 4.0 * k10 * k21))
    return float(np.log(2.0) / beta)


def simulate_svegfr2(df: pd.DataFrame, baseline_pg_ml: float = 10000.0) -> pd.DataFrame:
    kout_mouse = 0.693 / (3.0 * 24.0)
    kout_human = kout_mouse * (BW_HUMAN / BW_MOUSE) ** (-0.25)
    kin_human = kout_human * baseline_pg_ml
    t_ref = df["time_h"].to_numpy()
    inh_ref = df["Inh_gamma1_pct"].to_numpy() / 100.0

    def inh_at(t: float) -> float:
        return float(np.interp(t, t_ref, inh_ref))

    def treated_rhs(t: float, y: np.ndarray) -> list[float]:
        return [kin_human * (1.0 - inh_at(t)) - kout_human * y[0]]

    def untreated_rhs(_t: float, y: np.ndarray) -> list[float]:
        return [kin_human - kout_human * y[0]]

    treated = solve_ivp(
        treated_rhs,
        (0.0, T_END),
        [baseline_pg_ml],
        t_eval=TIME_GRID,
        rtol=1e-8,
        atol=1e-6,
        method="LSODA",
    )
    untreated = solve_ivp(
        untreated_rhs,
        (0.0, T_END),
        [baseline_pg_ml],
        t_eval=TIME_GRID,
        rtol=1e-8,
        atol=1e-6,
        method="LSODA",
    )
    if not treated.success:
        raise RuntimeError(treated.message)
    if not untreated.success:
        raise RuntimeError(untreated.message)

    return pd.DataFrame(
        {
            "time_h": TIME_GRID,
            "day": TIME_GRID / 24.0,
            "sVEGFR2_treated_pg_mL": treated.y[0],
            "sVEGFR2_untreated_pg_mL": untreated.y[0],
            "kout_human_h_inv": kout_human,
        }
    )


def setup_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5.5),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
        }
    )


def add_ic50_lines_mg_l(ax: plt.Axes) -> None:
    for value_nm, label, color in [
        (IC50_VEGFR2_NM, "VEGFR-2 IC50 0.5 nM", "#c0392b"),
        (IC50_VEGFR3_NM, "VEGFR-3 IC50 3 nM", "#8e44ad"),
        (IC50_VEGFR1_NM, "VEGFR-1 IC50 5 nM", "#2c3e50"),
    ]:
        ax.axhline(nM_to_mg_per_l(value_nm), color=color, linestyle="--", linewidth=1.2, label=label)


def add_ic50_lines_nM(ax: plt.Axes) -> None:
    for value_nm, label, color in [
        (IC50_VEGFR2_NM, "VEGFR-2 IC50 0.5 nM", "#c0392b"),
        (IC50_VEGFR3_NM, "VEGFR-3 IC50 3 nM", "#8e44ad"),
        (IC50_VEGFR1_NM, "VEGFR-1 IC50 5 nM", "#2c3e50"),
    ]:
        ax.axhline(value_nm, color=color, linestyle="--", linewidth=1.2, label=label)


def save_current_fig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def make_figures(
    pk20: pd.DataFrame,
    svegfr2: pd.DataFrame,
    dose_results: dict[int, pd.DataFrame],
    allometry: pd.DataFrame,
) -> None:
    setup_plot_style()

    fig, ax = plt.subplots()
    ax.plot(pk20["day"], pk20["Cp_total_mg_L"], label="Total Cp", color="#1f77b4", linewidth=2)
    ax.plot(pk20["day"], pk20["Cp_free_mg_L"], label="Free Cp", color="#d62728", linewidth=2)
    add_ic50_lines_mg_l(ax)
    ax.set_title("Cediranib 20 mg qd: 14-Day Plasma Concentration Profile")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.legend(loc="upper right")
    save_current_fig("figure_1_pk_14day_total_free.png")

    ss20 = steady_state_window(pk20)
    fig, ax = plt.subplots()
    ax.plot(ss20["time_h"] - (T_END - TAU), ss20["Cp_total_mg_L"], label="Total Cp", color="#1f77b4", linewidth=2)
    ax.plot(ss20["time_h"] - (T_END - TAU), ss20["Cp_free_mg_L"], label="Free Cp", color="#d62728", linewidth=2)
    add_ic50_lines_mg_l(ax)
    ax.set_title("Cediranib 20 mg qd: Steady-State Concentration Profile")
    ax.set_xlabel("Time after day 13 dose (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.legend(loc="upper right")
    save_current_fig("figure_2_pk_steady_state_total_free.png")

    fig, ax = plt.subplots()
    ax.plot(pk20["day"], pk20["Inh_gamma1_pct"], label="gamma = 1", color="#117864", linewidth=2)
    ax.plot(pk20["day"], pk20["Inh_gamma2_pct"], label="gamma = 2 sensitivity", color="#f39c12", linewidth=2)
    ax.axhline(50, color="#c0392b", linestyle="--", linewidth=1.2, label="50% inhibition")
    ax.axhline(90, color="#2c3e50", linestyle="--", linewidth=1.2, label="90% inhibition")
    ax.set_title("VEGFR-2 Inhibition Over 14 Days: Cediranib 20 mg qd")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("VEGFR-2 inhibition (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    save_current_fig("figure_3_vegfr2_inhibition_14day.png")

    fig, ax = plt.subplots()
    ax.plot(ss20["time_h"] - (T_END - TAU), ss20["Inh_gamma1_pct"], label="gamma = 1", color="#117864", linewidth=2)
    ax.plot(ss20["time_h"] - (T_END - TAU), ss20["Inh_gamma2_pct"], label="gamma = 2 sensitivity", color="#f39c12", linewidth=2)
    ax.axhline(50, color="#c0392b", linestyle="--", linewidth=1.2, label="50% inhibition")
    ax.axhline(90, color="#2c3e50", linestyle="--", linewidth=1.2, label="90% inhibition")
    ax.set_title("VEGFR-2 Inhibition at Steady State: Cediranib 20 mg qd")
    ax.set_xlabel("Time after day 13 dose (h)")
    ax.set_ylabel("VEGFR-2 inhibition (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    save_current_fig("figure_4_vegfr2_inhibition_steady_state.png")

    fig, ax = plt.subplots()
    ax.plot(
        svegfr2["day"],
        svegfr2["sVEGFR2_treated_pg_mL"],
        label="20 mg qd",
        color="#7d3c98",
        linewidth=2,
    )
    ax.plot(
        svegfr2["day"],
        svegfr2["sVEGFR2_untreated_pg_mL"],
        label="Untreated",
        color="#34495e",
        linewidth=2,
    )
    ax.axhline(10000, color="#c0392b", linestyle="--", linewidth=1.2, label="Baseline 10,000 pg/mL")
    ax.set_title("sVEGFR-2 Indirect Response: Treated vs Untreated")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("sVEGFR-2 (pg/mL)")
    ax.legend(loc="upper right")
    save_current_fig("figure_5_svegfr2_treated_vs_untreated.png")

    fig, ax = plt.subplots()
    colors = {15: "#1f77b4", 20: "#117864", 30: "#d35400"}
    for dose, df in dose_results.items():
        ss = steady_state_window(df)
        ax.plot(
            ss["time_h"] - (T_END - TAU),
            ss["Cp_free_nM"],
            label=f"{dose} mg qd",
            color=colors[dose],
            linewidth=2,
        )
    add_ic50_lines_nM(ax)
    ax.set_title("Steady-State Free Cediranib by Dose")
    ax.set_xlabel("Time after day 13 dose (h)")
    ax.set_ylabel("Free concentration (nM)")
    ax.legend(loc="upper right")
    save_current_fig("figure_6_dose_comparison_free_cp.png")

    fig, ax = plt.subplots()
    for dose, df in dose_results.items():
        ss = steady_state_window(df)
        ax.plot(
            ss["time_h"] - (T_END - TAU),
            ss["Inh_gamma1_pct"],
            label=f"{dose} mg qd",
            color=colors[dose],
            linewidth=2,
        )
    ax.axhline(50, color="#c0392b", linestyle="--", linewidth=1.2, label="50% inhibition")
    ax.axhline(90, color="#2c3e50", linestyle="--", linewidth=1.2, label="90% inhibition")
    ax.set_title("Steady-State VEGFR-2 Inhibition by Dose")
    ax.set_xlabel("Time after day 13 dose (h)")
    ax.set_ylabel("VEGFR-2 inhibition (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    save_current_fig("figure_7_dose_comparison_vegfr2_inhibition.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(allometry))
    width = 0.36
    pred = allometry["Rat (scaled)"].to_numpy()
    obs = allometry["Human (observed)"].to_numpy()
    bars1 = ax.bar(x - width / 2, pred, width, label="Predicted from rat", color="#2874a6")
    bars2 = ax.bar(x + width / 2, obs, width, label="Observed human", color="#85929e")
    for idx, error in enumerate(allometry["% Error"]):
        y = max(pred[idx], obs[idx])
        ax.text(idx, y * 1.04, f"{error:+.1f}%", ha="center", va="bottom", fontsize=10)
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
    ax.set_title("Rat-to-Human Allometric Scaling Check")
    ax.set_ylabel("Value in parameter units")
    ax.set_xticks(x)
    ax.set_xticklabels(allometry["Parameter"])
    ax.legend(loc="upper left")
    save_current_fig("figure_8_allometric_scaling_summary.png")


def build_dose_table(dose_results: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dose, df in dose_results.items():
        ss = steady_state_window(df)
        summary = pk_summary(df)
        rows.append(
            {
                "Dose": f"{dose} mg",
                "Cmax_free (nM)": summary["Cmax_ss_free_nM"],
                "Cmin_free (nM)": summary["Cmin_ss_free_nM"],
                "Mean_Inh%": float(ss["Inh_gamma1_pct"].mean()),
                "Time>IC50 (h/24h)": summary["Time_gt_IC50_h"],
            }
        )
    return pd.DataFrame(rows)


def value_at_time(df: pd.DataFrame, column: str, time_h: float) -> float:
    return float(np.interp(time_h, df["time_h"].to_numpy(), df[column].to_numpy()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    allometry = allometric_scaling()
    pk20 = simulate_pk(20.0)
    dose_results = {dose: simulate_pk(float(dose)) for dose in [15, 20, 30]}
    svegfr2 = simulate_svegfr2(pk20)

    make_figures(pk20, svegfr2, dose_results, allometry)

    allometry.to_csv(OUT_DIR / "allometric_scaling.csv", index=False)
    pk20.to_csv(OUT_DIR / "pkpd_20mg_timeseries.csv", index=False)
    svegfr2.to_csv(OUT_DIR / "svegfr2_timeseries.csv", index=False)
    dose_table = build_dose_table(dose_results)
    dose_table.to_csv(OUT_DIR / "dose_comparison.csv", index=False)

    summary20 = pk_summary(pk20)
    t_half = terminal_half_life()

    day7 = value_at_time(svegfr2, "sVEGFR2_treated_pg_mL", 7.0 * 24.0)
    day14 = value_at_time(svegfr2, "sVEGFR2_treated_pg_mL", 14.0 * 24.0)
    reduction_day7 = 100.0 * (10000.0 - day7) / 10000.0
    reduction_day14 = 100.0 * (10000.0 - day14) / 10000.0

    print("\nALLOMETRIC SCALING VERIFICATION")
    print(allometry.to_string(index=False, formatters={
        "Rat (scaled)": "{:.2f}".format,
        "Human (observed)": "{:.2f}".format,
        "% Error": "{:+.1f}".format,
    }))

    print("\n20 mg qd STEADY-STATE PK")
    pk_print = pd.DataFrame(
        [
            ("AUCss total (mg*h/L)", summary20["AUCss_mg_h_L"]),
            ("AUCss free (mg*h/L)", summary20["AUCss_free_mg_h_L"]),
            ("Cmax total (mg/L)", summary20["Cmax_ss_mg_L"]),
            ("Cmin total (mg/L)", summary20["Cmin_ss_mg_L"]),
            ("Cmax free (mg/L)", summary20["Cmax_ss_free_mg_L"]),
            ("Cmin free (mg/L)", summary20["Cmin_ss_free_mg_L"]),
            ("Cmax free (nM)", summary20["Cmax_ss_free_nM"]),
            ("Cmin free (nM)", summary20["Cmin_ss_free_nM"]),
        ],
        columns=["Metric", "Value"],
    )
    print(pk_print.to_string(index=False, formatters={"Value": "{:.4f}".format}))

    print("\n20 mg qd STEADY-STATE VEGFR-2 INHIBITION")
    print(f"Mean inhibition (%): {summary20['Mean_inh_pct']:.2f}")
    print(f"Min inhibition (%) : {summary20['Min_inh_pct']:.2f}")
    print(f"Max inhibition (%) : {summary20['Max_inh_pct']:.2f}")

    print("\nsVEGFR-2 RESPONSE")
    print(f"Day 7 treated (pg/mL) : {day7:.1f} ({reduction_day7:.1f}% reduction)")
    print(f"Day 14 treated (pg/mL): {day14:.1f} ({reduction_day14:.1f}% reduction)")

    print("\nDOSE COMPARISON")
    print(dose_table.to_string(index=False, formatters={
        "Cmax_free (nM)": "{:.2f}".format,
        "Cmin_free (nM)": "{:.2f}".format,
        "Mean_Inh%": "{:.1f}".format,
        "Time>IC50 (h/24h)": "{:.1f}".format,
    }))

    cl_pred = float(allometry.loc[allometry["Parameter"] == "CL (L/h)", "Rat (scaled)"].iloc[0])
    cl_error = float(allometry.loc[allometry["Parameter"] == "CL (L/h)", "% Error"].iloc[0])

    print("\n============================================")
    print("CEDIRANIB PKPD MODEL SUMMARY (20 mg qd)")
    print("============================================")
    print("PK (Steady State):")
    print(f"  AUCss (mg*h/L)     : {summary20['AUCss_mg_h_L']:.3f}")
    print(f"  Cmax,ss (mg/L)     : {summary20['Cmax_ss_mg_L']:.4f}")
    print(f"  Cmin,ss (mg/L)     : {summary20['Cmin_ss_mg_L']:.4f}")
    print(f"  Cmax,ss free (nM)  : {summary20['Cmax_ss_free_nM']:.2f}")
    print(f"  Cmin,ss free (nM)  : {summary20['Cmin_ss_free_nM']:.2f}")
    print(f"  t1/2 (h)           : {t_half:.1f}")
    print("")
    print("PD (VEGFR-2 Inhibition):")
    print(f"  Mean inhibition (%) : {summary20['Mean_inh_pct']:.1f}")
    print(f"  Min inhibition (%)  : {summary20['Min_inh_pct']:.1f}")
    print(f"  Max inhibition (%)  : {summary20['Max_inh_pct']:.1f}")
    print(f"  Time > 50% Inh (h) : {summary20['Time_gt_50_inh_h']:.1f}")
    print(f"  Time > 90% Inh (h) : {summary20['Time_gt_90_inh_h']:.1f}")
    print("")
    print("sVEGFR-2 Biomarker:")
    print("  Baseline (pg/mL)   : 10,000")
    print(f"  Day 7 (pg/mL)      : {day7:.0f}")
    print(f"  Day 14 (pg/mL)     : {day14:.0f}")
    print(f"  % reduction day 14 : {reduction_day14:.1f}")
    print("")
    print("Allometric Scaling:")
    print(f"  Predicted CL (L/h) : {cl_pred:.2f}")
    print("  Observed CL (L/h)  : ~15.8")
    print(f"  % Error CL         : {cl_error:+.1f}")
    print("============================================")

    print("\nSaved outputs:")
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(CASE_DIR)}")


if __name__ == "__main__":
    main()
