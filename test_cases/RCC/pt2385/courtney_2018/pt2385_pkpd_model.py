"""
Semi-mechanistic PKPD forward simulation for PT2385.

This script intentionally uses only the numerical values stated in the prompt.
No fitting, random effects, tumor dynamics, anemia, or hemoglobin model is used.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq


# =============================================================================
# Section 1: Parameter definitions
# =============================================================================

DRUG_NAME = "PT2385"

MG_TO_UG = 1000.0  # Unit conversion: 1 mg = 1000 ug
HOURS_PER_DAY = 24.0  # Unit conversion

DOSE_LEVELS_MG = np.array([100, 200, 400, 800, 1200, 1800], dtype=float)  # SOURCE: Courtney et al. 2018, JCO
TAU_H = 12.0  # SOURCE: Courtney et al. 2018, JCO
SIMULATION_HOURS = 336.0  # Prompt-specified duration: 14 days
N_DOSES = int(SIMULATION_HOURS / TAU_H)  # 28 BID doses over 14 days
POINTS_PER_DOSING_INTERVAL = 241

TMAX_TARGET_H = 2.0  # SOURCE: Courtney et al. 2018, JCO
CMAX_800_DAY15_TARGET_UG_PER_ML = 3.1  # SOURCE: Courtney et al. 2018, JCO
HALF_LIFE_H = 17.0  # SOURCE: Courtney et al. 2018, JCO
AUC_ACCUMULATION_RATIO_TARGET = 2.5  # SOURCE: Courtney et al. 2018, JCO
CMAX_800_SINGLE_DOSE_ESTIMATED_UG_PER_ML = (
    CMAX_800_DAY15_TARGET_UG_PER_ML / AUC_ACCUMULATION_RATIO_TARGET
)  # DERIVED from SOURCE: Courtney et al. 2018, JCO steady-state Cmax and accumulation ratio
RP2D_MG = 800.0  # SOURCE: Courtney et al. 2018, JCO
PK_SATURATION_DOSE_MG = 800.0  # SOURCE: Courtney et al. 2018, JCO
EFFICACY_TROUGH_THRESHOLD_UG_PER_ML = 0.5  # SOURCE: Courtney et al. 2018, JCO

F_ORAL = 1.0  # ASSUMED — not reported in paper
F0_BIOAVAILABILITY = 1.0  # ASSUMED — not reported in paper; low-dose bioavailability limit
F_MAX_BIOAVAILABILITY = 0.10  # ASSUMED — not reported in paper; high-dose bioavailability floor
KD_BIOAVAILABILITY_UG = 500_000.0  # ASSUMED — not reported in paper; requires fitting

EPO_BASELINE_NORMALIZED = 1.0
EPO_HALF_LIFE_H = 5.0  # FIXED — endogenous EPO T1/2=5h; SOURCE: Jelkmann W, Physiol Rev 2011
IMAX_EPO = 1.0  # ASSUMED — not reported in paper
IC50_EPO_UG_PER_ML = 0.475676  # FITTED — Courtney et al. 2018 Fig 2B digitization
# Sensitivity range for IC50: 0.390-0.507 ug/mL across EPO T1/2 4-8h.
HILL_COEFFICIENT = 1.0  # FIXED — insufficient data to estimate; changed from assumed 1.5

CLINICAL_HIGH_EXPOSURE_PROBABILITY = 0.62  # SOURCE: Courtney et al. 2018, JCO
CLINICAL_LOW_EXPOSURE_PROBABILITY = 0.18  # SOURCE: Courtney et al. 2018, JCO
CLINICAL_HIGH_EXPOSURE_N = 26  # SOURCE: Courtney et al. 2018, JCO
CLINICAL_LOW_EXPOSURE_N = 22  # SOURCE: Courtney et al. 2018, JCO

CASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# =============================================================================
# Section 2: ka back-calculation
# =============================================================================

KE_H_INV = 0.693 / HALF_LIFE_H


def tmax_for_first_order_absorption(ka_h_inv: float, ke_h_inv: float) -> float:
    """Analytic Tmax for a one-compartment oral model with first-order absorption."""
    return np.log(ka_h_inv / ke_h_inv) / (ka_h_inv - ke_h_inv)


def solve_ka_for_tmax(target_tmax_h: float, ke_h_inv: float) -> float:
    """Solve Tmax = ln(ka/ke) / (ka - ke) for ka."""

    def objective(ka_h_inv: float) -> float:
        return tmax_for_first_order_absorption(ka_h_inv, ke_h_inv) - target_tmax_h

    return brentq(objective, ke_h_inv + 1e-9, 10.0)


KA_H_INV = solve_ka_for_tmax(TMAX_TARGET_H, KE_H_INV)


# Absorption ceiling parameters:
# decreasing bioavailability model for saturable oral absorption/first-pass effects.
def bioavailability_fraction(
    dose_ug: float,
    f0: float = F0_BIOAVAILABILITY,
    f_max_bio: float = F_MAX_BIOAVAILABILITY,
    kd_ug: float = KD_BIOAVAILABILITY_UG,
) -> float:
    """Return dose-dependent oral bioavailability fraction for a nominal oral dose."""
    return f0 - (f0 - f_max_bio) * dose_ug / (kd_ug + dose_ug)


def absorbed_amount_ug(
    dose_ug: float,
    f0: float = F0_BIOAVAILABILITY,
    f_max_bio: float = F_MAX_BIOAVAILABILITY,
    kd_ug: float = KD_BIOAVAILABILITY_UG,
) -> float:
    """
    Return the amount entering the absorption depot after each oral dose.

    ASSUMED model — not directly reported. Reproduces reduced exposure increase
    above 800 mg BID by applying a decreasing bioavailability fraction.
    """
    return bioavailability_fraction(dose_ug, f0, f_max_bio, kd_ug) * dose_ug


# =============================================================================
# Section 3: Vd back-calculation from estimated single-dose Cmax equation
# =============================================================================


def back_calculate_vd_ml(
    dose_ug: float,
    f_oral: float,
    ka_h_inv: float,
    ke_h_inv: float,
    tmax_h: float,
    cmax_ug_per_ml: float,
) -> float:
    """Back-calculate apparent Vd/F in mL from the stated single-dose Cmax equation."""
    concentration_shape = np.exp(-ke_h_inv * tmax_h) - np.exp(-ka_h_inv * tmax_h)
    numerator = f_oral * dose_ug * ka_h_inv * concentration_shape
    denominator = cmax_ug_per_ml * (ka_h_inv - ke_h_inv)
    return numerator / denominator


VD_OVER_F_ML = back_calculate_vd_ml(
    dose_ug=absorbed_amount_ug(RP2D_MG * MG_TO_UG),
    f_oral=F_ORAL,
    ka_h_inv=KA_H_INV,
    ke_h_inv=KE_H_INV,
    tmax_h=TMAX_TARGET_H,
    cmax_ug_per_ml=CMAX_800_SINGLE_DOSE_ESTIMATED_UG_PER_ML,
)
VD_ML = VD_OVER_F_ML
CL_OVER_F_ML_PER_H = KE_H_INV * VD_OVER_F_ML
KOUT_EPO_H_INV = 0.693 / EPO_HALF_LIFE_H
KIN_EPO_NORMALIZED_PER_H = KOUT_EPO_H_INV * EPO_BASELINE_NORMALIZED


# =============================================================================
# Section 4: ODE system function (PK + PD coupled)
# =============================================================================


def pkpd_ode(_time_h: float, y: np.ndarray) -> list[float]:
    """Coupled first-order absorption-depot PK and indirect-response EPO PD system."""
    a_depot_ug = max(y[0], 0.0)
    c_plasma_ug_per_ml = max(y[1], 0.0)
    epo_normalized = y[2]

    absorption_rate_ug_per_h = KA_H_INV * a_depot_ug
    if c_plasma_ug_per_ml <= 0.0:
        inhibition = 0.0
    else:
        c_power = c_plasma_ug_per_ml**HILL_COEFFICIENT
        ic50_power = IC50_EPO_UG_PER_ML**HILL_COEFFICIENT
        inhibition = IMAX_EPO * c_power / (ic50_power + c_power)

    d_a_depot_dt = -absorption_rate_ug_per_h
    d_c_plasma_dt = (absorption_rate_ug_per_h / VD_ML) - (KE_H_INV * c_plasma_ug_per_ml)
    d_epo_dt = KIN_EPO_NORMALIZED_PER_H * (1.0 - inhibition) - KOUT_EPO_H_INV * epo_normalized
    return [d_a_depot_dt, d_c_plasma_dt, d_epo_dt]


# =============================================================================
# Section 5: Dosing event loop
# =============================================================================


def simulate_bid_dosing(dose_mg: float) -> pd.DataFrame:
    """Simulate BID dosing by adding absorbed drug to the absorption depot."""
    dose_ug = dose_mg * MG_TO_UG
    absorbed_ug = absorbed_amount_ug(dose_ug)
    y_current = np.array([0.0, 0.0, EPO_BASELINE_NORMALIZED], dtype=float)

    time_parts = []
    state_parts = []

    for dose_index in range(N_DOSES):
        t_start = dose_index * TAU_H
        t_end = t_start + TAU_H
        y_current[0] += absorbed_ug

        t_eval = np.linspace(t_start, t_end, POINTS_PER_DOSING_INTERVAL)
        solution = solve_ivp(
            pkpd_ode,
            (t_start, t_end),
            y_current,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            method="LSODA",
        )
        if not solution.success:
            raise RuntimeError(f"ODE solve failed for {dose_mg:g} mg BID: {solution.message}")

        if dose_index == 0:
            time_parts.append(solution.t)
            state_parts.append(solution.y.T)
        else:
            time_parts.append(solution.t[1:])
            state_parts.append(solution.y.T[1:])

        y_current = solution.y[:, -1].copy()

    states = np.vstack(state_parts)
    times = np.concatenate(time_parts)
    return pd.DataFrame(
        {
            "time_h": times,
            "dose_mg": dose_mg,
            "A_depot_ug": states[:, 0],
            "C_plasma_ug_per_ml": states[:, 1],
            "EPO_normalized": states[:, 2],
        }
    )


def interval_data(profile: pd.DataFrame, start_h: float, end_h: float) -> pd.DataFrame:
    """Return one closed dosing interval."""
    eps = 1e-9
    mask = (profile["time_h"] >= start_h - eps) & (profile["time_h"] <= end_h + eps)
    return profile.loc[mask].copy()


def interval_metrics(profile: pd.DataFrame, start_h: float, end_h: float) -> dict[str, float]:
    """Calculate Cmax, trough Cmin, AUC(0-12h), Tmax, and trough EPO for one interval."""
    window = interval_data(profile, start_h, end_h)
    time_since_dose = window["time_h"].to_numpy() - start_h
    concentration = window["C_plasma_ug_per_ml"].to_numpy()
    epo = window["EPO_normalized"].to_numpy()

    cmax_index = int(np.argmax(concentration))
    return {
        "Cmax_ug_per_ml": float(concentration[cmax_index]),
        "Tmax_h": float(time_since_dose[cmax_index]),
        "Cmin_trough_ug_per_ml": float(concentration[-1]),
        "AUC_0_12h_ug_h_per_ml": float(trapezoid(concentration, time_since_dose)),
        "EPO_trough_normalized": float(epo[-1]),
    }


# =============================================================================
# Section 6: Simulation loop over all dose levels
# =============================================================================


def run_all_simulations() -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    """Run the forward simulation for all requested BID dose levels."""
    profiles = {}
    summary_rows = []
    day15_start_h = SIMULATION_HOURS - TAU_H
    day15_end_h = SIMULATION_HOURS

    for dose_mg in DOSE_LEVELS_MG:
        profile = simulate_bid_dosing(dose_mg)
        profiles[dose_mg] = profile

        day1 = interval_metrics(profile, 0.0, TAU_H)
        day15 = interval_metrics(profile, day15_start_h, day15_end_h)

        summary_rows.append(
            {
                "dose_mg_BID": dose_mg,
                "Day1_Cmax_ug_per_ml": day1["Cmax_ug_per_ml"],
                "Day1_Tmax_h": day1["Tmax_h"],
                "Day1_AUC_0_12h_ug_h_per_ml": day1["AUC_0_12h_ug_h_per_ml"],
                "Day15_Cmax_ug_per_ml": day15["Cmax_ug_per_ml"],
                "Day15_Cmin_trough_ug_per_ml": day15["Cmin_trough_ug_per_ml"],
                "Day15_AUC_0_12h_ug_h_per_ml": day15["AUC_0_12h_ug_h_per_ml"],
                "Day15_Tmax_h": day15["Tmax_h"],
                "Day15_EPO_trough_normalized": day15["EPO_trough_normalized"],
                "AUC_accumulation_ratio_Day15_Day1": (
                    day15["AUC_0_12h_ug_h_per_ml"] / day1["AUC_0_12h_ug_h_per_ml"]
                ),
            }
        )

    return profiles, pd.DataFrame(summary_rows)


# =============================================================================
# Section 7: Plotting functions
# =============================================================================


def dose_colors() -> dict[float, tuple[float, float, float, float]]:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DOSE_LEVELS_MG)))
    return {float(dose): colors[i] for i, dose in enumerate(DOSE_LEVELS_MG)}


def plot_pk_day1_day15(profiles: dict[float, pd.DataFrame]) -> Path:
    """Plot plasma concentration-time profiles on Day 1 and the final steady-state interval."""
    colors = dose_colors()
    day15_start_h = SIMULATION_HOURS - TAU_H
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for dose_mg, profile in profiles.items():
        day1 = interval_data(profile, 0.0, TAU_H)
        day15 = interval_data(profile, day15_start_h, SIMULATION_HOURS)

        axes[0].plot(
            day1["time_h"],
            day1["C_plasma_ug_per_ml"],
            color=colors[float(dose_mg)],
            label=f"{dose_mg:g} mg BID",
        )
        axes[1].plot(
            day15["time_h"] - day15_start_h,
            day15["C_plasma_ug_per_ml"],
            color=colors[float(dose_mg)],
            label=f"{dose_mg:g} mg BID",
        )

    for ax, title in zip(axes, ["Day 1", "Day 15 steady-state interval"]):
        ax.axhline(
            EFFICACY_TROUGH_THRESHOLD_UG_PER_ML,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="0.5 ug/mL threshold" if title == "Day 1" else None,
        )
        ax.set_title(title)
        ax.set_xlabel("Time since dose (h)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("PT2385 plasma concentration (ug/mL)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("PT2385 plasma concentration after BID oral dosing")
    fig.tight_layout()

    output_path = FIGURES_DIR / "pt2385_pk_day1_day15.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_trough_threshold(summary: pd.DataFrame) -> Path:
    """Plot steady-state trough concentration by dose."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        summary["dose_mg_BID"],
        summary["Day15_Cmin_trough_ug_per_ml"],
        marker="o",
        color="#2f6f9f",
        linewidth=2,
    )
    ax.axhline(
        EFFICACY_TROUGH_THRESHOLD_UG_PER_ML,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="0.5 ug/mL threshold",
    )
    ax.set_xlabel("Dose (mg BID)")
    ax.set_ylabel("Day 15 trough concentration (ug/mL)")
    ax.set_title("PT2385 steady-state trough concentration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = FIGURES_DIR / "pt2385_day15_trough_threshold.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_epo_time(profiles: dict[float, pd.DataFrame]) -> Path:
    """Plot normalized EPO over the full simulation duration."""
    colors = dose_colors()
    fig, ax = plt.subplots(figsize=(10, 5))

    for dose_mg, profile in profiles.items():
        ax.plot(
            profile["time_h"] / HOURS_PER_DAY,
            profile["EPO_normalized"],
            color=colors[float(dose_mg)],
            label=f"{dose_mg:g} mg BID",
        )

    ax.axhline(0.2, color="black", linestyle=":", linewidth=1.2, label="EPO = 0.2")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized EPO")
    ax.set_title("Indirect-response EPO suppression by PT2385")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    output_path = FIGURES_DIR / "pt2385_epo_time.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_epo_dose_response(summary: pd.DataFrame) -> Path:
    """Plot steady-state EPO suppression at trough versus dose."""
    fig, ax = plt.subplots(figsize=(7, 5))
    epo_suppression = 1.0 - summary["Day15_EPO_trough_normalized"]
    ax.plot(summary["dose_mg_BID"], epo_suppression, marker="o", color="#59753d", linewidth=2)
    ax.set_xlabel("Dose (mg BID)")
    ax.set_ylabel("Day 15 trough EPO suppression (1 - normalized EPO)")
    ax.set_title("PT2385 EPO exposure-response at steady state")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output_path = FIGURES_DIR / "pt2385_epo_dose_response.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_exposure_efficacy(summary: pd.DataFrame) -> Path:
    """Illustrative clinical exposure-efficacy visualization without fitting a model."""
    cmin_values = summary["Day15_Cmin_trough_ug_per_ml"].to_numpy()
    x_max = max(float(np.max(cmin_values)) * 1.08, EFFICACY_TROUGH_THRESHOLD_UG_PER_ML * 2.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [0.0, EFFICACY_TROUGH_THRESHOLD_UG_PER_ML],
        [CLINICAL_LOW_EXPOSURE_PROBABILITY * 100.0] * 2,
        color="#a66a39",
        linewidth=5,
        solid_capstyle="butt",
        label=(
            f"Cmin,ss < 0.5 ug/mL: {CLINICAL_LOW_EXPOSURE_PROBABILITY:.0%} "
            f"SD >= 4 mo (n={CLINICAL_LOW_EXPOSURE_N})"
        ),
    )
    ax.plot(
        [EFFICACY_TROUGH_THRESHOLD_UG_PER_ML, x_max],
        [CLINICAL_HIGH_EXPOSURE_PROBABILITY * 100.0] * 2,
        color="#326b69",
        linewidth=5,
        solid_capstyle="butt",
        label=(
            f"Cmin,ss >= 0.5 ug/mL: {CLINICAL_HIGH_EXPOSURE_PROBABILITY:.0%} "
            f"SD >= 4 mo (n={CLINICAL_HIGH_EXPOSURE_N})"
        ),
    )
    ax.axvline(
        EFFICACY_TROUGH_THRESHOLD_UG_PER_ML,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="0.5 ug/mL threshold",
    )

    rug_y0 = 2.0
    rug_y1 = 9.0
    for _, row in summary.iterrows():
        cmin = row["Day15_Cmin_trough_ug_per_ml"]
        dose = row["dose_mg_BID"]
        ax.plot([cmin, cmin], [rug_y0, rug_y1], color="#404040", linewidth=1.4)
        ax.text(cmin, rug_y1 + 1.2, f"{dose:g}", rotation=90, ha="center", va="bottom", fontsize=8)

    ax.text(np.mean(cmin_values), rug_y0 - 0.8, "Simulated dose troughs (mg BID)", ha="center", va="top")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 75.0)
    ax.set_xlabel("Steady-state trough concentration, Cmin,ss (ug/mL)")
    ax.set_ylabel("Probability of stable disease >= 4 months (%)")
    ax.set_title("Clinical exposure-response threshold with simulated troughs")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    output_path = FIGURES_DIR / "pt2385_exposure_efficacy.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# =============================================================================
# Section 8: Verification print statements
# =============================================================================


def print_parameter_block() -> None:
    """Print derived PK parameters and assumptions."""
    print("\nDerived PK parameters")
    print("---------------------")
    print(f"ke = {KE_H_INV:.5f} h^-1")
    print(f"ka = {KA_H_INV:.5f} h^-1")
    print(f"Cmax_single_dose used for Vd = {CMAX_800_SINGLE_DOSE_ESTIMATED_UG_PER_ML:.2f} ug/mL")
    print(f"Vd/F = {VD_OVER_F_ML:,.1f} mL ({VD_OVER_F_ML / 1000.0:,.2f} L)")
    print(f"CL/F = {CL_OVER_F_ML_PER_H:,.1f} mL/h ({CL_OVER_F_ML_PER_H / 1000.0:,.2f} L/h)")
    print(f"F0 bioavailability = {F0_BIOAVAILABILITY:.2f}")
    print(f"Fmax bioavailability floor = {F_MAX_BIOAVAILABILITY:.2f}")
    print(f"Kd bioavailability = {KD_BIOAVAILABILITY_UG:,.0f} ug")


def print_absorbed_amount_block() -> None:
    """Print absorbed amount and fraction for each simulated nominal dose."""
    print("\nAbsorbed amount per dose by dose level")
    print("--------------------------------------")
    for dose_mg in DOSE_LEVELS_MG.astype(int):
        dose_ug = dose_mg * MG_TO_UG
        absorbed = absorbed_amount_ug(dose_ug)
        fraction = absorbed / dose_ug
        print(f"  {dose_mg:5d} mg: absorbed = {absorbed / MG_TO_UG:.1f} mg  ({fraction * 100.0:.1f}% of dose)")


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print the steady-state PK table requested in the prompt."""
    table = summary[
        [
            "dose_mg_BID",
            "Day15_Cmax_ug_per_ml",
            "Day15_Cmin_trough_ug_per_ml",
            "Day15_AUC_0_12h_ug_h_per_ml",
            "Day15_EPO_trough_normalized",
        ]
    ].copy()
    print("\nDay 15 steady-state summary")
    print("---------------------------")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def print_pd_response_checks(profiles: dict[float, pd.DataFrame], summary: pd.DataFrame) -> None:
    """Print PD checks requested in the prompt."""
    print("\nPD response checks")
    print("------------------")
    for dose_mg in DOSE_LEVELS_MG:
        profile = profiles[float(dose_mg)]
        epo_24h = float(np.interp(24.0, profile["time_h"], profile["EPO_normalized"]))
        epo_48h = float(np.interp(48.0, profile["time_h"], profile["EPO_normalized"]))
        print(f"{dose_mg:g} mg BID: EPO at 24 h = {epo_24h:.3f}; EPO at 48 h = {epo_48h:.3f}")

    high_dose_pd = summary.loc[summary["dose_mg_BID"] >= PK_SATURATION_DOSE_MG]
    high_dose_all_below_02 = bool((high_dose_pd["Day15_EPO_trough_normalized"] < 0.2).all())
    epo_800 = float(summary.loc[summary["dose_mg_BID"] == 800.0, "Day15_EPO_trough_normalized"].iloc[0])
    epo_1800 = float(summary.loc[summary["dose_mg_BID"] == 1800.0, "Day15_EPO_trough_normalized"].iloc[0])
    print(
        f"Day 15 EPO < 0.2 for doses >= 800 mg BID: "
        f"{'PASS' if high_dose_all_below_02 else 'FAIL'}"
    )
    print(f"PD saturation check: EPO trough 800 mg = {epo_800:.3f}; 1800 mg = {epo_1800:.3f}")


def print_verification_block(summary: pd.DataFrame) -> None:
    """Print the requested verification block."""
    row_800 = summary.loc[summary["dose_mg_BID"] == RP2D_MG].iloc[0]
    row_1800 = summary.loc[summary["dose_mg_BID"] == 1800.0].iloc[0]

    cmax_800 = row_800["Day15_Cmax_ug_per_ml"]
    tmax_800_day1 = row_800["Day1_Tmax_h"]
    auc_ratio_800 = row_800["AUC_accumulation_ratio_Day15_Day1"]
    cmin_800 = row_800["Day15_Cmin_trough_ug_per_ml"]
    epo_800 = row_800["Day15_EPO_trough_normalized"]
    cmax_ratio_1800_to_800 = row_1800["Day15_Cmax_ug_per_ml"] / cmax_800
    absorbed_800 = absorbed_amount_ug(800.0 * MG_TO_UG)
    absorbed_1800 = absorbed_amount_ug(1800.0 * MG_TO_UG)
    absorbed_ratio_1800_to_800 = absorbed_1800 / absorbed_800
    absorbed_fraction_800 = absorbed_800 / (800.0 * MG_TO_UG)
    absorbed_fraction_1800 = absorbed_1800 / (1800.0 * MG_TO_UG)
    absorbed_ratio_pass = absorbed_ratio_1800_to_800 < 1.6
    saturation_pass = cmax_ratio_1800_to_800 < 1.5

    print("\nVerification checklist")
    print("----------------------")
    print(
        f"[CHECK 1] Simulated Cmax at 800 mg BID Day 15: "
        f"{cmax_800:.2f} ug/mL   (Target: 3.0-3.2 ug/mL)"
    )
    print(
        f"[CHECK 2] Simulated Tmax at 800 mg BID Day 1: "
        f"{tmax_800_day1:.1f} h         (Target: 1.5-2.5 h)"
    )
    print(
        f"[CHECK 3] AUC accumulation ratio Day15/Day1:   "
        f"{auc_ratio_800:.1f}           (Target: 2.3-2.7)"
    )
    print(
        f"[CHECK 4] Simulated Cmin,ss at 800 mg Day 15:  "
        f"{cmin_800:.2f} ug/mL   (Target: >= 0.5 ug/mL)"
    )
    print(
        f"[CHECK 5] EPO normalized at steady state 800mg: "
        f"{epo_800:.2f}         (Target: < 0.2)"
    )
    print(
        f"[CHECK 6] 1800mg/800mg absorbed amount ratio: "
        f"{absorbed_ratio_1800_to_800:.2f}  "
        f"(Target: < 1.6 = {'PASS' if absorbed_ratio_pass else 'FAIL'})"
    )
    print(
        f"[CHECK 7] 1800mg/800mg Cmax ratio Day 15:     "
        f"{cmax_ratio_1800_to_800:.2f}  "
        f"(Target: < 1.5 = saturation {'PASS' if saturation_pass else 'FAIL'})"
    )
    print(
        f"[CHECK 8] Absorbed fraction at 800 mg:        "
        f"{absorbed_fraction_800:.2f}  (Print for transparency)"
    )
    print(
        f"[CHECK 9] Absorbed fraction at 1800 mg:       "
        f"{absorbed_fraction_1800:.2f}  (Print for transparency)"
    )

    checks_pass = [
        3.0 <= cmax_800 <= 3.2,
        1.5 <= tmax_800_day1 <= 2.5,
        2.3 <= auc_ratio_800 <= 2.7,
        cmin_800 >= EFFICACY_TROUGH_THRESHOLD_UG_PER_ML,
        epo_800 < 0.2,
        absorbed_ratio_pass,
        saturation_pass,
    ]
    if not all(checks_pass):
        print(
            "\nNOTE: The corrected prompt-specified model was not further refit. "
            "Any remaining target miss is reported without introducing additional parameters."
        )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print_absorbed_amount_block()
    profiles, summary = run_all_simulations()
    summary_path = RESULTS_DIR / "pt2385_day15_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_paths = [
        plot_pk_day1_day15(profiles),
        plot_trough_threshold(summary),
        plot_epo_time(profiles),
        plot_epo_dose_response(summary),
        plot_exposure_efficacy(summary),
    ]

    print_parameter_block()
    print_summary_table(summary)
    print_verification_block(summary)
    print_pd_response_checks(profiles, summary)

    print("\nSaved outputs")
    print("-------------")
    print(summary_path)
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
