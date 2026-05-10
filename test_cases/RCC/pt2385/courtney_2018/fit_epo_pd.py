"""
Fit PT2385 EPO pharmacodynamic parameters to digitized Courtney et al. 2018
Figure 2B data.

The script fixes Kout and fits only IC50. PK parameters are fixed exactly as stated
in the fitting prompt.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Section 1: Input data
# =============================================================================

# Digitized Figure 2B - Courtney et al. 2018, JCO
# EPO % change from baseline (mean +/- SD)
# Time unit: days
# Digitization uncertainty: +/-5 percentage points on mean, +/-3 on SD
# 200 mg Day 8 excluded from fitting (anomalous, n=3, high variability)

epo_obs = [
    # (dose_mg, time_days, mean_pct_change, sd)
    # 100 mg
    (100, 0.042, -20, 10),
    (100, 0.5, -28, 12),
    (100, 8.0, -21, 15),
    # 200 mg - Day 8 excluded
    (200, 0.042, -15, 12),
    (200, 0.5, -13, 10),
    # 400 mg
    (400, 0.042, -25, 10),
    (400, 0.5, -28, 8),
    (400, 8.0, -38, 10),
    # 800 mg
    (800, 0.042, -47, 8),
    (800, 0.5, -51, 6),
    (800, 8.0, -82, 8),
    # 1200 mg
    (1200, 0.042, -38, 8),
    (1200, 0.5, -41, 7),
    (1200, 8.0, -50, 10),
    # 1800 mg
    (1800, 0.042, -43, 10),
    (1800, 0.5, -46, 9),
    (1800, 8.0, -64, 12),
]


# =============================================================================
# Section 2: PK input from validated base model
# =============================================================================

# All units: ug, mL, hours

ke = 0.693 / 17.0  # h^-1 - SOURCE: T1/2=17h, Courtney et al. 2018
Vd = 594_649.0  # mL - SOURCE: back-calculated in pt2385_pkpd_model.py
ka = 1.9830527405774512  # h^-1 - SOURCE: back-calculated from Tmax=2h

F0 = 1.0  # ASSUMED in pt2385_pkpd_model.py
F_MAX_BIO = 0.10  # ASSUMED in pt2385_pkpd_model.py (Patch D)
Kd_abs = 500_000.0  # ug - ASSUMED in pt2385_pkpd_model.py (Patch D)

DOSING_INTERVAL = 12.0  # hours - BID dosing
MG_TO_UG = 1000.0
HOURS_PER_DAY = 24.0
CASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DoseLevels = np.array([100, 200, 400, 800, 1200, 1800], dtype=float)


def bioavailability(dose_ug, F0, F_max_bio, Kd_abs):
    F = F0 - (F0 - F_max_bio) * dose_ug / (Kd_abs + dose_ug)
    return max(F, F_max_bio)


def absorbed_amount_ug(dose_ug):
    return bioavailability(dose_ug, F0, F_MAX_BIO, Kd_abs) * dose_ug


# =============================================================================
# Section 3: PK simulation function
# =============================================================================


def simulate_pk(
    dose_mg,
    t_obs_days,
    ke,
    ka,
    Vd,
    absorbed_amount_ug,
    dosing_interval_h=12.0,
):
    """
    Simulate plasma concentration C_plasma (ug/mL) at requested observation
    times for a given dose using BID oral dosing with first-order absorption
    and linear elimination.

    Returns: array of C_plasma values (ug/mL) at each t_obs_days timepoint.
    """
    dose_ug = dose_mg * MG_TO_UG
    t_obs_h = np.array(t_obs_days, dtype=float) * HOURS_PER_DAY
    t_end_h = max(t_obs_h) + dosing_interval_h

    dose_times = np.arange(0.0, t_end_h, dosing_interval_h)

    A_depot = 0.0
    C_plasma = 0.0
    t_all = [0.0]
    C_all = [0.0]

    for i, t_dose in enumerate(dose_times):
        A_depot += absorbed_amount_ug(dose_ug)

        t_next = dose_times[i + 1] if i + 1 < len(dose_times) else t_end_h
        t_span = (t_dose, t_next)
        t_eval = np.linspace(t_dose, t_next, 500)

        y0 = [A_depot, C_plasma]

        def odes(_t, y):
            A, C = y
            dA = -ka * A
            dC = (ka * A) / Vd - ke * C
            return [dA, dC]

        sol = solve_ivp(
            odes,
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            max_step=0.1,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"PK solve failed for {dose_mg:g} mg: {sol.message}")

        A_depot = sol.y[0][-1]
        C_plasma = sol.y[1][-1]

        t_all.extend(sol.t[1:].tolist())
        C_all.extend(sol.y[1][1:].tolist())

    interp = interp1d(
        t_all,
        C_all,
        kind="linear",
        bounds_error=False,
        fill_value=(C_all[0], C_all[-1]),
    )
    return interp(t_obs_h)


# =============================================================================
# Section 4: EPO indirect response model
# =============================================================================


def simulate_epo(
    dose_mg,
    t_obs_days,
    Kout,
    IC50,
    Imax,
    n_hill,
    ke,
    ka,
    Vd,
    absorbed_amount_ug,
    dosing_interval_h=12.0,
):
    """
    Simulate EPO % change from baseline using indirect response Type I model
    (inhibition of EPO production by PT2385).
    """
    dose_ug = dose_mg * MG_TO_UG
    t_obs_h = np.array(t_obs_days, dtype=float) * HOURS_PER_DAY
    t_end_h = max(t_obs_h) + dosing_interval_h
    dose_times = np.arange(0.0, t_end_h, dosing_interval_h)

    A_depot = 0.0
    C_plasma = 0.0
    EPO = 1.0
    Kin = Kout

    t_all = [0.0]
    EPO_all = [1.0]

    for i, t_dose in enumerate(dose_times):
        A_depot += absorbed_amount_ug(dose_ug)
        t_next = dose_times[i + 1] if i + 1 < len(dose_times) else t_end_h
        t_span = (t_dose, t_next)
        t_eval = np.linspace(t_dose, t_next, 500)

        y0 = [A_depot, C_plasma, EPO]

        def odes(_t, y):
            A, C, E = y
            C_nonnegative = max(C, 0.0)
            dA = -ka * A
            dC = (ka * A) / Vd - ke * C
            inh = Imax * (C_nonnegative**n_hill) / (IC50**n_hill + C_nonnegative**n_hill)
            dE = Kin * (1.0 - inh) - Kout * E
            return [dA, dC, dE]

        sol = solve_ivp(
            odes,
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            max_step=0.1,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"PKPD solve failed for {dose_mg:g} mg: {sol.message}")

        A_depot = sol.y[0][-1]
        C_plasma = sol.y[1][-1]
        EPO = sol.y[2][-1]

        t_all.extend(sol.t[1:].tolist())
        EPO_all.extend(sol.y[2][1:].tolist())

    EPO_arr = np.array(EPO_all)
    pct_change = (EPO_arr - 1.0) * 100.0

    interp = interp1d(
        t_all,
        pct_change,
        kind="linear",
        bounds_error=False,
        fill_value=(pct_change[0], pct_change[-1]),
    )
    return interp(t_obs_h)


# =============================================================================
# Section 5: Fixed parameters
# =============================================================================

Imax = 1.0
# FIXED: near-complete EPO suppression at high doses
# SOURCE: qualitative, Courtney et al. 2018 Fig 2B

n_hill = 1.0
# FIXED: insufficient timepoints to estimate Hill coefficient
# ASSUMPTION: classical Michaelis-Menten kinetics

Kout = 0.693 / 5.0  # = 0.1386 h^-1
# FIXED: endogenous EPO half-life = 5 hours in humans
# SOURCE: Jelkmann W, Physiol Rev 2011; standard indirect response modeling practice
# NOT estimated from this dataset - data too sparse (3 timepoints/group)
# to reliably identify Kout. Sensitivity analysis over T1/2 = 4, 5, 6, 8 h
# is performed in Section 8 below.
Kin = Kout  # normalized baseline EPO = 1.0, so Kin = Kout * 1.0

T_half_EPO = 0.693 / Kout
print(f"Fixed Kout = {Kout:.6f} h^-1  (EPO T1/2 = {T_half_EPO:.2f} h)")
print("Reason: EPO half-life fixed at physiological literature value.")
print("        Only IC50 will be estimated from digitized Fig 2B data.\n")


# =============================================================================
# Section 6: Weighted least squares objective function
# =============================================================================


def objective_IC50(log10_IC50, epo_obs, Kout, ke, ka, Vd, absorbed_amount_ug, Imax, n_hill):
    """
    Weighted least squares objective - fit IC50 only.
    Kout is fixed at physiological value.
    log10_IC50: scalar, fit in log space to prevent negative values.
    """
    IC50 = 10**log10_IC50

    total_wss = 0.0
    for dose_mg, t_day, mean_pct, sd in epo_obs:
        if sd <= 0:
            continue
        epo_pred = simulate_epo(
            dose_mg,
            [t_day],
            Kout,
            IC50,
            Imax,
            n_hill,
            ke,
            ka,
            Vd,
            absorbed_amount_ug,
        )[0]
        weight = 1.0 / (sd**2)
        total_wss += weight * (epo_pred - mean_pct) ** 2

    return total_wss


# =============================================================================
# Section 7: Optimization
# =============================================================================


def fit_parameters():
    result = minimize_scalar(
        objective_IC50,
        bounds=(-2.0, 0.70),
        method="bounded",
        args=(epo_obs, Kout, ke, ka, Vd, absorbed_amount_ug, Imax, n_hill),
        options={"xatol": 1e-8, "maxiter": 500},
    )

    IC50_fitted = 10**result.x
    Kout_fitted = Kout
    T_half_EPO_fitted = 0.693 / Kout_fitted

    print("\n══════════════════════════════════════════")
    print("  FITTED PD PARAMETERS")
    print("══════════════════════════════════════════")
    print(f"  Kout  = {Kout_fitted:.6f} h⁻¹  (FIXED — literature value)")
    print(f"  T½EPO = {T_half_EPO_fitted:.2f} h         (FIXED — literature value)")
    print(f"  IC50  = {IC50_fitted:.4f} µg/mL  (FITTED from Fig 2B)")
    print(f"  Imax  = {Imax}              (FIXED)")
    print(f"  n     = {n_hill}              (FIXED)")
    print(f"  Final weighted SSR = {result.fun:.4f}")
    print(f"  Optimizer converged: {result.success if hasattr(result, 'success') else 'N/A'}")
    print("══════════════════════════════════════════")

    return result, Kout_fitted, IC50_fitted, T_half_EPO_fitted


# =============================================================================
# Section 8: Plausibility checks
# =============================================================================


def print_plausibility_checks(Kout_fitted, IC50_fitted, _T_half_EPO_fitted):
    print("\n── Plausibility checks ──────────────────────────────────────────────")

    if 0.05 <= IC50_fitted <= 4.0:
        print(f"[PASS] IC50 = {IC50_fitted:.4f} µg/mL  (within PK exposure range)")
    else:
        print(f"[WARN] IC50 = {IC50_fitted:.4f} µg/mL  OUTSIDE expected range 0.05–4.0")

    epo_800_day8 = simulate_epo(
        800,
        [8.0],
        Kout_fitted,
        IC50_fitted,
        Imax,
        n_hill,
        ke,
        ka,
        Vd,
        absorbed_amount_ug,
    )[0]
    print(
        f"[CHECK] 800mg Day 8:  predicted {epo_800_day8:.1f}%  observed −82%  "
        f"{'PASS' if abs(epo_800_day8 + 82) <= 20 else 'WARN >20pp deviation'}"
    )

    epo_100_day8 = simulate_epo(
        100,
        [8.0],
        Kout_fitted,
        IC50_fitted,
        Imax,
        n_hill,
        ke,
        ka,
        Vd,
        absorbed_amount_ug,
    )[0]
    print(
        f"[CHECK] 100mg Day 8:  predicted {epo_100_day8:.1f}%  observed −21%  "
        f"{'PASS' if abs(epo_100_day8 + 21) <= 15 else 'WARN >15pp deviation'}"
    )

    epo_400_day8 = simulate_epo(
        400,
        [8.0],
        Kout_fitted,
        IC50_fitted,
        Imax,
        n_hill,
        ke,
        ka,
        Vd,
        absorbed_amount_ug,
    )[0]
    print(
        f"[CHECK] 400mg Day 8:  predicted {epo_400_day8:.1f}%  observed −38%  "
        f"{'PASS' if abs(epo_400_day8 + 38) <= 20 else 'WARN >20pp deviation'}"
    )

    print("\n── Sensitivity analysis: IC50 vs fixed EPO T½ ──────────────────────")
    print(f"{'EPO T½ (h)':>12}  {'Kout (h⁻¹)':>12}  {'IC50 (µg/mL)':>14}  {'SSR':>10}")
    print("-" * 55)
    for t_half in [4.0, 5.0, 6.0, 8.0]:
        Kout_test = 0.693 / t_half
        res = minimize_scalar(
            objective_IC50,
            bounds=(-2.0, 0.70),
            method="bounded",
            args=(epo_obs, Kout_test, ke, ka, Vd, absorbed_amount_ug, Imax, n_hill),
            options={"xatol": 1e-8},
        )
        IC50_test = 10**res.x
        marker = " ◄ selected" if t_half == 5.0 else ""
        print(
            f"{t_half:>12.1f}  {Kout_test:>12.6f}  {IC50_test:>14.4f}  "
            f"{res.fun:>10.4f}{marker}"
        )
    print("\nIf IC50 is stable across T½ range → IC50 is robustly identified.")
    print("If IC50 varies strongly → results are sensitive to Kout assumption.\n")


# =============================================================================
# Section 9: Diagnostic plots
# =============================================================================


def dose_color_map():
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DoseLevels)))
    return {int(dose): colors[i] for i, dose in enumerate(DoseLevels)}


def plot_observed_vs_predicted(Kout_fitted, IC50_fitted):
    output_path = FIGURES_DIR / "epo_fit_observed_vs_predicted.png"
    colors = dose_color_map()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    dense_days = np.linspace(0.0, 9.0, 500)

    for ax, dose_mg in zip(axes.flat, DoseLevels.astype(int)):
        dose_obs = [row for row in epo_obs if row[0] == dose_mg]
        obs_t = np.array([row[1] for row in dose_obs])
        obs_mean = np.array([row[2] for row in dose_obs])
        obs_sd = np.array([row[3] for row in dose_obs])
        color = colors[dose_mg]

        pred = simulate_epo(
            dose_mg,
            dense_days,
            Kout_fitted,
            IC50_fitted,
            Imax,
            n_hill,
            ke,
            ka,
            Vd,
            absorbed_amount_ug,
        )

        ax.errorbar(
            obs_t,
            obs_mean,
            yerr=obs_sd,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
            label="Observed mean +/- SD",
        )
        ax.plot(dense_days, pred, color=color, linewidth=2, label="Model prediction")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)

        title = f"{dose_mg} mg BID"
        if dose_mg == 200:
            title += " (Day 8 excluded)"
        ax.set_title(title)
        ax.set_xlim(0.0, 9.0)
        ax.set_ylim(-110.0, 30.0)
        ax.grid(alpha=0.25)
        ax.text(
            0.04,
            0.06,
            f"IC50={IC50_fitted:.3f} µg/mL\nKout={Kout_fitted:.4f} h⁻¹",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (days)")
    for ax in axes[:, 0]:
        ax.set_ylabel("EPO % change from baseline")

    fig.suptitle("PT2385 EPO PD Fit to Digitized Courtney et al. 2018 Figure 2B")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def compute_day15_trough_cmin(dose_mg):
    return float(simulate_pk(dose_mg, [14.0], ke, ka, Vd, absorbed_amount_ug, DOSING_INTERVAL)[0])


def plot_exposure_response_steady_state(Kout_fitted, IC50_fitted):
    output_path = FIGURES_DIR / "epo_exposure_response_steady_state.png"
    colors = dose_color_map()

    cmin_values = np.array([compute_day15_trough_cmin(dose) for dose in DoseLevels])
    epo_day8_values = np.array(
        [
            simulate_epo(dose, [8.0], Kout_fitted, IC50_fitted, Imax, n_hill, ke, ka, Vd, absorbed_amount_ug)[0]
            for dose in DoseLevels
        ]
    )

    c_curve = np.linspace(0.0, 5.0, 500)
    er_curve = -Imax * 100.0 * (c_curve**n_hill) / (IC50_fitted**n_hill + c_curve**n_hill)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c_curve, er_curve, color="#2f5d62", linewidth=2.2, label="Fitted Emax relationship")
    for dose, cmin, epo_pred in zip(DoseLevels.astype(int), cmin_values, epo_day8_values):
        ax.scatter(cmin, epo_pred, s=70, color=colors[dose], edgecolor="black", linewidth=0.4, label=f"{dose} mg")

    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="0.5 µg/mL threshold")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-110.0, 10.0)
    ax.set_xlabel("Simulated steady-state Cmin (µg/mL)")
    ax.set_ylabel("Predicted EPO % change from baseline at Day 8")
    ax.set_title("EPO Exposure-Response at Steady State (Fitted Model)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# =============================================================================
# Section 10: Output file
# =============================================================================


def write_parameter_report(Kout_fitted, IC50_fitted, T_half_EPO_fitted, weighted_ssr):
    output_path = RESULTS_DIR / "fitted_pd_parameters.txt"
    n_obs_used = sum(1 for row in epo_obs if row[3] > 0)
    text = f"""══════════════════════════════════════════════
  PT2385 FITTED PD PARAMETERS
  Source data: Courtney et al. 2018, JCO, Fig 2B (digitized)
  Fitting method: Weighted least squares, bounded scalar optimization for IC50
  Fixed parameters: Kout=0.693/5h, Imax=1.0, n_hill=1.0
══════════════════════════════════════════════
  Kout  = {Kout_fitted:.6f} h⁻¹     (FIXED — literature EPO T½=5h)
  T½EPO = {T_half_EPO_fitted:.2f} h
  IC50  = {IC50_fitted:.6f} µg/mL   (95% CI: not estimated — sparse data)
  Weighted SSR = {weighted_ssr:.4f}
  N observations used = {n_obs_used}
  N observations excluded = 1  (200 mg, Day 8 — anomalous)
══════════════════════════════════════════════
  IMPORTANT LIMITATIONS:
  1. Only 3 timepoints per dose group — parameters are weakly identified
  2. IC50 and Kout are correlated — confidence intervals require bootstrap
  3. Digitization error ±5 pct points on mean values
  4. Hill coefficient fixed at 1.0 — not estimated from data
  5. 200 mg Day 8 excluded — results sensitive to this exclusion
══════════════════════════════════════════════
"""
    output_path.write_text(text, encoding="utf-8-sig")
    return output_path


# =============================================================================
# Section 11: Instructions for updating base model
# =============================================================================


def print_update_instructions(Kout_fitted, IC50_fitted):
    print("\n── Next step: update pt2385_pkpd_model.py ──────────────────────────")
    print(f"  Replace:  Kout = 0.1066   with:  Kout = {Kout_fitted:.6f}")
    print("            # FIXED at literature EPO T½=5h (Jelkmann 2011)")
    print(f"  Replace:  IC50 = 0.5      with:  IC50 = {IC50_fitted:.6f}")
    print("            # FITTED from digitized Fig 2B, Courtney et al. 2018")
    print("  Replace:  n_hill = 1.5    with:  n_hill = 1.0")
    print("            # FIXED — corrected from original assumed value")
    print("  Keep:     Imax = 1.0      (unchanged)")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    result, Kout_fitted, IC50_fitted, T_half_EPO_fitted = fit_parameters()

    print_plausibility_checks(Kout_fitted, IC50_fitted, T_half_EPO_fitted)
    plot_paths = [
        plot_observed_vs_predicted(Kout_fitted, IC50_fitted),
        plot_exposure_response_steady_state(Kout_fitted, IC50_fitted),
    ]
    report_path = write_parameter_report(Kout_fitted, IC50_fitted, T_half_EPO_fitted, result.fun)

    print("\nSaved outputs")
    print("-------------")
    for path in plot_paths:
        print(path)
    print(report_path)

    print_update_instructions(Kout_fitted, IC50_fitted)

    if hasattr(result, "success") and not result.success:
        print(f"\n[WARN] bounded scalar optimizer did not report success: {result.message}")


if __name__ == "__main__":
    main()
