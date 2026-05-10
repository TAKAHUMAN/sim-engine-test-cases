"""Acquired-resistance refinement for the adult RCC PD model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from simulation.pd_simulate import DEFAULT_PD_SEED, LAMBDA_KILL_LITERATURE_MEDIAN, simulate_pd_cohort
from simulation.pd_validate import (
    RECORD1_MEDIAN_PFS_MONTHS,
    RECORD1_PFS_6M,
    format_pd_validation_report,
    survival_probability_at,
    validate_against_record1,
)

DEFAULT_TAIL_CALIBRATED_TAU_DAYS = 69.0
RECORD1_TARGETS = {"median": RECORD1_MEDIAN_PFS_MONTHS, "pfs_6m": RECORD1_PFS_6M, "pfs_12m": 0.075}


@dataclass(frozen=True)
class ResistanceRefinementResult:
    """Outputs from a resistance-refined PD run."""

    tau_resist_days: float
    lambda_kill_0_median: float
    patients: pd.DataFrame
    checks: pd.DataFrame
    report: str


@dataclass(frozen=True)
class JointCalibrationResult:
    """Outputs from joint calibration of lambda_kill_0 and tau_resist."""

    lambda_kill_0_median: float
    tau_resist_days: float
    objective_value: float
    optimizer_success: bool
    optimizer_message: str
    calibration_trace: pd.DataFrame
    patients: pd.DataFrame
    checks: pd.DataFrame
    report: str


def run_resistance_refined_pipeline(
    *,
    n_individuals: int = 500,
    seed: int = DEFAULT_PD_SEED,
    tau_resist_days: float = DEFAULT_TAIL_CALIBRATED_TAU_DAYS,
    lambda_kill_0_median: float = LAMBDA_KILL_LITERATURE_MEDIAN,
    output_dir: str | Path = "outputs",
) -> ResistanceRefinementResult:
    """Run the PD model with lambda_kill(t) = lambda_kill_0 * exp(-t / tau)."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = simulate_pd_cohort(
        N_individuals=n_individuals,
        seed=seed,
        resistance_tau_days=tau_resist_days,
        lambda_kill_median=lambda_kill_0_median,
    )
    checks = validate_against_record1(
        results.patients,
        output_dir=output,
        plot_filename="pd_resistance_pfs_validation.png",
    )
    report = format_resistance_refinement_report(
        results.patients,
        checks,
        tau_resist_days=tau_resist_days,
        lambda_kill_0_median=lambda_kill_0_median,
    )
    results.patients.to_csv(output / "pd_resistance_pfs_results.csv", index=False)
    results.pk_profiles.to_csv(output / "pd_resistance_pk_profiles.csv", index=False)
    checks.to_csv(output / "pd_resistance_validation_checks.csv", index=False)
    (output / "pd_resistance_validation_report.txt").write_text(report, encoding="utf-8")
    return ResistanceRefinementResult(
        tau_resist_days=tau_resist_days,
        lambda_kill_0_median=lambda_kill_0_median,
        patients=results.patients,
        checks=checks,
        report=report,
    )


def calibrate_tau_grid(
    *,
    tau_grid_days: tuple[float, ...] = (55.0, 60.0, 65.0, 67.0, 68.0, 69.0, 70.0, 75.0, 80.0),
    n_individuals: int = 150,
    seed: int = DEFAULT_PD_SEED,
    lambda_kill_0_median: float = LAMBDA_KILL_LITERATURE_MEDIAN,
) -> pd.DataFrame:
    """Explore tau_resist values against RECORD-1 aggregate anchors."""

    rows: list[dict[str, float]] = []
    for tau in tau_grid_days:
        results = simulate_pd_cohort(
            N_individuals=n_individuals,
            seed=seed,
            resistance_tau_days=tau,
            lambda_kill_median=lambda_kill_0_median,
        )
        patients = results.patients
        median_pfs = float(patients["PFS_months"].median())
        pfs_6m = survival_probability_at(patients, 6.0 * 30.44)
        pfs_12m = survival_probability_at(patients, 365.0)
        tail_score = abs(pfs_6m - 0.26) / 0.26 + abs(pfs_12m - 0.08) / 0.08
        median_score = abs(median_pfs - 4.9) / 4.9
        rows.append(
            {
                "tau_resist_days": tau,
                "lambda_kill_0_median": lambda_kill_0_median,
                "median_pfs_months": median_pfs,
                "pfs_6m": pfs_6m,
                "pfs_12m": pfs_12m,
                "tail_score": tail_score,
                "median_score": median_score,
                "combined_score": tail_score + median_score,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("combined_score")


def resistance_endpoints(pd_df: pd.DataFrame) -> dict[str, float]:
    """Compute RECORD-1 calibration endpoints from a PD result table."""

    return {
        "median": float(pd_df["PFS_months"].median()),
        "pfs_6m": survival_probability_at(pd_df, 6.0 * 30.44),
        "pfs_12m": survival_probability_at(pd_df, 365.0),
    }


def joint_calibration_loss(
    endpoints: dict[str, float],
    targets: dict[str, float] = RECORD1_TARGETS,
) -> float:
    """Weighted squared loss for median PFS, 6-month PFS, and 12-month PFS."""

    return float(
        ((endpoints["median"] - targets["median"]) / targets["median"]) ** 2
        + ((endpoints["pfs_6m"] - targets["pfs_6m"]) / targets["pfs_6m"]) ** 2
        + ((endpoints["pfs_12m"] - targets["pfs_12m"]) / targets["pfs_12m"]) ** 2
    )


def joint_calibration_objective(
    params: np.ndarray,
    *,
    record1_targets: dict[str, float] = RECORD1_TARGETS,
    n_individuals: int = 200,
    seed: int = DEFAULT_PD_SEED,
    trace: list[dict[str, float]] | None = None,
    cache: dict[tuple[float, float], float] | None = None,
) -> float:
    """Objective for joint calibration of lambda_kill_0 and tau_resist.

    The same seed is used for every evaluation. This common-random-number
    design makes the objective deterministic for a fixed virtual cohort size.
    """

    lambda_kill_0, tau_resist = float(params[0]), float(params[1])
    if lambda_kill_0 <= 0.0 or tau_resist <= 0.0:
        return float("inf")
    cache_key = (round(lambda_kill_0, 6), round(tau_resist, 3))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    results = simulate_pd_cohort(
        N_individuals=n_individuals,
        seed=seed,
        resistance_tau_days=tau_resist,
        lambda_kill_median=lambda_kill_0,
    )
    endpoints = resistance_endpoints(results.patients)
    loss = joint_calibration_loss(endpoints, record1_targets)
    if trace is not None:
        trace.append(
            {
                "lambda_kill_0_median": lambda_kill_0,
                "tau_resist_days": tau_resist,
                "median_pfs_months": endpoints["median"],
                "pfs_6m": endpoints["pfs_6m"],
                "pfs_12m": endpoints["pfs_12m"],
                "loss": loss,
            }
        )
    if cache is not None:
        cache[cache_key] = loss
    return loss


def run_joint_calibration(
    *,
    record1_targets: dict[str, float] = RECORD1_TARGETS,
    seed: int = DEFAULT_PD_SEED,
    n_calibration: int = 200,
    n_validation: int = 500,
    x0: tuple[float, float] = (LAMBDA_KILL_LITERATURE_MEDIAN, DEFAULT_TAIL_CALIBRATED_TAU_DAYS),
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.010, 0.025), (30.0, 120.0)),
    maxiter: int = 24,
    coarse_lambda_grid: tuple[float, ...] = (0.015, 0.018, 0.021, 0.024, 0.0245, 0.025),
    coarse_tau_grid: tuple[float, ...] = (35.0, 38.0, 39.5, 40.5, 42.0, 50.0, 60.0, 69.0, 80.0),
    output_dir: str | Path = "outputs",
) -> JointCalibrationResult:
    """Jointly optimize lambda_kill_0 and tau_resist, then validate the fit.

    Uses Powell's bounded optimizer because the simulated median and survival
    probabilities are step-like functions of the parameters.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace: list[dict[str, float]] = []
    cache: dict[tuple[float, float], float] = {}
    best_x0 = np.array(x0, dtype=float)
    best_loss = joint_calibration_objective(
        best_x0,
        record1_targets=record1_targets,
        n_individuals=n_calibration,
        seed=seed,
        trace=trace,
        cache=cache,
    )
    for lambda_candidate in coarse_lambda_grid:
        for tau_candidate in coarse_tau_grid:
            candidate = np.array([lambda_candidate, tau_candidate], dtype=float)
            loss = joint_calibration_objective(
                candidate,
                record1_targets=record1_targets,
                n_individuals=n_calibration,
                seed=seed,
                trace=trace,
                cache=cache,
            )
            if loss < best_loss:
                best_loss = loss
                best_x0 = candidate

    if maxiter > 0:
        result = minimize(
            lambda x: joint_calibration_objective(
                x,
                record1_targets=record1_targets,
                n_individuals=n_calibration,
                seed=seed,
                trace=trace,
                cache=cache,
            ),
            best_x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-3, "disp": False},
        )
        optimizer_success = bool(result.success)
        optimizer_message = str(result.message)
        if float(result.fun) <= best_loss:
            opt_lambda = float(result.x[0])
            opt_tau = float(result.x[1])
            objective_value = float(result.fun)
        else:
            opt_lambda = float(best_x0[0])
            opt_tau = float(best_x0[1])
            objective_value = best_loss
    else:
        opt_lambda = float(best_x0[0])
        opt_tau = float(best_x0[1])
        objective_value = best_loss
        optimizer_success = True
        optimizer_message = "Skipped local optimizer; selected best coarse-grid candidate."

    validation = simulate_pd_cohort(
        N_individuals=n_validation,
        seed=seed,
        resistance_tau_days=opt_tau,
        lambda_kill_median=opt_lambda,
    )
    checks = validate_against_record1(
        validation.patients,
        output_dir=output,
        plot_filename="pd_calibrated_pfs_validation.png",
    )
    trace_df = pd.DataFrame.from_records(trace).drop_duplicates(
        subset=["lambda_kill_0_median", "tau_resist_days"],
        keep="last",
    )
    trace_df = trace_df.sort_values("loss")
    report = format_joint_calibration_report(
        validation.patients,
        checks,
        lambda_kill_0_median=opt_lambda,
        tau_resist_days=opt_tau,
        objective_value=objective_value,
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
        n_calibration=n_calibration,
        n_validation=n_validation,
    )
    validation.patients.to_csv(output / "pd_calibrated_results.csv", index=False)
    validation.pk_profiles.to_csv(output / "pd_calibrated_pk_profiles.csv", index=False)
    checks.to_csv(output / "pd_calibrated_validation_checks.csv", index=False)
    trace_df.to_csv(output / "pd_joint_calibration_trace.csv", index=False)
    (output / "pd_calibrated_validation_report.txt").write_text(report, encoding="utf-8")
    return JointCalibrationResult(
        lambda_kill_0_median=opt_lambda,
        tau_resist_days=opt_tau,
        objective_value=objective_value,
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
        calibration_trace=trace_df,
        patients=validation.patients,
        checks=checks,
        report=report,
    )


def format_resistance_refinement_report(
    pd_df: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    tau_resist_days: float,
    lambda_kill_0_median: float,
) -> str:
    """Format a report for the acquired-resistance refinement."""

    base_report = format_pd_validation_report(pd_df, checks)
    pfs_6m = survival_probability_at(pd_df, 6.0 * 30.44)
    pfs_12m = survival_probability_at(pd_df, 365.0)
    lines = [
        "Adult RCC acquired-resistance refinement",
        "",
        f"Structure: lambda_kill(t) = lambda_kill_0 * exp(-t / tau_resist)",
        f"lambda_kill_0 median: {lambda_kill_0_median:.4f} /day",
        f"tau_resist: {tau_resist_days:.1f} days",
        f"Tail effect: 6-month PFS {pfs_6m:.1%}, 12-month PFS {pfs_12m:.1%}",
        "",
        base_report,
    ]
    return "\n".join(lines)


def format_joint_calibration_report(
    pd_df: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    lambda_kill_0_median: float,
    tau_resist_days: float,
    objective_value: float,
    optimizer_success: bool,
    optimizer_message: str,
    n_calibration: int,
    n_validation: int,
) -> str:
    """Format a report for jointly calibrated acquired resistance."""

    endpoints = resistance_endpoints(pd_df)
    base_report = format_pd_validation_report(pd_df, checks)
    return "\n".join(
        [
            "Adult RCC jointly calibrated acquired-resistance model",
            "",
            "Structure: lambda_kill(t) = lambda_kill_0 * exp(-t / tau_resist)",
            f"Calibration cohort size: {n_calibration}",
            f"Validation cohort size: {n_validation}",
            f"Optimal lambda_kill_0 median: {lambda_kill_0_median:.5f} /day",
            f"Optimal tau_resist: {tau_resist_days:.2f} days",
            f"Objective value: {objective_value:.6f}",
            f"Optimizer success: {optimizer_success}",
            f"Optimizer message: {optimizer_message}",
            "",
            "Validation endpoints:",
            f"Median PFS: {endpoints['median']:.2f} months (target 4.90)",
            f"6-month PFS: {endpoints['pfs_6m']:.1%} (target 26.0%)",
            f"12-month PFS: {endpoints['pfs_12m']:.1%} (target 7.5%)",
            "",
            base_report,
        ]
    )
