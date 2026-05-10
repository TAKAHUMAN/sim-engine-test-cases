"""Responder/resistant mixture refinement for adult RCC PD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.pd_simulate import DEFAULT_PD_SEED, LAMBDA_KILL_LITERATURE_MEDIAN, simulate_pd_cohort
from simulation.pd_validate import format_pd_validation_report, survival_probability_at, validate_against_record1
from model.tumor_growth_model import DEFAULT_RESPONDER_FRACTION


@dataclass(frozen=True)
class MixtureRefinementResult:
    """Outputs from a responder/resistant mixture PD run."""

    responder_fraction: float
    tau_resist_days: float | None
    lambda_kill_0_median: float
    patients: pd.DataFrame
    checks: pd.DataFrame
    report: str


def estimate_responder_fraction_from_anchors(
    *,
    responder_median_months: float = 7.0,
    resistant_median_months: float = 2.0,
    observed_median_months: float = 4.9,
) -> tuple[float, float, float]:
    """Estimate pi from two exponential subpopulations and an observed median.

    This is an anchor-based approximation for use when RECORD-1 KM coordinates
    are unavailable. It solves:
    pi * exp(-r_resp*t) + (1-pi) * exp(-r_resist*t) = 0.5
    at t = observed median PFS.
    """

    responder_rate = np.log(2.0) / responder_median_months
    resistant_rate = np.log(2.0) / resistant_median_months
    s_resp = np.exp(-responder_rate * observed_median_months)
    s_resist = np.exp(-resistant_rate * observed_median_months)
    pi = (0.5 - s_resist) / (s_resp - s_resist)
    return float(np.clip(pi, 0.0, 1.0)), float(responder_rate), float(resistant_rate)


def run_mixture_refined_pipeline(
    *,
    n_individuals: int = 500,
    seed: int = DEFAULT_PD_SEED,
    responder_fraction: float = DEFAULT_RESPONDER_FRACTION,
    tau_resist_days: float | None = 69.0,
    lambda_kill_0_median: float = LAMBDA_KILL_LITERATURE_MEDIAN,
    output_dir: str | Path = "outputs",
    output_prefix: str = "pd_mixture",
) -> MixtureRefinementResult:
    """Run the responder/resistant mixture model."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = simulate_pd_cohort(
        N_individuals=n_individuals,
        seed=seed,
        resistance_tau_days=tau_resist_days,
        lambda_kill_median=lambda_kill_0_median,
        use_mixture_model=True,
        responder_fraction=responder_fraction,
    )
    checks = validate_against_record1(
        results.patients,
        output_dir=output,
        plot_filename=f"{output_prefix}_pfs_validation.png",
    )
    report = format_mixture_refinement_report(
        results.patients,
        checks,
        responder_fraction=responder_fraction,
        tau_resist_days=tau_resist_days,
        lambda_kill_0_median=lambda_kill_0_median,
    )
    results.patients.to_csv(output / f"{output_prefix}_pfs_results.csv", index=False)
    results.pk_profiles.to_csv(output / f"{output_prefix}_pk_profiles.csv", index=False)
    checks.to_csv(output / f"{output_prefix}_validation_checks.csv", index=False)
    (output / f"{output_prefix}_validation_report.txt").write_text(report, encoding="utf-8")
    return MixtureRefinementResult(
        responder_fraction=responder_fraction,
        tau_resist_days=tau_resist_days,
        lambda_kill_0_median=lambda_kill_0_median,
        patients=results.patients,
        checks=checks,
        report=report,
    )


def calibrate_mixture_grid(
    *,
    responder_fraction_grid: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    tau_grid_days: tuple[float, ...] = (None, 55.0, 65.0, 69.0, 75.0),
    n_individuals: int = 150,
    seed: int = DEFAULT_PD_SEED,
    lambda_kill_0_median: float = LAMBDA_KILL_LITERATURE_MEDIAN,
) -> pd.DataFrame:
    """Explore mixture fraction and resistance tau against RECORD-1 anchors."""

    rows: list[dict[str, float | None]] = []
    for pi in responder_fraction_grid:
        for tau in tau_grid_days:
            results = simulate_pd_cohort(
                N_individuals=n_individuals,
                seed=seed,
                resistance_tau_days=tau,
                lambda_kill_median=lambda_kill_0_median,
                use_mixture_model=True,
                responder_fraction=pi,
            )
            patients = results.patients
            median_pfs = float(patients["PFS_months"].median())
            pfs_6m = survival_probability_at(patients, 6.0 * 30.44)
            pfs_12m = survival_probability_at(patients, 365.0)
            score = (
                abs(median_pfs - 4.9) / 4.9
                + abs(pfs_6m - 0.26) / 0.26
                + abs(pfs_12m - 0.08) / 0.08
            )
            rows.append(
                {
                    "responder_fraction": pi,
                    "tau_resist_days": tau,
                    "lambda_kill_0_median": lambda_kill_0_median,
                    "median_pfs_months": median_pfs,
                    "pfs_6m": pfs_6m,
                    "pfs_12m": pfs_12m,
                    "combined_score": score,
                }
            )
    return pd.DataFrame.from_records(rows).sort_values("combined_score")


def format_mixture_refinement_report(
    pd_df: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    responder_fraction: float,
    tau_resist_days: float | None,
    lambda_kill_0_median: float,
) -> str:
    """Format a report for the responder/resistant mixture refinement."""

    base_report = format_pd_validation_report(pd_df, checks)
    observed_pi = float(pd_df["is_responder"].mean())
    pfs_6m = survival_probability_at(pd_df, 6.0 * 30.44)
    pfs_12m = survival_probability_at(pd_df, 365.0)
    return "\n".join(
        [
            "Adult RCC responder/resistant mixture refinement",
            "",
            f"Responder fraction pi: {responder_fraction:.2f} (observed simulated {observed_pi:.2f})",
            f"lambda_kill_0 median for responders: {lambda_kill_0_median:.4f} /day",
            f"tau_resist for responders: {tau_resist_days if tau_resist_days is not None else 'none'} days",
            "Resistant branch: drug-independent growth, PFS = ln(1.20) / lambda_growth",
            f"Tail effect: 6-month PFS {pfs_6m:.1%}, 12-month PFS {pfs_12m:.1%}",
            "",
            base_report,
        ]
    )
