"""Dovitinib (TKI258) aggregate PK/PD model for RCC.

This workflow calibrates an oral two-compartment PK model with saturable
elimination and time-varying autoinduced clearance to the Escudier et al. 2014
published aggregate PK summaries. It then links the PK profile to sVEGFR1,
sVEGFR2, FGF23, and VEGF biomarker changes through turnover Emax models.

Outputs are written next to this script under ./results and ./results/figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


CASE_DIR = Path(__file__).resolve().parent
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from pkpd_model.fit import (  # noqa: E402
    calibrate_pd_models,
    fit_pk_model,
    load_pd_targets,
    load_pk_targets,
    residual_error_table,
)
from pkpd_model.model import (  # noqa: E402
    PK_PARAMETER_NAMES,
    ResidualError,
    population_prediction_grid,
    simulate_sparse_population,
)
from pkpd_model.plots import (  # noqa: E402
    save_concentration_profiles,
    save_individual_predictions,
    save_observed_vs_predicted,
    save_pd_response,
    save_random_effects,
    save_residuals_vs_predicted,
    save_vpc,
)


DATA_DIR = CASE_DIR / "data"
RESULTS_DIR = CASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"


def write_equations(path: Path) -> None:
    path.write_text(
        """# Dovitinib Aggregate PK/PD Model Equations

## PK Model

Gut amount: `A_gut`, central amount: `A_c`, peripheral amount: `A_p`.

`C = A_c / Vc`

`CL(t) = CL_day1 + (CL_day15 - CL_day1) * (1 - exp(-kaut * t))`

`Vmax(t) = CL(t) * Km`

`dA_gut/dt = -Ka * A_gut`

`dA_c/dt = F * Ka * A_gut - (Q/Vc) * A_c + (Q/Vp) * A_p - Vmax(t) * C / (Km + C)`

`dA_p/dt = (Q/Vc) * A_c - (Q/Vp) * A_p`

The model is parameterized in apparent oral terms because absolute oral
bioavailability is not identifiable from aggregate oral summaries.

## Residual Error

`Y = IPRED * (1 + eps_prop) + eps_add`

## PD Turnover Model

For inhibition markers:

`I(C) = Emax * C / (EC50 + C)`

`dR/dt = kout * (1 - I(C) + escape(t) - R)`

For stimulation markers:

`S(C) = Emax * C / (EC50 + C)`

`dR/dt = kout * (1 + S(C) + escape(t) - R)`

`escape(t) = escape_max * (1 - exp(-escape_k * t))`

Reported PD output is `(R - 1) * 100`, the percent change from baseline.
""",
        encoding="utf-8",
    )


def covariate_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "covariate": "baseline_vegfr_inhibitor",
                "implemented_as": "Pseudo-population indicator; 49/65 set to previously treated",
                "model_effect": "Recorded for stratification; PK effect not estimated from aggregate summaries",
            },
            {
                "covariate": "prior_count_group",
                "implemented_as": "Pseudo-population indicator; 82.1% assigned >=2 prior regimens",
                "model_effect": "Recorded for stratification; PK effect not estimated from aggregate summaries",
            },
            {
                "covariate": "body_weight_kg",
                "implemented_as": "Pseudo-population continuous covariate",
                "model_effect": "Vc and Vp scaled by WT/70",
            },
            {
                "covariate": "creatinine_clearance_ml_min",
                "implemented_as": "Pseudo-population continuous covariate",
                "model_effect": "Recorded for assessment; clearance effect not estimated without raw renal function data",
            },
        ]
    )


def run_workflow() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pk_targets = load_pk_targets(DATA_DIR / "paper_pk_targets.csv")
    pd_targets = load_pd_targets(DATA_DIR / "paper_pd_targets.csv")

    pk_params, pk_parameter_table, pk_comparison, diagnostics = fit_pk_model(pk_targets)
    residual_error = ResidualError()
    iiv_cv = {name: 0.30 for name in PK_PARAMETER_NAMES}

    sparse, eta = simulate_sparse_population(
        pk_params,
        n_patients=65,
        iiv_cv=iiv_cv,
        residual_error=residual_error,
        seed=258,
    )
    population_grid = population_prediction_grid(
        pk_params,
        n_patients=250,
        iiv_cv=iiv_cv,
        seed=80258,
        dt_h=0.5,
    )
    pd_parameter_table, pd_comparison = calibrate_pd_models(pk_params, pd_targets)

    diagnostics_table = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in diagnostics.items()]
        + [
            {
                "metric": "raw_data_limitation",
                "value": (
                    "Published aggregate summaries only; individual likelihood, "
                    "empirical Bayes ETAs, and shrinkage cannot be estimated directly"
                ),
            }
        ]
    )
    shrinkage = pd.DataFrame(
        {
            "parameter": PK_PARAMETER_NAMES,
            "shrinkage_percent": [pd.NA] * len(PK_PARAMETER_NAMES),
            "note": ["Not estimable without raw individual observations and EBEs"] * len(PK_PARAMETER_NAMES),
        }
    )

    pk_parameter_table.to_csv(RESULTS_DIR / "pk_parameter_estimates.csv", index=False)
    residual_error_table(residual_error).to_csv(RESULTS_DIR / "residual_error_parameters.csv", index=False)
    pk_comparison.to_csv(RESULTS_DIR / "paper_pk_comparison.csv", index=False)
    pd_parameter_table.to_csv(RESULTS_DIR / "pd_parameter_estimates.csv", index=False)
    pd_comparison.to_csv(RESULTS_DIR / "paper_pd_comparison.csv", index=False)
    sparse.to_csv(RESULTS_DIR / "simulated_sparse_pk.csv", index=False)
    eta.to_csv(RESULTS_DIR / "random_effects.csv", index=False)
    population_grid.to_csv(RESULTS_DIR / "vpc_population_grid.csv", index=False)
    diagnostics_table.to_csv(RESULTS_DIR / "model_diagnostics.csv", index=False)
    shrinkage.to_csv(RESULTS_DIR / "shrinkage_estimates.csv", index=False)
    covariate_table().to_csv(RESULTS_DIR / "covariate_analysis.csv", index=False)
    write_equations(RESULTS_DIR / "model_equations.md")

    save_observed_vs_predicted(sparse, FIG_DIR / "observed_vs_predicted.png")
    save_residuals_vs_predicted(sparse, FIG_DIR / "residuals_vs_predicted.png")
    save_random_effects(eta, FIG_DIR / "random_effects_distribution.png")
    save_concentration_profiles(pk_params, population_grid, FIG_DIR / "concentration_profiles_day1_day15.png")
    save_vpc(population_grid, pk_targets, FIG_DIR / "vpc_concentration.png")
    save_individual_predictions(sparse, pk_params, FIG_DIR / "individual_predictions.png")
    save_pd_response(pk_params, pd_parameter_table, pd_targets, FIG_DIR / "svegfr_response.png")

    print("Dovitinib PK fit complete")
    print(pk_comparison[["occasion", "metric", "paper_value", "predicted", "percent_error"]].to_string(index=False))
    print()
    print("Dovitinib PD calibration complete")
    print(pd_comparison.to_string(index=False))
    print()
    print(f"Results written to: {RESULTS_DIR}")
    print(f"Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    run_workflow()
