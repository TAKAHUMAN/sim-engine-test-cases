"""Entry point for the temsirolimus-sirolimus popPK reproduction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.pk_model import DOSE_COVARIATE_NOTE
from model.pd_resistance import run_joint_calibration
from simulation.pd_simulate import run_pd_pipeline
from simulation.pd_validate import format_pd_validation_report, validate_against_record1
from simulation.pk_simulate import SimulationConfig, run_virtual_trial, summarize_auc_by_dose
from simulation.pk_validate import paper_auc_ratio_target, validation_outputs


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_one_page_report(
    recovered: pd.DataFrame,
    auc_summary: pd.DataFrame,
    ci_checks: pd.DataFrame,
    *,
    pcvpc_path: str | Path,
    sensitivity_summary: pd.DataFrame | None = None,
    sensitivity_ref_mg_per_kg: float | None = None,
) -> str:
    """Format the requested Section 9 validation report."""

    recovered_table = recovered.copy()
    recovered_table["estimated"] = recovered_table["estimated"].map(lambda x: _format_float(x, 3))
    recovered_table["paper"] = recovered_table["paper"].map(lambda x: _format_float(x, 3))
    recovered_table["deviation_pct"] = recovered_table["deviation_pct"].map(lambda x: _format_float(x, 1))

    targets = paper_auc_ratio_target()
    auc_table = auc_summary.copy()
    auc_table["paper_ratio"] = auc_table["dose_mg_m2"].map(targets)
    for column in ["auc_ratio_mean", "auc_ratio_median", "auc_ratio_sd"]:
        auc_table[column] = auc_table[column].map(lambda x: _format_float(x, 3))
    auc_table = auc_table[
        ["dose_mg_m2", "auc_ratio_mean", "auc_ratio_median", "auc_ratio_sd", "paper_ratio"]
    ]

    ci_table = ci_checks.copy()
    ci_table["status"] = ci_table["pass"].map(lambda passed: "PASS" if passed else "FAIL")
    ci_table = ci_table[["parameter", "typical_value", "ci_low", "ci_high", "status"]]

    lines = [
        "Mizuno 2016 temsirolimus-sirolimus reproduction validation",
        "",
        f"Known simplification: {DOSE_COVARIATE_NOTE}",
        f"pcVPC-style plot: {pcvpc_path}",
        "",
        "Estimated typical parameters vs paper:",
        recovered_table.to_string(index=False),
        "",
        "Sirolimus:temsirolimus AUC0-inf ratio by dose level:",
        auc_table.to_string(index=False),
        "",
        "Bootstrap 95% CI checks for Table 3 typical parameters:",
        ci_table.to_string(index=False),
    ]
    if sensitivity_summary is not None:
        sensitivity_table = sensitivity_summary.copy()
        for column in [
            "auc_ratio_mean",
            "auc_ratio_median",
            "auc_ratio_sd",
            "dose_covariate_multiplier_median",
            "dose_covariate_multiplier_p5",
            "dose_covariate_multiplier_p95",
        ]:
            sensitivity_table[column] = sensitivity_table[column].map(lambda x: _format_float(x, 3))
        lines.extend(
            [
                "",
                (
                    "15 mg/m2 dose-covariate sensitivity "
                    f"(ref={sensitivity_ref_mg_per_kg:g} mg/kg):"
                ),
                sensitivity_table.to_string(index=False),
            ]
        )
    return "\n".join(lines)


def run_dose_covariate_sensitivity(
    *,
    n_per_dose: int = 1000,
    seed: int = 20240501,
    ref_mg_per_kg: float = 0.27,
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """Run only the 15 mg/m2 arm with the published dose-covariate formula enabled."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = run_virtual_trial(
        SimulationConfig(
            n_per_dose=n_per_dose,
            seed=seed,
            dose_levels_mg_m2=(15.0,),
            dose_covariate_ref_mg_per_kg=ref_mg_per_kg,
        )
    )
    results.profiles.to_csv(out / "sensitivity_15mgm2_profiles.csv", index=False)
    results.aucs.to_csv(out / "sensitivity_15mgm2_auc.csv", index=False)
    results.individual_parameters.to_csv(
        out / "sensitivity_15mgm2_individual_parameters.csv", index=False
    )

    auc_summary = summarize_auc_by_dose(results.aucs)
    multiplier_summary = (
        results.aucs.groupby("dose_mg_m2", as_index=False)
        .agg(
            dose_covariate_multiplier_median=("dose_covariate_multiplier", "median"),
            dose_covariate_multiplier_p5=(
                "dose_covariate_multiplier",
                lambda x: x.quantile(0.05),
            ),
            dose_covariate_multiplier_p95=(
                "dose_covariate_multiplier",
                lambda x: x.quantile(0.95),
            ),
        )
        .sort_values("dose_mg_m2")
    )
    sensitivity = auc_summary.merge(multiplier_summary, on="dose_mg_m2", how="left")
    sensitivity.to_csv(out / "sensitivity_15mgm2_summary.csv", index=False)
    return sensitivity


def run_pipeline(
    *,
    n_per_dose: int = 1000,
    seed: int = 20240501,
    output_dir: str | Path = "outputs",
) -> str:
    """Run the full simulation and validation pipeline."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = run_virtual_trial(SimulationConfig(n_per_dose=n_per_dose, seed=seed))
    validation = validation_outputs(results.profiles, results.aucs, output_dir=out)
    sensitivity = run_dose_covariate_sensitivity(
        n_per_dose=n_per_dose,
        seed=seed,
        ref_mg_per_kg=0.27,
        output_dir=out,
    )

    results.profiles.to_csv(out / "individual_profiles.csv", index=False)
    results.aucs.to_csv(out / "individual_auc.csv", index=False)
    results.individual_parameters.to_csv(out / "individual_parameters.csv", index=False)
    validation["summary"].to_csv(out / "summary_statistics.csv", index=False)

    return build_one_page_report(
        validation["recovered"],
        validation["auc_summary"],
        validation["ci_checks"],
        pcvpc_path=validation["pcvpc_path"],
        sensitivity_summary=sensitivity,
        sensitivity_ref_mg_per_kg=0.27,
    )


def run_pd_validation_pipeline(
    *,
    n_individuals: int = 500,
    seed: int = 20240501,
    output_dir: str | Path = "outputs",
) -> str:
    """Run the adult RCC PD model and return the RECORD-1 validation report."""

    pd_results = run_pd_pipeline(N_individuals=n_individuals, seed=seed, output_dir=output_dir)
    checks = validate_against_record1(pd_results.patients, output_dir=output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checks.to_csv(Path(output_dir) / "pd_validation_checks.csv", index=False)
    report = format_pd_validation_report(pd_results.patients, checks)
    (Path(output_dir) / "pd_validation_report.txt").write_text(report, encoding="utf-8")
    return report


def run_calibrated_pd_pipeline(
    *,
    n_individuals: int = 500,
    seed: int = 20240501,
    output_dir: str | Path = "outputs",
) -> str:
    """Run the final calibrated acquired-resistance PD model."""

    result = run_joint_calibration(
        n_calibration=n_individuals,
        n_validation=n_individuals,
        seed=seed,
        x0=(0.02455, 39.5),
        coarse_lambda_grid=(0.02455,),
        coarse_tau_grid=(39.5,),
        maxiter=0,
        output_dir=output_dir,
    )
    return result.report


def main() -> None:
    pk_report = run_pipeline()
    print(pk_report)
    print()
    pd_report = run_pd_validation_pipeline()
    print(pd_report)
    print()
    calibrated_report = run_calibrated_pd_pipeline()
    print(calibrated_report)


if __name__ == "__main__":
    main()
