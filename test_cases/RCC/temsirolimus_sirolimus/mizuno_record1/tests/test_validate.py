from model.pk_model import BOOTSTRAP_95CI
from simulation.pk_simulate import SimulationConfig, run_virtual_trial
from simulation.pk_validate import bootstrap_ci_checks, concentration_summary, recover_typical_parameters


def test_summary_and_ci_checks_small_run() -> None:
    results = run_virtual_trial(SimulationConfig(n_per_dose=3, include_residual_error=False))
    summary = concentration_summary(results.profiles)
    ci = bootstrap_ci_checks()

    assert set(summary["analyte"]) == {"temsirolimus", "sirolimus"}
    assert len(ci) == len(BOOTSTRAP_95CI)
    assert ci["pass"].all()


def test_recover_typical_parameters_schema_small_run() -> None:
    results = run_virtual_trial(SimulationConfig(n_per_dose=3, include_residual_error=False))
    recovered = recover_typical_parameters(results.profiles, results.aucs)

    assert set(recovered["parameter"]) == {"CL_TEM", "V1", "CL_SIR", "V4", "Fm"}
    assert recovered["estimated"].notna().all()
