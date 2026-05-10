import numpy as np

from model.pk_model import SAMPLE_TIMES_H, allometric_parameters
from simulation.pk_simulate import (
    SimulationConfig,
    costeff_bsa_m2,
    dose_covariate_multiplier,
    run_virtual_trial,
    simulate_individual,
)


def test_costeff_bsa_and_single_profile_are_positive() -> None:
    bsa = costeff_bsa_m2(35.7)
    assert np.isclose(bsa, (4.0 * 35.7 + 7.0) / (35.7 + 90.0))

    params = allometric_parameters(70.0)
    simulated = simulate_individual(
        params,
        25.0,
        np.random.default_rng(20240501),
        sample_times_h=SAMPLE_TIMES_H,
        include_residual_error=False,
    )
    assert np.asarray(simulated["conc_TEM_pred"])[0] == 0.0
    assert np.asarray(simulated["conc_SIR_pred"])[0] == 0.0
    assert simulated["auc_TEM"] > 0.0
    assert simulated["auc_SIR"] > 0.0


def test_dose_covariate_multiplier_defaults_to_v1_simplification() -> None:
    assert dose_covariate_multiplier(18.0, 35.7, ref_mg_per_kg=None) == 1.0
    active = dose_covariate_multiplier(18.0, 35.7, ref_mg_per_kg=0.27)
    assert active > 1.0


def test_virtual_trial_schema_small_run() -> None:
    results = run_virtual_trial(SimulationConfig(n_per_dose=2, include_residual_error=False))
    assert list(results.profiles.columns) == [
        "id",
        "dose_mg_m2",
        "BW",
        "time_h",
        "conc_TEM_pred",
        "conc_TEM_obs",
        "conc_SIR_pred",
        "conc_SIR_obs",
    ]
    assert len(results.aucs) == 6
    assert results.profiles["id"].nunique() == 6
