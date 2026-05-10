from simulation.pd_simulate import simulate_pd_cohort


def test_pd_cohort_small_run_schema() -> None:
    results = simulate_pd_cohort(N_individuals=2, seed=20240501)
    assert len(results.patients) == 2
    assert results.pk_profiles["id"].nunique() == 2
    assert {"PFS_days", "PFS_months", "median_s6k1_inhibition"}.issubset(
        results.patients.columns
    )


def test_pd_cohort_mixture_schema() -> None:
    results = simulate_pd_cohort(N_individuals=3, seed=20240501, use_mixture_model=True)
    assert "is_responder" in results.patients.columns
    assert results.patients["use_mixture_model"].all()
