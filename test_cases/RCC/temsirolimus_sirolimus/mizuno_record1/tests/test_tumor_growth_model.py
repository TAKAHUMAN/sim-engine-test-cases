import numpy as np

from model.tumor_growth_model import MixtureResponderModel, TumorGrowthModel


def test_tumor_pfs_without_drug_matches_growth_threshold() -> None:
    model = TumorGrowthModel(
        lambda_growth=0.01,
        lambda_kill=0.0,
        biomarker_model=None,
        times=np.array([0.0, 7.0]),
        pS6K1_array=np.array([100.0, 100.0]),
        pS6K1_0=100.0,
    )
    pfs_days, progressed = model.simulate_pfs(np.log(70.0), t_max=365.0)
    assert progressed
    assert np.isclose(pfs_days, np.log(1.20) / 0.01, rtol=0.02)


def test_resistance_reduces_effective_drug_suppression_over_time() -> None:
    model = TumorGrowthModel(
        lambda_growth=0.01,
        lambda_kill=0.02,
        biomarker_model=None,
        times=np.array([0.0, 7.0]),
        pS6K1_array=np.array([20.0, 20.0]),
        pS6K1_0=100.0,
        resistance_tau_days=30.0,
    )
    early = model.growth_rate(0.0, np.array([np.log(70.0)]))[0]
    late = model.growth_rate(90.0, np.array([np.log(70.0)]))[0]
    assert late > early


def test_mixture_resistant_branch_uses_baseline_growth() -> None:
    mixture = MixtureResponderModel(responder_fraction=0.0)
    pfs_days, progressed, is_responder = mixture.simulate_pfs_mixture(
        L_0=np.log(70.0),
        lambda_growth=0.01,
        lambda_kill_0=0.02,
        biomarker_model=None,
        times=np.array([0.0, 7.0]),
        pS6K1_array=np.array([20.0, 20.0]),
        pS6K1_0=100.0,
        rng=np.random.default_rng(20240501),
        tau_resist_days=69.0,
    )
    assert progressed
    assert not is_responder
    assert np.isclose(pfs_days, np.log(1.20) / 0.01)
