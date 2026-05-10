import numpy as np

from model.biomarker_model import BiomarkerModel, compute_molar_concentration, s6k1_inhibition_emax


def test_molar_conversion_and_emax_response() -> None:
    c_um = compute_molar_concentration(1030.3, 0.0)
    assert np.isclose(c_um, 1.0)

    inhibition = s6k1_inhibition_emax(np.array([0.0, 0.010, 1.0]))
    assert np.isclose(inhibition[0], 0.0)
    assert np.isclose(inhibition[1], 0.95 / 2.0)
    assert inhibition[2] > 0.94


def test_biomarker_model_evaluate_shapes() -> None:
    model = BiomarkerModel()
    times = np.array([0.0, 1.0])
    ps6k1, inhibition = model.evaluate(times, np.array([0.0, 1030.3]), np.zeros(2), 100.0)
    assert ps6k1.shape == times.shape
    assert inhibition.shape == times.shape
    assert ps6k1[1] < ps6k1[0]
