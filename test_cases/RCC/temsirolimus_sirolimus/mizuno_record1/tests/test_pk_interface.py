import numpy as np
import pandas as pd

from model.pk_interface import adult_rcc_parameters, get_concentration_profile, get_typical_pk_profile


def test_adult_fm_and_profile_generation() -> None:
    params = adult_rcc_parameters(70.0)
    assert np.isclose(params.Fm, 0.70)

    times, tem, sir = get_typical_pk_profile(75.0)
    assert times[0] == 0.0
    assert times[-1] == 168.0
    assert tem.max() > 0.0
    assert sir.max() > 0.0


def test_get_concentration_profile_extracts_sorted_arrays() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "time_h": [1.0, 0.0],
            "conc_TEM_pred": [2.0, 1.0],
            "conc_SIR_pred": [4.0, 3.0],
        }
    )
    times, tem, sir = get_concentration_profile(df, 1)
    assert list(times) == [0.0, 1.0]
    assert list(tem) == [1.0, 2.0]
    assert list(sir) == [3.0, 4.0]
