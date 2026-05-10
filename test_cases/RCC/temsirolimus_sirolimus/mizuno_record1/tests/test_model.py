import numpy as np
from scipy.integrate import solve_ivp

from model.pk_model import (
    MW_RATIO_SIR_TO_TEM,
    TYPICAL_70KG,
    allometric_parameters,
    concentrations_ng_per_ml,
    temsirolimus_sirolimus_ode,
)


def test_typical_70kg_parameters_are_locked() -> None:
    params = allometric_parameters(70.0)
    assert params == TYPICAL_70KG


def test_bolus_mass_balance_with_elimination_integrals() -> None:
    params = TYPICAL_70KG
    dose_ug = 25_000.0

    def augmented_rhs(t: float, y: np.ndarray) -> np.ndarray:
        amounts = y[:5]
        c1, _, _, c4, _ = concentrations_ng_per_ml(amounts, params)
        rhs = temsirolimus_sirolimus_ode(t, amounts, params, 0.0, 0.0)
        parent_true_elim = (1.0 - params.Fm) * params.CL_TEM * c1
        parent_to_metabolite = params.Fm * params.CL_TEM * c1
        sir_formed = parent_to_metabolite * MW_RATIO_SIR_TO_TEM
        sir_elim = params.CL_SIR * c4
        return np.concatenate([rhs, [parent_true_elim, parent_to_metabolite, sir_formed, sir_elim]])

    y0 = np.zeros(9)
    y0[0] = dose_ug
    solution = solve_ivp(augmented_rhs, (0.0, 500.0), y0, method="LSODA", rtol=1e-8, atol=1e-9)

    final = solution.y[:, -1]
    parent_balance = final[0] + final[1] + final[2] + final[5] + final[6]
    sirolimus_balance = final[3] + final[4] + final[8]

    assert np.isclose(parent_balance, dose_ug, rtol=1e-5, atol=1e-3)
    assert np.isclose(sirolimus_balance, final[7], rtol=1e-5, atol=1e-3)
