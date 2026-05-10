"""S6K1 biomarker inhibition model for active temsirolimus plus sirolimus."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MW_TEM = 1030.3
MW_SIR = 914.2
EC50_UM = 0.010
EMAX = 0.95
HILL_GAMMA = 1.2


def compute_molar_concentration(
    tem_ng_ml: float | NDArray[np.float64],
    sir_ng_ml: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Convert PK concentrations in ng/mL to molar-summed active drug in uM.

    ng/mL is numerically equivalent to ug/L. Since molecular weight in g/mol is
    numerically equivalent to ug/umol, ng/mL divided by MW gives umol/L, or uM.
    """

    tem = np.asarray(tem_ng_ml, dtype=float)
    sir = np.asarray(sir_ng_ml, dtype=float)
    return (tem / MW_TEM) + (sir / MW_SIR)


def s6k1_inhibition_emax(
    c_um: float | NDArray[np.float64],
    EC50: float = EC50_UM,
    Emax: float = EMAX,
    gamma: float = HILL_GAMMA,
) -> float | NDArray[np.float64]:
    """Hill equation for fractional S6K1 inhibition."""

    c = np.maximum(np.asarray(c_um, dtype=float), 0.0)
    numerator = Emax * np.power(c, gamma)
    denominator = np.power(EC50, gamma) + np.power(c, gamma)
    return np.divide(numerator, denominator, out=np.zeros_like(c), where=denominator > 0.0)


def pS6K1_observed(
    inhibition: float | NDArray[np.float64], pS6K1_0: float
) -> float | NDArray[np.float64]:
    """Observed pS6K1 = baseline times one minus inhibition."""

    return pS6K1_0 * (1.0 - np.asarray(inhibition, dtype=float))


class BiomarkerModel:
    """Emax-Hill model for pS6K1 suppression."""

    def __init__(self, EC50: float = EC50_UM, Emax: float = EMAX, gamma: float = HILL_GAMMA):
        self.EC50 = EC50
        self.Emax = Emax
        self.gamma = gamma

    def evaluate(
        self,
        times: NDArray[np.float64],
        TEM_conc_array: NDArray[np.float64],
        SIR_conc_array: NDArray[np.float64],
        pS6K1_0: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute pS6K1(t) and fractional inhibition over a time array."""

        _ = times
        c_array = compute_molar_concentration(TEM_conc_array, SIR_conc_array)
        e_array = s6k1_inhibition_emax(c_array, self.EC50, self.Emax, self.gamma)
        ps6k1_array = pS6K1_observed(e_array, pS6K1_0)
        return np.asarray(ps6k1_array, dtype=float), np.asarray(e_array, dtype=float)
