"""Combined temsirolimus-sirolimus population PK model.

The unit convention is intentionally explicit:

* amounts are stored in micrograms (ug)
* volumes are stored in litres (L)
* clearances are stored in L/h
* A / V therefore has units ug/L, which is numerically equal to ng/mL

The dose covariate reported by Mizuno et al. is not applied because the
reference dose-per-kg used to center it was not published. This v1
reproduction therefore sets the dose covariate multiplier to 1.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import log, sqrt
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

MW_TEMSIROLIMUS = 1030.3
MW_SIROLIMUS = 914.2
MW_RATIO_SIR_TO_TEM = MW_SIROLIMUS / MW_TEMSIROLIMUS

DEFAULT_RNG_SEED = 20240501
DOSE_COVARIATE_MULTIPLIER = 1.0
THETA_DOSE = 0.855
DOSE_COVARIATE_NOTE = (
    "Dose covariate disabled: dose_per_kg reference was not published, "
    "so CL_TEM, Q2, and Q3 use multiplier 1.0."
)

SAMPLE_TIMES_H = np.array([0.0, 0.25, 0.5, 1.0, 3.0, 6.0, 24.0, 48.0, 72.0, 168.0])
DOSE_LEVELS_MG_M2 = (8.0, 10.0, 15.0)
DEFAULT_INFUSION_DURATION_H = 0.5


@dataclass(frozen=True)
class PKParameters:
    """Individual PK parameters for the linked parent-metabolite model."""

    CL_TEM: float
    V1: float
    Q2: float
    V2: float
    Q3: float
    V3: float
    Fm: float
    CL_SIR: float
    V4: float
    Q5: float
    V5: float


TYPICAL_70KG = PKParameters(
    CL_TEM=4.31,
    V1=18.9,
    Q2=10.4,
    V2=12.9,
    Q3=9.15,
    V3=140.0,
    Fm=0.459,
    CL_SIR=6.08,
    V4=48.0,
    Q5=11.6,
    V5=72.8,
)

IIV_CV: Mapping[str, float] = {
    "CL_TEM": 0.495,
    "V1": 0.67,
    "Q2": 0.0,
    "V2": 0.0,
    "Q3": 0.791,
    "V3": 0.0,
    "CL_SIR": 1.03,
    "V4": 1.21,
    "Q5": 0.0,
    "V5": 0.0,
}

RESIDUAL_ERROR = {
    "TEM": {"sigma_prop": 0.239, "sigma_add": 7.25},
    "SIR": {"sigma_prop": 0.255, "sigma_add": 1.69},
}

BOOTSTRAP_95CI = {
    "CL_TEM": (2.40, 6.16),
    "V1": (10.7, 26.3),
    "Q2": (7.00, 22.8),
    "V2": (7.46, 19.4),
    "Q3": (5.87, 12.3),
    "V3": (92.5, 254.0),
    "Fm": (0.389, 0.707),
    "CL_SIR": (3.15, 10.9),
    "V4": (22.9, 95.0),
    "Q5": (9.51, 22.0),
    "V5": (55.8, 140.0),
}


def cv_to_omega(cv: float) -> float:
    """Convert exponential-model CV to omega."""

    if cv < 0:
        raise ValueError("CV must be non-negative")
    return sqrt(log(1.0 + cv * cv))


def as_parameter_dict(parameters: PKParameters) -> dict[str, float]:
    """Return parameters as a plain dictionary."""

    return asdict(parameters)


def allometric_parameters(
    bw_kg: float,
    *,
    typical_70kg: PKParameters = TYPICAL_70KG,
    dose_covariate_multiplier: float = DOSE_COVARIATE_MULTIPLIER,
) -> PKParameters:
    """Scale the 70-kg typical parameters to an individual's body weight."""

    if bw_kg <= 0:
        raise ValueError("Body weight must be positive")
    if dose_covariate_multiplier <= 0:
        raise ValueError("Dose covariate multiplier must be positive")

    cl_scale = (bw_kg / 70.0) ** 0.75
    v_scale = bw_kg / 70.0

    return PKParameters(
        CL_TEM=typical_70kg.CL_TEM * cl_scale * dose_covariate_multiplier,
        V1=typical_70kg.V1 * v_scale,
        Q2=typical_70kg.Q2 * cl_scale * dose_covariate_multiplier,
        V2=typical_70kg.V2 * v_scale,
        Q3=typical_70kg.Q3 * cl_scale * dose_covariate_multiplier,
        V3=typical_70kg.V3 * v_scale,
        Fm=typical_70kg.Fm,
        CL_SIR=typical_70kg.CL_SIR * cl_scale,
        V4=typical_70kg.V4 * v_scale,
        Q5=typical_70kg.Q5 * cl_scale,
        V5=typical_70kg.V5 * v_scale,
    )


def sample_individual_parameters(
    bw_kg: float,
    rng: np.random.Generator,
    *,
    include_iiv: bool = True,
    typical_70kg: PKParameters = TYPICAL_70KG,
    dose_covariate_multiplier: float = DOSE_COVARIATE_MULTIPLIER,
) -> PKParameters:
    """Sample individual parameters using exponential inter-individual variability."""

    typical = allometric_parameters(
        bw_kg,
        typical_70kg=typical_70kg,
        dose_covariate_multiplier=dose_covariate_multiplier,
    )
    if not include_iiv:
        return typical

    values = as_parameter_dict(typical)
    for name, cv in IIV_CV.items():
        if cv == 0.0:
            continue
        omega = cv_to_omega(cv)
        values[name] *= float(np.exp(rng.normal(0.0, omega)))
    return PKParameters(**values)


def with_replaced_parameters(parameters: PKParameters, **updates: float) -> PKParameters:
    """Return a PKParameters instance with selected fields replaced."""

    return replace(parameters, **updates)


def infusion_rate_ug_per_h(t_h: float, dose_ug: float, infusion_duration_h: float) -> float:
    """Zero-order temsirolimus infusion rate into the central compartment."""

    if dose_ug < 0:
        raise ValueError("Dose must be non-negative")
    if infusion_duration_h <= 0.0 or dose_ug == 0.0:
        return 0.0
    return dose_ug / infusion_duration_h if 0.0 <= t_h <= infusion_duration_h else 0.0


def concentrations_ng_per_ml(
    amounts_ug: NDArray[np.float64], parameters: PKParameters
) -> NDArray[np.float64]:
    """Convert compartment amounts to concentrations.

    Because amounts are ug and volumes are L, each concentration is ug/L,
    which is numerically equal to ng/mL.
    """

    return np.array(
        [
            amounts_ug[0] / parameters.V1,
            amounts_ug[1] / parameters.V2,
            amounts_ug[2] / parameters.V3,
            amounts_ug[3] / parameters.V4,
            amounts_ug[4] / parameters.V5,
        ],
        dtype=float,
    )


def temsirolimus_sirolimus_ode(
    t_h: float,
    amounts_ug: NDArray[np.float64],
    parameters: PKParameters,
    dose_ug: float,
    infusion_duration_h: float,
) -> NDArray[np.float64]:
    """ODE system from the combined parent-metabolite model."""

    c1, c2, c3, c4, c5 = concentrations_ng_per_ml(amounts_ug, parameters)
    r_inf = infusion_rate_ug_per_h(t_h, dose_ug, infusion_duration_h)

    d_a1 = (
        r_inf
        - parameters.CL_TEM * c1
        - parameters.Q2 * (c1 - c2)
        - parameters.Q3 * (c1 - c3)
    )
    d_a2 = parameters.Q2 * (c1 - c2)
    d_a3 = parameters.Q3 * (c1 - c3)
    d_a4 = (
        parameters.Fm * parameters.CL_TEM * c1 * MW_RATIO_SIR_TO_TEM
        - parameters.CL_SIR * c4
        - parameters.Q5 * (c4 - c5)
    )
    d_a5 = parameters.Q5 * (c4 - c5)
    return np.array([d_a1, d_a2, d_a3, d_a4, d_a5], dtype=float)
