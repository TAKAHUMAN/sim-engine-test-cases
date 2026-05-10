"""Virtual trial simulation for the Mizuno temsirolimus-sirolimus model."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from model.pk_model import (
    DEFAULT_INFUSION_DURATION_H,
    DEFAULT_RNG_SEED,
    DOSE_COVARIATE_MULTIPLIER,
    DOSE_LEVELS_MG_M2,
    RESIDUAL_ERROR,
    SAMPLE_TIMES_H,
    THETA_DOSE,
    PKParameters,
    concentrations_ng_per_ml,
    sample_individual_parameters,
    temsirolimus_sirolimus_ode,
)

TRIAL_BW_MEAN_KG = 45.7
TRIAL_BW_SD_KG = 28.0
TRIAL_BW_MIN_KG = 7.3
TRIAL_BW_MAX_KG = 114.7
TYPICAL_BW_KG = 35.7
TYPICAL_BSA_M2 = 1.2


@dataclass(frozen=True)
class SimulationConfig:
    """Settings for the virtual trial."""

    n_per_dose: int = 1000
    seed: int = DEFAULT_RNG_SEED
    dose_levels_mg_m2: tuple[float, ...] = DOSE_LEVELS_MG_M2
    sample_times_h: tuple[float, ...] = tuple(float(x) for x in SAMPLE_TIMES_H)
    infusion_duration_h: float = DEFAULT_INFUSION_DURATION_H
    include_iiv: bool = True
    include_residual_error: bool = True
    dose_covariate_ref_mg_per_kg: float | None = None
    theta_dose: float = THETA_DOSE


@dataclass(frozen=True)
class TrialResults:
    """Primary trial outputs."""

    profiles: pd.DataFrame
    aucs: pd.DataFrame
    individual_parameters: pd.DataFrame


def costeff_bsa_m2(bw_kg: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Height-free Costeff BSA approximation in m2 from weight in kg."""

    return (4.0 * np.asarray(bw_kg) + 7.0) / (np.asarray(bw_kg) + 90.0)


def lognormal_mu_sigma(mean: float, sd: float) -> tuple[float, float]:
    """Return log-space mu and sigma for a lognormal with arithmetic mean and SD."""

    if mean <= 0 or sd <= 0:
        raise ValueError("Mean and SD must be positive")
    sigma = sqrt(log(1.0 + (sd / mean) ** 2))
    mu = log(mean) - 0.5 * sigma * sigma
    return mu, sigma


def sample_body_weights(
    n: int,
    rng: np.random.Generator,
    *,
    mean_kg: float = TRIAL_BW_MEAN_KG,
    sd_kg: float = TRIAL_BW_SD_KG,
    min_kg: float = TRIAL_BW_MIN_KG,
    max_kg: float = TRIAL_BW_MAX_KG,
) -> NDArray[np.float64]:
    """Sample truncated lognormal body weights matching the reported mean and SD."""

    if n <= 0:
        raise ValueError("n must be positive")
    mu, sigma = lognormal_mu_sigma(mean_kg, sd_kg)
    values = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        draw_count = max((n - filled) * 3, 100)
        draws = rng.lognormal(mu, sigma, size=draw_count)
        keep = draws[(draws >= min_kg) & (draws <= max_kg)]
        take = min(keep.size, n - filled)
        if take:
            values[filled : filled + take] = keep[:take]
            filled += take
    return values


def dose_mg_from_mg_m2(dose_mg_m2: float, bsa_m2: float) -> float:
    """Convert a protocol dose in mg/m2 to mg."""

    if dose_mg_m2 <= 0 or bsa_m2 <= 0:
        raise ValueError("Dose level and BSA must be positive")
    return dose_mg_m2 * bsa_m2


def dose_covariate_multiplier(
    dose_mg: float,
    bw_kg: float,
    *,
    ref_mg_per_kg: float | None,
    theta_dose: float = THETA_DOSE,
) -> float:
    """Return the Mizuno dose covariate multiplier for CL_TEM, Q2, and Q3.

    If ref_mg_per_kg is None, the documented v1 simplification is used and
    the multiplier is fixed to 1.0.
    """

    if ref_mg_per_kg is None:
        return DOSE_COVARIATE_MULTIPLIER
    if dose_mg <= 0.0 or bw_kg <= 0.0 or ref_mg_per_kg <= 0.0:
        raise ValueError("Dose, body weight, and reference dose-per-kg must be positive")
    dose_per_kg = dose_mg / bw_kg
    return float((dose_per_kg / ref_mg_per_kg) ** theta_dose)


def solve_amounts(
    parameters: PKParameters,
    dose_mg: float,
    sample_times_h: Iterable[float] = SAMPLE_TIMES_H,
    *,
    infusion_duration_h: float = DEFAULT_INFUSION_DURATION_H,
) -> NDArray[np.float64]:
    """Solve the ODE with LSODA and return amounts at requested times."""

    times = np.asarray(tuple(sample_times_h), dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("sample_times_h must be a non-empty one-dimensional sequence")
    if np.any(np.diff(times) < 0):
        raise ValueError("sample_times_h must be sorted")
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")

    dose_ug = dose_mg * 1000.0
    y0 = np.zeros(5, dtype=float)
    solution = solve_ivp(
        lambda t, y: temsirolimus_sirolimus_ode(
            t, y, parameters, dose_ug, infusion_duration_h
        ),
        (float(times[0]), float(times[-1])),
        y0,
        method="LSODA",
        t_eval=times,
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solve failed: {solution.message}")
    return np.asarray(solution.y.T, dtype=float)


def predicted_concentrations(
    amounts_ug: NDArray[np.float64], parameters: PKParameters
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return temsirolimus and sirolimus central concentrations in ng/mL."""

    concentrations = np.vstack(
        [concentrations_ng_per_ml(row, parameters) for row in np.asarray(amounts_ug)]
    )
    return concentrations[:, 0], concentrations[:, 3]


def apply_residual_error(
    prediction: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    sigma_prop: float,
    sigma_add: float,
) -> NDArray[np.float64]:
    """Apply the specified combined additive plus proportional residual model."""

    eps_prop = rng.normal(0.0, sigma_prop, size=prediction.shape)
    eps_add = rng.normal(0.0, sigma_add, size=prediction.shape)
    return prediction * (1.0 + eps_prop) + eps_add


def auc_linear_trapezoidal(time_h: NDArray[np.float64], conc_ng_ml: NDArray[np.float64]) -> float:
    """Linear trapezoidal AUC through the last sampled time."""

    return float(np.trapezoid(conc_ng_ml, time_h))


def terminal_lambda_z(
    time_h: NDArray[np.float64], conc_ng_ml: NDArray[np.float64], n_points: int = 3
) -> float:
    """Estimate terminal slope from the last positive concentrations."""

    positive = np.where((conc_ng_ml > 0.0) & np.isfinite(conc_ng_ml))[0]
    if positive.size < n_points:
        return float("nan")
    idx = positive[-n_points:]
    slope, _ = np.polyfit(time_h[idx], np.log(conc_ng_ml[idx]), 1)
    lambda_z = -float(slope)
    return lambda_z if lambda_z > 0.0 else float("nan")


def auc_to_infinity(
    time_h: NDArray[np.float64], conc_ng_ml: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Compute AUC0-inf by linear trapezoid plus terminal extrapolation."""

    auc_last = auc_linear_trapezoidal(time_h, conc_ng_ml)
    lambda_z = terminal_lambda_z(time_h, conc_ng_ml)
    last_positive = np.where((conc_ng_ml > 0.0) & np.isfinite(conc_ng_ml))[0]
    if not np.isfinite(lambda_z) or last_positive.size == 0:
        return auc_last, auc_last, lambda_z
    c_last = float(conc_ng_ml[last_positive[-1]])
    return auc_last + c_last / lambda_z, auc_last, lambda_z


def simulate_individual(
    parameters: PKParameters,
    dose_mg: float,
    rng: np.random.Generator,
    *,
    sample_times_h: Iterable[float] = SAMPLE_TIMES_H,
    infusion_duration_h: float = DEFAULT_INFUSION_DURATION_H,
    include_residual_error: bool = True,
) -> dict[str, NDArray[np.float64] | float]:
    """Simulate one individual's profile and AUCs."""

    times = np.asarray(tuple(sample_times_h), dtype=float)
    amounts = solve_amounts(
        parameters,
        dose_mg,
        times,
        infusion_duration_h=infusion_duration_h,
    )
    tem_pred, sir_pred = predicted_concentrations(amounts, parameters)
    if include_residual_error:
        tem_obs = apply_residual_error(tem_pred, rng, **RESIDUAL_ERROR["TEM"])
        sir_obs = apply_residual_error(sir_pred, rng, **RESIDUAL_ERROR["SIR"])
    else:
        tem_obs = tem_pred.copy()
        sir_obs = sir_pred.copy()

    auc_tem, auc_tem_last, lambda_tem = auc_to_infinity(times, tem_pred)
    auc_sir, auc_sir_last, lambda_sir = auc_to_infinity(times, sir_pred)
    return {
        "time_h": times,
        "conc_TEM_pred": tem_pred,
        "conc_TEM_obs": tem_obs,
        "conc_SIR_pred": sir_pred,
        "conc_SIR_obs": sir_obs,
        "auc_TEM": auc_tem,
        "auc_TEM_last": auc_tem_last,
        "lambda_z_TEM": lambda_tem,
        "auc_SIR": auc_sir,
        "auc_SIR_last": auc_sir_last,
        "lambda_z_SIR": lambda_sir,
        "auc_ratio_SIR_TEM": auc_sir / auc_tem if auc_tem > 0.0 else float("nan"),
    }


def run_virtual_trial(config: SimulationConfig = SimulationConfig()) -> TrialResults:
    """Run the requested N-patient by three-dose-level virtual trial."""

    rng = np.random.default_rng(config.seed)
    profile_records: list[dict[str, float | int]] = []
    auc_records: list[dict[str, float | int]] = []
    parameter_records: list[dict[str, float | int]] = []
    patient_id = 1

    for dose_level in config.dose_levels_mg_m2:
        body_weights = sample_body_weights(config.n_per_dose, rng)
        for bw_kg in body_weights:
            bsa_m2 = float(costeff_bsa_m2(bw_kg))
            dose_mg = dose_mg_from_mg_m2(dose_level, bsa_m2)
            dose_per_kg = dose_mg / bw_kg
            dose_multiplier = dose_covariate_multiplier(
                dose_mg,
                bw_kg,
                ref_mg_per_kg=config.dose_covariate_ref_mg_per_kg,
                theta_dose=config.theta_dose,
            )
            parameters = sample_individual_parameters(
                bw_kg,
                rng,
                include_iiv=config.include_iiv,
                dose_covariate_multiplier=dose_multiplier,
            )
            simulated = simulate_individual(
                parameters,
                dose_mg,
                rng,
                sample_times_h=config.sample_times_h,
                infusion_duration_h=config.infusion_duration_h,
                include_residual_error=config.include_residual_error,
            )

            times = np.asarray(simulated["time_h"], dtype=float)
            for row_index, time_h in enumerate(times):
                profile_records.append(
                    {
                        "id": patient_id,
                        "dose_mg_m2": float(dose_level),
                        "BW": float(bw_kg),
                        "time_h": float(time_h),
                        "conc_TEM_pred": float(
                            np.asarray(simulated["conc_TEM_pred"])[row_index]
                        ),
                        "conc_TEM_obs": float(
                            np.asarray(simulated["conc_TEM_obs"])[row_index]
                        ),
                        "conc_SIR_pred": float(
                            np.asarray(simulated["conc_SIR_pred"])[row_index]
                        ),
                        "conc_SIR_obs": float(
                            np.asarray(simulated["conc_SIR_obs"])[row_index]
                        ),
                    }
                )

            auc_records.append(
                {
                    "id": patient_id,
                    "dose_mg_m2": float(dose_level),
                    "BW": float(bw_kg),
                    "BSA": bsa_m2,
                    "dose_mg": dose_mg,
                    "dose_per_kg": dose_per_kg,
                    "dose_covariate_multiplier": dose_multiplier,
                    "auc_TEM": float(simulated["auc_TEM"]),
                    "auc_TEM_last": float(simulated["auc_TEM_last"]),
                    "lambda_z_TEM": float(simulated["lambda_z_TEM"]),
                    "auc_SIR": float(simulated["auc_SIR"]),
                    "auc_SIR_last": float(simulated["auc_SIR_last"]),
                    "lambda_z_SIR": float(simulated["lambda_z_SIR"]),
                    "auc_ratio_SIR_TEM": float(simulated["auc_ratio_SIR_TEM"]),
                }
            )
            parameter_row = {"id": patient_id, "dose_mg_m2": float(dose_level), "BW": float(bw_kg)}
            parameter_row.update({key: float(value) for key, value in parameters.__dict__.items()})
            parameter_records.append(parameter_row)
            patient_id += 1

    profiles = pd.DataFrame.from_records(
        profile_records,
        columns=[
            "id",
            "dose_mg_m2",
            "BW",
            "time_h",
            "conc_TEM_pred",
            "conc_TEM_obs",
            "conc_SIR_pred",
            "conc_SIR_obs",
        ],
    )
    aucs = pd.DataFrame.from_records(auc_records)
    individual_parameters = pd.DataFrame.from_records(parameter_records)
    return TrialResults(profiles=profiles, aucs=aucs, individual_parameters=individual_parameters)


def simulate_typical_profiles(
    dose_levels_mg_m2: Iterable[float] = DOSE_LEVELS_MG_M2,
    *,
    bw_kg: float = TYPICAL_BW_KG,
    bsa_m2: float = TYPICAL_BSA_M2,
    sample_times_h: Iterable[float] = SAMPLE_TIMES_H,
    infusion_duration_h: float = DEFAULT_INFUSION_DURATION_H,
) -> pd.DataFrame:
    """Simulate median-patient typical-value profiles without IIV or residual error."""

    from model.pk_model import allometric_parameters

    rng = np.random.default_rng(DEFAULT_RNG_SEED)
    records: list[dict[str, float | int]] = []
    patient_id = 1
    parameters = allometric_parameters(bw_kg)
    for dose_level in dose_levels_mg_m2:
        dose_mg = dose_mg_from_mg_m2(float(dose_level), bsa_m2)
        simulated = simulate_individual(
            parameters,
            dose_mg,
            rng,
            sample_times_h=sample_times_h,
            infusion_duration_h=infusion_duration_h,
            include_residual_error=False,
        )
        for row_index, time_h in enumerate(np.asarray(simulated["time_h"], dtype=float)):
            records.append(
                {
                    "id": patient_id,
                    "dose_mg_m2": float(dose_level),
                    "BW": bw_kg,
                    "time_h": float(time_h),
                    "conc_TEM_pred": float(np.asarray(simulated["conc_TEM_pred"])[row_index]),
                    "conc_TEM_obs": float(np.asarray(simulated["conc_TEM_obs"])[row_index]),
                    "conc_SIR_pred": float(np.asarray(simulated["conc_SIR_pred"])[row_index]),
                    "conc_SIR_obs": float(np.asarray(simulated["conc_SIR_obs"])[row_index]),
                }
            )
        patient_id += 1
    return pd.DataFrame.from_records(records)


def summarize_auc_by_dose(aucs: pd.DataFrame) -> pd.DataFrame:
    """Summarize AUCs and sirolimus:temsirolimus AUC ratio by dose level."""

    return (
        aucs.groupby("dose_mg_m2", as_index=False)
        .agg(
            auc_TEM_median=("auc_TEM", "median"),
            auc_TEM_sd=("auc_TEM", "std"),
            auc_SIR_median=("auc_SIR", "median"),
            auc_SIR_sd=("auc_SIR", "std"),
            auc_ratio_mean=("auc_ratio_SIR_TEM", "mean"),
            auc_ratio_median=("auc_ratio_SIR_TEM", "median"),
            auc_ratio_sd=("auc_ratio_SIR_TEM", "std"),
        )
        .sort_values("dose_mg_m2")
    )
