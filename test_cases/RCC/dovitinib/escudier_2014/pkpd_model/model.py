from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PKParameters:
    """Typical apparent oral PK parameters for dovitinib.

    Amount is tracked in mg, volume in L, concentration in ng/mL for reports.
    Michaelis-Menten elimination uses concentration in mg/L, with Vmax(t)
    defined as CL(t) * Km so CL(t) is the low-concentration apparent clearance.
    """

    ka_h: float
    vc_l: float
    q_l_h: float
    vp_l: float
    cl_day1_l_h: float
    cl_day15_l_h: float
    kaut_h: float
    km_mg_l: float
    bioavailability: float = 1.0


@dataclass(frozen=True)
class ResidualError:
    proportional: float = 0.20
    additive_ng_ml: float = 5.0


@dataclass(frozen=True)
class PDParameters:
    marker: str
    mode: str
    emax: float
    ec50_ng_ml: float
    kout_h: float
    escape_max: float = 0.0
    escape_k_h: float = 0.0
    baseline: float = 100.0
    baseline_unit: str = "percent"


PK_PARAMETER_NAMES = [
    "ka_h",
    "vc_l",
    "q_l_h",
    "vp_l",
    "cl_day1_l_h",
    "cl_day15_l_h",
    "kaut_h",
    "km_mg_l",
]


def dosing_times_5_on_2_off(end_h: float, interval_h: float = 24.0) -> list[float]:
    """Return dose times for a 500 mg QD, 5-days-on/2-days-off schedule."""

    times: list[float] = []
    day = 1
    while (day - 1) * interval_h <= end_h + 1e-9:
        day_in_week = (day - 1) % 7
        if day_in_week < 5:
            times.append((day - 1) * interval_h)
        day += 1
    return times


def clearance_at(time_h: float | np.ndarray, params: PKParameters) -> float | np.ndarray:
    """Autoinduced low-concentration clearance.

    CL(t) = CLday1 + (CLday15 - CLday1) * (1 - exp(-kaut * t))
    """

    return params.cl_day1_l_h + (params.cl_day15_l_h - params.cl_day1_l_h) * (
        1.0 - np.exp(-params.kaut_h * np.asarray(time_h))
    )


def concentration_ng_ml(central_amount_mg: np.ndarray, vc_l: float) -> np.ndarray:
    return central_amount_mg * 1000.0 / vc_l


def _pk_rhs(time_h: float, state: np.ndarray, params: PKParameters) -> list[float]:
    gut_mg, central_mg, peripheral_mg = state
    central_conc_mg_l = max(central_mg / params.vc_l, 0.0)
    cl_t = float(clearance_at(time_h, params))
    vmax_mg_h = cl_t * params.km_mg_l
    elimination_mg_h = (
        vmax_mg_h * central_conc_mg_l / (params.km_mg_l + central_conc_mg_l)
        if central_conc_mg_l > 0.0
        else 0.0
    )

    absorption_mg_h = params.bioavailability * params.ka_h * gut_mg
    distribution_to_peripheral = (params.q_l_h / params.vc_l) * central_mg
    distribution_to_central = (params.q_l_h / params.vp_l) * peripheral_mg

    return [
        -params.ka_h * gut_mg,
        absorption_mg_h
        - distribution_to_peripheral
        + distribution_to_central
        - elimination_mg_h,
        distribution_to_peripheral - distribution_to_central,
    ]


def simulate_pk(
    params: PKParameters,
    end_h: float,
    dose_mg: float = 500.0,
    dose_times_h: Iterable[float] | None = None,
    dt_h: float = 0.1,
    initial_state: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Simulate the oral two-compartment PK model with bolus gut dosing."""

    dose_times = (
        sorted(float(t) for t in dose_times_h)
        if dose_times_h is not None
        else dosing_times_5_on_2_off(end_h)
    )
    state = np.array(list(initial_state) if initial_state is not None else [0.0, 0.0, 0.0])
    current_t = 0.0
    out_t: list[float] = []
    out_y: list[np.ndarray] = []

    breakpoints = sorted(set(t for t in dose_times if 0.0 <= t <= end_h) | {end_h})
    for stop_t in breakpoints:
        if stop_t > current_t:
            n_eval = max(2, int(np.ceil((stop_t - current_t) / dt_h)) + 1)
            t_eval = np.linspace(current_t, stop_t, n_eval)
            sol = solve_ivp(
                lambda t, y: _pk_rhs(t, y, params),
                (current_t, stop_t),
                state,
                t_eval=t_eval,
                method="LSODA",
                rtol=1e-6,
                atol=1e-8,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            times = sol.t
            values = sol.y.T
            if out_t and len(times) and np.isclose(times[0], out_t[-1]):
                times = times[1:]
                values = values[1:]
            out_t.extend(times.tolist())
            out_y.extend(values)
            state = sol.y[:, -1]
            current_t = stop_t

        if any(np.isclose(stop_t, dose_t) for dose_t in dose_times):
            state = state.copy()
            state[0] += dose_mg

    values = np.asarray(out_y)
    result = pd.DataFrame(
        {
            "time_h": np.asarray(out_t),
            "gut_mg": values[:, 0],
            "central_mg": values[:, 1],
            "peripheral_mg": values[:, 2],
        }
    )
    result["concentration_ng_ml"] = concentration_ng_ml(result["central_mg"].to_numpy(), params.vc_l)
    result["clearance_l_h"] = clearance_at(result["time_h"].to_numpy(), params)
    return result


def terminal_half_life_h(time_after_dose_h: np.ndarray, conc_ng_ml: np.ndarray) -> float:
    """Estimate apparent terminal half-life from 10-24 h after an occasion dose."""

    mask = (time_after_dose_h >= 10.0) & (time_after_dose_h <= 24.0) & (conc_ng_ml > 1e-9)
    if int(mask.sum()) < 3:
        return np.nan
    slope, _ = np.polyfit(time_after_dose_h[mask], np.log(conc_ng_ml[mask]), 1)
    return float(np.log(2.0) / (-slope)) if slope < 0 else np.nan


def occasion_metrics(profile: pd.DataFrame, start_h: float, label: str) -> dict[str, float | str]:
    window = profile[(profile["time_h"] >= start_h) & (profile["time_h"] <= start_h + 24.0)].copy()
    window["time_after_dose_h"] = window["time_h"] - start_h
    conc = window["concentration_ng_ml"].to_numpy()
    time_after = window["time_after_dose_h"].to_numpy()
    cmax_idx = int(np.argmax(conc))
    return {
        "occasion": label,
        "Cmax": float(conc[cmax_idx]),
        "Tmax": float(time_after[cmax_idx]),
        "AUC": float(np.trapezoid(conc, time_after)),
        "Half_life": terminal_half_life_h(time_after, conc),
    }


def summarize_pk_metrics(params: PKParameters, dt_h: float = 0.1) -> pd.DataFrame:
    profile = simulate_pk(params, end_h=360.0, dt_h=dt_h)
    rows = [
        occasion_metrics(profile, 0.0, "day1"),
        occasion_metrics(profile, 14.0 * 24.0, "day15"),
    ]
    return pd.DataFrame(rows)


def interpolate_concentration(profile: pd.DataFrame):
    time = profile["time_h"].to_numpy()
    conc = profile["concentration_ng_ml"].to_numpy()
    return lambda x: np.interp(x, time, conc, left=0.0, right=float(conc[-1]))


def simulate_pd(
    pk_profile: pd.DataFrame,
    params: PDParameters,
    end_h: float,
    dt_h: float = 0.5,
) -> pd.DataFrame:
    """Simulate normalized biomarker turnover driven by PK concentration.

    Inhibition markers use target input 1 - Emax*C/(EC50+C) plus optional
    escape/adaptation. Stimulation markers use 1 + Emax*C/(EC50+C).
    """

    conc_at = interpolate_concentration(pk_profile)
    sign = -1.0 if params.mode == "inhibition" else 1.0

    def rhs(time_h: float, y: np.ndarray) -> list[float]:
        conc = max(float(conc_at(time_h)), 0.0)
        drug_effect = params.emax * conc / (params.ec50_ng_ml + conc)
        escape = params.escape_max * (1.0 - np.exp(-params.escape_k_h * time_h))
        target = 1.0 + sign * drug_effect + escape
        return [params.kout_h * (target - y[0])]

    t_eval = np.arange(0.0, end_h + dt_h, dt_h)
    sol = solve_ivp(rhs, (0.0, end_h), [1.0], t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)
    normalized = sol.y[0]
    return pd.DataFrame(
        {
            "marker": params.marker,
            "time_h": sol.t,
            "day": sol.t / 24.0 + 1.0,
            "normalized_response": normalized,
            "percent_change_from_baseline": (normalized - 1.0) * 100.0,
            "absolute_response": normalized * params.baseline,
            "baseline_unit": params.baseline_unit,
        }
    )


def with_random_effects(params: PKParameters, eta: dict[str, float]) -> PKParameters:
    updates = {name: getattr(params, name) * float(np.exp(eta[name])) for name in PK_PARAMETER_NAMES}
    if updates["cl_day15_l_h"] <= updates["cl_day1_l_h"]:
        updates["cl_day15_l_h"] = updates["cl_day1_l_h"] * 1.05
    return replace(params, **updates)


def with_covariates(params: PKParameters, body_weight_kg: float) -> PKParameters:
    """Apply prespecified covariate scaling used for population simulation."""

    weight_ratio = body_weight_kg / 70.0
    return replace(
        params,
        vc_l=params.vc_l * weight_ratio,
        vp_l=params.vp_l * weight_ratio,
    )


def simulate_sparse_population(
    params: PKParameters,
    n_patients: int = 65,
    iiv_cv: dict[str, float] | None = None,
    residual_error: ResidualError | None = None,
    seed: int = 258,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a sparse two-sample-per-patient population dataset.

    The real raw PK observations are not public in the prompt, so this creates
    reproducible pseudo-observations at the reported median Tmax on day 1 and
    day 15 for diagnostics and VPC scaffolding.
    """

    rng = np.random.default_rng(seed)
    residual_error = residual_error or ResidualError()
    iiv_cv = iiv_cv or {name: 0.30 for name in PK_PARAMETER_NAMES}
    omega = {name: np.sqrt(np.log(cv**2 + 1.0)) for name, cv in iiv_cv.items()}

    rows: list[dict[str, float | int | str]] = []
    eta_rows: list[dict[str, float | int | str]] = []
    for patient_id in range(1, n_patients + 1):
        prior_vegfr = patient_id <= 49
        prior_count_group = ">=2" if rng.random() < 0.821 else "1"
        body_weight_kg = float(np.clip(rng.normal(75.0, 14.0), 45.0, 125.0))
        creatinine_clearance_ml_min = float(np.clip(rng.normal(75.0, 20.0), 30.0, 140.0))
        eta = {name: float(rng.normal(0.0, omega[name])) for name in PK_PARAMETER_NAMES}
        individual_params = with_covariates(with_random_effects(params, eta), body_weight_kg)
        profile = simulate_pk(individual_params, end_h=360.0, dt_h=0.2)

        for occasion, day, start_h in [("day1", 1, 0.0), ("day15", 15, 14.0 * 24.0)]:
            sample_t = start_h + 6.0
            ipred = float(np.interp(sample_t, profile["time_h"], profile["concentration_ng_ml"]))
            sd = np.sqrt((residual_error.proportional * ipred) ** 2 + residual_error.additive_ng_ml**2)
            observed = max(float(rng.normal(ipred, sd)), 0.0)
            rows.append(
                {
                    "patient_id": patient_id,
                    "occasion": occasion,
                    "day": day,
                    "absolute_time_h": sample_t,
                    "time_after_dose_h": 6.0,
                    "ipred_concentration_ng_ml": ipred,
                    "observed_concentration_ng_ml": observed,
                    "cwres": (observed - ipred) / sd if sd > 0 else np.nan,
                    "prior_vegfr_inhibitor": prior_vegfr,
                    "prior_count_group": prior_count_group,
                    "body_weight_kg": body_weight_kg,
                    "creatinine_clearance_ml_min": creatinine_clearance_ml_min,
                }
            )
        for name, value in eta.items():
            eta_rows.append({"patient_id": patient_id, "parameter": name, "eta": value})

    return pd.DataFrame(rows), pd.DataFrame(eta_rows)


def population_prediction_grid(
    params: PKParameters,
    n_patients: int = 300,
    iiv_cv: dict[str, float] | None = None,
    seed: int = 90210,
    dt_h: float = 0.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    iiv_cv = iiv_cv or {name: 0.30 for name in PK_PARAMETER_NAMES}
    omega = {name: np.sqrt(np.log(cv**2 + 1.0)) for name, cv in iiv_cv.items()}
    rows = []
    for patient_id in range(1, n_patients + 1):
        eta = {name: float(rng.normal(0.0, omega[name])) for name in PK_PARAMETER_NAMES}
        body_weight_kg = float(np.clip(rng.normal(75.0, 14.0), 45.0, 125.0))
        individual_params = with_covariates(with_random_effects(params, eta), body_weight_kg)
        profile = simulate_pk(individual_params, end_h=360.0, dt_h=dt_h)
        for occasion, start_h in [("day1", 0.0), ("day15", 14.0 * 24.0)]:
            window = profile[(profile["time_h"] >= start_h) & (profile["time_h"] <= start_h + 24.0)].copy()
            window["time_after_dose_h"] = window["time_h"] - start_h
            rows.append(
                window[["time_after_dose_h", "concentration_ng_ml"]].assign(
                    patient_id=patient_id,
                    occasion=occasion,
                )
            )
    return pd.concat(rows, ignore_index=True)
