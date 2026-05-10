"""Interface between the audited Mizuno PK model and adult RCC PD simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from model.pk_model import DEFAULT_INFUSION_DURATION_H, TYPICAL_70KG, allometric_parameters
from model.pk_model import with_replaced_parameters
from simulation.pk_simulate import predicted_concentrations, solve_amounts

ADULT_RCC_DOSE_MG = 25.0
ADULT_FM = 0.70
ACCUMULATION_FACTOR = 1.5
WEEK_H = 168.0


@dataclass(frozen=True)
class PKProfile:
    """One adult RCC concentration-time profile."""

    time_h: NDArray[np.float64]
    conc_TEM_ng_ml: NDArray[np.float64]
    conc_SIR_ng_ml: NDArray[np.float64]


def load_pk_simulation(filepath: str | Path) -> pd.DataFrame:
    """Load precomputed Mizuno PK output."""

    return pd.read_csv(filepath)


def get_concentration_profile(
    pk_df: pd.DataFrame, individual_id: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract time, temsirolimus concentration, and sirolimus concentration arrays."""

    subset = pk_df[pk_df["id"] == individual_id].sort_values("time_h")
    if subset.empty:
        raise ValueError(f"No PK profile found for individual id {individual_id}")
    return (
        subset["time_h"].to_numpy(dtype=float),
        subset["conc_TEM_pred"].to_numpy(dtype=float),
        subset["conc_SIR_pred"].to_numpy(dtype=float),
    )


def interpolate_concentration(
    times: Iterable[float],
    concs: Iterable[float],
    t_eval: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Linearly interpolate concentration at arbitrary time, with zero beyond range."""

    f = interp1d(
        np.asarray(tuple(times), dtype=float),
        np.asarray(tuple(concs), dtype=float),
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return f(t_eval)


def adult_rcc_parameters(bw_kg: float, *, fm: float = ADULT_FM):
    """Return allometrically scaled adult parameters with adult temsirolimus-to-sirolimus Fm."""

    pediatric = allometric_parameters(bw_kg, typical_70kg=TYPICAL_70KG)
    return with_replaced_parameters(pediatric, Fm=fm)


def get_typical_pk_profile(
    bw_kg: float,
    *,
    dose_mg: float = ADULT_RCC_DOSE_MG,
    fm: float = ADULT_FM,
    accumulation_factor: float = ACCUMULATION_FACTOR,
    infusion_duration_h: float = DEFAULT_INFUSION_DURATION_H,
    times_h: Iterable[float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a 25 mg weekly adult RCC PK profile from the audited ODE model.

    The profile is a week-1, day-1 curve scaled by the specified accumulation
    factor as requested for the steady-state PD approximation.
    """

    if times_h is None:
        times = np.linspace(0.0, WEEK_H, 169)
    else:
        times = np.asarray(tuple(times_h), dtype=float)
    parameters = adult_rcc_parameters(bw_kg, fm=fm)
    amounts = solve_amounts(
        parameters,
        dose_mg,
        times,
        infusion_duration_h=infusion_duration_h,
    )
    tem, sir = predicted_concentrations(amounts, parameters)
    return times, tem * accumulation_factor, sir * accumulation_factor


def build_adult_pk_dataframe(
    adult_df: pd.DataFrame,
    *,
    dose_mg: float = ADULT_RCC_DOSE_MG,
    fm: float = ADULT_FM,
    accumulation_factor: float = ACCUMULATION_FACTOR,
) -> pd.DataFrame:
    """Build a reusable adult PK profile table for simulated RCC patients."""

    records: list[dict[str, float | int]] = []
    for row in adult_df.itertuples(index=False):
        times, tem, sir = get_typical_pk_profile(
            float(row.BW),
            dose_mg=dose_mg,
            fm=fm,
            accumulation_factor=accumulation_factor,
        )
        for time_h, tem_c, sir_c in zip(times, tem, sir, strict=True):
            records.append(
                {
                    "id": int(row.id),
                    "time_h": float(time_h),
                    "conc_TEM_pred": float(tem_c),
                    "conc_SIR_pred": float(sir_c),
                }
            )
    return pd.DataFrame.from_records(records)
