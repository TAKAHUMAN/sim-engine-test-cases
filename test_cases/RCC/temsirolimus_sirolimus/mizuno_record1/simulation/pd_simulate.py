"""Adult RCC PD cohort simulation linked to the Mizuno PK model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from model.biomarker_model import BiomarkerModel
from model.pk_interface import ACCUMULATION_FACTOR, ADULT_FM, ADULT_RCC_DOSE_MG, build_adult_pk_dataframe
from model.pk_interface import get_typical_pk_profile
from model.tumor_growth_model import DEFAULT_RESPONDER_FRACTION, MixtureResponderModel, TumorGrowthModel

DEFAULT_PD_SEED = 20240501
DEFAULT_N_INDIVIDUALS = 500
PFS_T_MAX_DAYS = 365.0

AGE_MEAN = 62.0
AGE_SD = 10.0
AGE_MIN = 50.0
AGE_MAX = 80.0
BW_MEAN = 75.0
BW_SD = 15.0
BW_MIN = 50.0
BW_MAX = 110.0

BASELINE_TUMOR_MEDIAN_MM = 70.0
BASELINE_TUMOR_SIGMA = 0.4
LAMBDA_GROWTH_MEDIAN = 0.0077
LAMBDA_GROWTH_SIGMA = 0.5
LAMBDA_KILL_LITERATURE_MEDIAN = 0.015
LAMBDA_KILL_MEDIAN = 0.0077
LAMBDA_KILL_SIGMA = 0.6
PS6K1_BASELINE_MEDIAN = 100.0
PS6K1_BASELINE_SIGMA = 0.3


@dataclass(frozen=True)
class PDResults:
    """PD patient-level results and linked PK profiles."""

    patients: pd.DataFrame
    pk_profiles: pd.DataFrame


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    low: float,
    high: float,
) -> float:
    a = (low - mean) / sd
    b = (high - mean) / sd
    return float(stats.truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng))


def sample_adult_rcc_demographics(rng: np.random.Generator) -> dict[str, float | str]:
    """Sample adult RCC demographics and prognostic group."""

    return {
        "age": _truncated_normal(rng, AGE_MEAN, AGE_SD, AGE_MIN, AGE_MAX),
        "BW": _truncated_normal(rng, BW_MEAN, BW_SD, BW_MIN, BW_MAX),
        "sex": "M" if rng.random() < 0.70 else "F",
        "mskcc_risk": str(
            rng.choice(["favorable", "intermediate", "poor"], p=[0.30, 0.50, 0.20])
        ),
        "prior_therapy": "VEGF TKI failure",
    }


def simulate_pd_cohort(
    pk_df: pd.DataFrame | None = None,
    N_individuals: int = DEFAULT_N_INDIVIDUALS,
    seed: int = DEFAULT_PD_SEED,
    *,
    dose_mg: float = ADULT_RCC_DOSE_MG,
    adult_fm: float = ADULT_FM,
    accumulation_factor: float = ACCUMULATION_FACTOR,
    resistance_tau_days: float | None = None,
    lambda_kill_median: float = LAMBDA_KILL_MEDIAN,
    use_mixture_model: bool = False,
    responder_fraction: float = DEFAULT_RESPONDER_FRACTION,
) -> PDResults:
    """Simulate adult RCC patients through PK, biomarker inhibition, and PFS."""

    _ = pk_df
    rng = np.random.default_rng(seed=seed)
    biomarker = BiomarkerModel(EC50=0.010, Emax=0.95, gamma=1.2)
    mixture = MixtureResponderModel(responder_fraction=responder_fraction)
    patient_records: list[dict[str, float | int | str | bool]] = []
    pk_records: list[dict[str, float | int]] = []

    for i in range(1, N_individuals + 1):
        demographics = sample_adult_rcc_demographics(rng)
        bw = float(demographics["BW"])
        baseline_tumor_mm = float(
            rng.lognormal(mean=np.log(BASELINE_TUMOR_MEDIAN_MM), sigma=BASELINE_TUMOR_SIGMA)
        )
        L_0 = float(np.log(baseline_tumor_mm))
        lambda_growth = float(
            rng.lognormal(mean=np.log(LAMBDA_GROWTH_MEDIAN), sigma=LAMBDA_GROWTH_SIGMA)
        )
        lambda_kill = float(
            rng.lognormal(mean=np.log(lambda_kill_median), sigma=LAMBDA_KILL_SIGMA)
        )
        pS6K1_0 = float(
            rng.lognormal(mean=np.log(PS6K1_BASELINE_MEDIAN), sigma=PS6K1_BASELINE_SIGMA)
        )

        times_h, tem_conc, sir_conc = get_typical_pk_profile(
            bw,
            dose_mg=dose_mg,
            fm=adult_fm,
            accumulation_factor=accumulation_factor,
        )
        pS6K1_array, e_array = biomarker.evaluate(times_h, tem_conc, sir_conc, pS6K1_0)
        times_day = times_h / 24.0
        if use_mixture_model:
            pfs_days, progressed, is_responder = mixture.simulate_pfs_mixture(
                L_0=L_0,
                lambda_growth=lambda_growth,
                lambda_kill_0=lambda_kill,
                biomarker_model=biomarker,
                times=times_day,
                pS6K1_array=pS6K1_array,
                pS6K1_0=pS6K1_0,
                rng=rng,
                tau_resist_days=resistance_tau_days,
                t_max=PFS_T_MAX_DAYS,
            )
        else:
            tumor = TumorGrowthModel(
                lambda_growth,
                lambda_kill,
                biomarker,
                times_day,
                pS6K1_array,
                pS6K1_0,
                resistance_tau_days=resistance_tau_days,
            )
            pfs_days, progressed = tumor.simulate_pfs(L_0, t_max=PFS_T_MAX_DAYS)
            is_responder = True

        for time_h, tem_c, sir_c in zip(times_h, tem_conc, sir_conc, strict=True):
            pk_records.append(
                {
                    "id": i,
                    "time_h": float(time_h),
                    "conc_TEM_pred": float(tem_c),
                    "conc_SIR_pred": float(sir_c),
                }
            )

        patient_records.append(
            {
                "id": i,
                **demographics,
                "dose_mg": dose_mg,
                "adult_fm": adult_fm,
                "accumulation_factor": accumulation_factor,
                "baseline_tumor_mm": baseline_tumor_mm,
                "L_0": L_0,
                "lambda_growth": lambda_growth,
                "lambda_kill": lambda_kill,
                "lambda_kill_median": lambda_kill_median,
                "resistance_tau_days": resistance_tau_days,
                "use_mixture_model": use_mixture_model,
                "responder_fraction": responder_fraction if use_mixture_model else None,
                "is_responder": is_responder,
                "pS6K1_0": pS6K1_0,
                "median_s6k1_inhibition": float(np.median(e_array)),
                "mean_s6k1_inhibition": float(np.mean(e_array)),
                "trough_s6k1_inhibition": float(e_array[-1]),
                "PFS_days": pfs_days,
                "PFS_months": pfs_days / 30.44,
                "event_observed": progressed,
            }
        )

    patients = pd.DataFrame.from_records(patient_records)
    if pk_records:
        pk_profiles = pd.DataFrame.from_records(pk_records)
    else:
        pk_profiles = build_adult_pk_dataframe(patients)
    return PDResults(patients=patients, pk_profiles=pk_profiles)


def run_pd_pipeline(
    *,
    N_individuals: int = DEFAULT_N_INDIVIDUALS,
    seed: int = DEFAULT_PD_SEED,
    output_dir: str | Path = "outputs",
    resistance_tau_days: float | None = None,
    lambda_kill_median: float = LAMBDA_KILL_MEDIAN,
    use_mixture_model: bool = False,
    responder_fraction: float = DEFAULT_RESPONDER_FRACTION,
) -> PDResults:
    """Run and save the adult RCC PD cohort."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = simulate_pd_cohort(
        N_individuals=N_individuals,
        seed=seed,
        resistance_tau_days=resistance_tau_days,
        lambda_kill_median=lambda_kill_median,
        use_mixture_model=use_mixture_model,
        responder_fraction=responder_fraction,
    )
    results.patients.to_csv(output / "pd_pfs_results.csv", index=False)
    results.pk_profiles.to_csv(output / "pd_pk_profiles.csv", index=False)
    return results
