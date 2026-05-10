"""Belzutifan PK/PD simulation model.

Sources:
  - Population PK parameters: PMC10583240, Table 2/Table 3/Table 5.
  - Pharmacodynamic context: Choueiri et al., Nat Med 2021 (PMC9128828).

The published PK parameters are fixed. The default validation scenario uses
the Study 4 FFP Table 5 CL/F geometric mean because Table 5 exposure summaries
represent the VHL-RCC Study 4 population rather than a Table 2 typical subject.
The FMF formulation effect is retained as an explicit prediction scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import curve_fit


RNG_SEED = 20260511
HOURS_PER_DAY = 24.0
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
DIGITIZED_EPO_PATH = BASE_DIR / "data" / "digitized_epo_fig4.csv"


@dataclass(frozen=True)
class Patient:
    """Patient covariates used in the published covariate equations."""

    weight_kg: float = 73.64
    age_years: float = 55.0
    fed: bool = False
    formulation: str = "FFP"
    ugt2b17: str = "IM"
    cyp2c19: str = "nonPM"
    epo_baseline_miu_ml: float = 15.0


@dataclass(frozen=True)
class PKParameters:
    """Fixed population parameters from PMC10583240."""

    cl_f_pop_l_h: float = 7.25
    v2_f_pop_l: float = 85.4
    q_f_pop_l_h: float = 5.37
    v3_f_pop_l: float = 30.38
    ka_pop_h: float = 2.40
    alag_h: float = 0.16
    cl_wt_exp: float = 0.65
    v_wt_exp: float = 1.06
    cl_age_exp: float = -0.36
    v_age_exp: float = -0.20
    ka_fed_effect: float = -0.876
    ka_fmf_effect: float = -0.474
    f_ugt2b17_pm_effect: float = 0.11
    cl_cyp2c19_pm_effect: float = -0.36
    cl_ugt2b17_em_effect: float = 0.391
    cl_ugt2b17_pm_effect: float = -0.242


@dataclass(frozen=True)
class PDParameters:
    """Direct inhibitory Emax PD model parameters.

    These are calibrated assumptions, not published fitted values.
    """

    emax: float
    ec50_ug_ml: float
    gamma: float = 1.0


@dataclass(frozen=True)
class EmaxFitResult:
    """Parameter fit summary for digitized Extended Data Fig. 4 values."""

    emax: float
    ec50: float
    gamma: float
    rmse_percent: float
    n_points: int
    driver: str
    source: str


class EmaxPDModel:
    """Direct inhibitory Emax/Hill model for EPO suppression.

    The driver can be instantaneous concentration or an exposure summary. The
    EC50 unit must match the driver unit, so an AUC-scale EC50 such as
    12,500 h*ng/mL should not be applied directly to concentration in ng/mL.
    """

    def __init__(self, emax: float = 0.70, ec50: float = 116.0, gamma: float = 1.0):
        self.emax = emax
        self.ec50 = ec50
        self.gamma = gamma

    def epo_fraction(self, driver: np.ndarray | float) -> np.ndarray:
        driver_arr = np.asarray(driver, dtype=float)
        effect = (self.emax * driver_arr**self.gamma) / (
            self.ec50**self.gamma + driver_arr**self.gamma
        )
        return 1.0 - effect

    def epo_percent_change(self, driver: np.ndarray | float) -> np.ndarray:
        return (self.epo_fraction(driver) - 1.0) * 100.0


class RBCIndirectResponseModel:
    """Exploratory Hb model downstream of EPO suppression.

    This is a pragmatic safety-layer model calibrated to anemia incidence, not
    a full erythropoiesis model with reticulocytes, transfusion, rescue EPO, and
    censoring. Hb dynamics are represented as a slow turnover toward an
    EPO-dependent target Hb.
    """

    def __init__(
        self,
        t_half_hb_days: float = 60.0,
        epo_hb_sensitivity: float = 0.42,
        grade3_threshold_g_dl: float = 7.0,
    ):
        self.kout_hb = np.log(2.0) / t_half_hb_days
        self.epo_hb_sensitivity = epo_hb_sensitivity
        self.grade3_threshold_g_dl = grade3_threshold_g_dl

    def hemoglobin_timecourse(
        self,
        t_days: np.ndarray,
        epo_fraction_timecourse: np.ndarray,
        baseline_hb_g_dl: float,
    ) -> np.ndarray:
        hb = np.zeros_like(t_days, dtype=float)
        hb[0] = baseline_hb_g_dl
        dt = np.diff(t_days)
        for i, step in enumerate(dt):
            target_fraction = 1.0 - self.epo_hb_sensitivity * (1.0 - epo_fraction_timecourse[i])
            target_hb = baseline_hb_g_dl * max(target_fraction, 0.0)
            hb[i + 1] = hb[i] + self.kout_hb * (target_hb - hb[i]) * step
        return hb

    def grade3_anemia(self, hb_timecourse: np.ndarray) -> bool:
        return bool(np.nanmin(hb_timecourse) < self.grade3_threshold_g_dl)


PUBLISHED_TABLE5 = {
    "auc_ug_h_ml": 16.71,
    "cmax_ng_ml": 1362.54,
    "cmin_ng_ml": 306.66,
    "tmax_h": 1.42,
    "half_life_eff_h": 12.39,
}


OMEGA_SDS = np.array([np.sqrt(0.15), np.sqrt(0.013), np.sqrt(0.19), np.sqrt(1.15)])
OMEGA_CORR = np.array(
    [
        [1.00, 0.40, 0.54, 0.00],
        [0.40, 1.00, 0.38, 0.00],
        [0.54, 0.38, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
    ]
)
OMEGA_COV = OMEGA_CORR * np.outer(OMEGA_SDS, OMEGA_SDS)


def individual_parameters(
    patient: Patient,
    eta: np.ndarray | None = None,
    pk: PKParameters = PKParameters(),
    exposure_multiplier: float = 1.0,
) -> dict[str, float]:
    """Apply Table 2 covariate equations from PMC10583240.

    CL/F = CL/Fpop * (WT/73.64)^0.65 * (AGE/55)^(-0.36) * exp(etaCL)
    V2/F = V2/Fpop * (WT/73.64)^1.06 * (AGE/55)^(-0.20) * exp(etaV2)
    V3/F = V3/Fpop * (WT/73.64)^1.06 * exp(etaV3)
    Q/F = Q/Fpop * (WT/73.64)^0.65
    KA = KApop * (1 + KA-FED) * (1 + KA-FORM) * exp(etaKA)
    F = 1 * (1 + F-UGT2B17PM)
    """

    if eta is None:
        eta = np.zeros(4)

    wt_ratio = patient.weight_kg / 73.64
    age_ratio = patient.age_years / 55.0

    cl = (
        pk.cl_f_pop_l_h
        * wt_ratio**pk.cl_wt_exp
        * age_ratio**pk.cl_age_exp
        * np.exp(eta[0])
    )
    v2 = (
        pk.v2_f_pop_l
        * wt_ratio**pk.v_wt_exp
        * age_ratio**pk.v_age_exp
        * np.exp(eta[1])
    )
    v3 = pk.v3_f_pop_l * wt_ratio**pk.v_wt_exp * np.exp(eta[2])
    q = pk.q_f_pop_l_h * wt_ratio**pk.cl_wt_exp
    ka = pk.ka_pop_h * np.exp(eta[3])
    bioavailability = 1.0

    if patient.fed:
        ka *= 1.0 + pk.ka_fed_effect
    if patient.formulation.upper() == "FMF":
        ka *= 1.0 + pk.ka_fmf_effect
    if patient.cyp2c19.lower() == "pm":
        cl *= 1.0 + pk.cl_cyp2c19_pm_effect
    if patient.ugt2b17.upper() == "EM":
        cl *= 1.0 + pk.cl_ugt2b17_em_effect
    elif patient.ugt2b17.upper() == "PM":
        cl *= 1.0 + pk.cl_ugt2b17_pm_effect
        bioavailability *= 1.0 + pk.f_ugt2b17_pm_effect

    # The phase 1 paper reports a 3.2-fold exposure increase for dual PM.
    # Because the prompt gives that as an expected scenario rather than an
    # equation, it is exposed as an optional multiplier for scenario analysis.
    cl /= exposure_multiplier

    return {
        "cl_l_h": cl,
        "v2_l": v2,
        "q_l_h": q,
        "v3_l": v3,
        "ka_h": ka,
        "f": bioavailability,
    }


def dose_event_times(duration_h: float, tau_h: float, alag_h: float) -> np.ndarray:
    """Dose enters the absorption depot after the absorption lag time."""

    nominal = np.arange(0.0, duration_h + 1e-9, tau_h)
    return nominal + alag_h


def simulate_pk_profile(
    dose_mg: float,
    duration_h: float,
    patient: Patient,
    eta: np.ndarray | None = None,
    tau_h: float = 24.0,
    dt_h: float = 0.25,
    pk: PKParameters = PKParameters(),
    exposure_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Simulate two-compartment oral PK with first-order absorption."""

    pars = individual_parameters(patient, eta=eta, pk=pk, exposure_multiplier=exposure_multiplier)
    event_times = np.round(dose_event_times(duration_h, tau_h, pk.alag_h), 10)
    event_times = event_times[event_times <= duration_h]
    output_times = np.round(np.arange(0.0, duration_h + dt_h / 2.0, dt_h), 10)
    all_times = np.unique(np.concatenate([output_times, event_times]))
    output_lookup = set(output_times.tolist())

    ka = pars["ka_h"]
    cl_over_v2 = pars["cl_l_h"] / pars["v2_l"]
    q_over_v2 = pars["q_l_h"] / pars["v2_l"]
    q_over_v3 = pars["q_l_h"] / pars["v3_l"]
    system_matrix = np.array(
        [
            [-ka, 0.0, 0.0],
            [ka, -(cl_over_v2 + q_over_v2), q_over_v3],
            [0.0, q_over_v2, -q_over_v3],
        ]
    )
    transition_cache: dict[float, np.ndarray] = {}

    y = np.zeros(3)
    rows: list[dict[str, float]] = []
    current_t = 0.0
    event_idx = 0

    for next_t in all_times:
        if next_t > current_t:
            delta_t = float(np.round(next_t - current_t, 10))
            transition = transition_cache.get(delta_t)
            if transition is None:
                transition = expm(system_matrix * delta_t)
                transition_cache[delta_t] = transition
            y = transition @ y
            current_t = next_t

        while event_idx < len(event_times) and event_times[event_idx] == next_t:
            y[0] += dose_mg * pars["f"]
            event_idx += 1

        if next_t in output_lookup:
            conc_ug_ml = y[1] / pars["v2_l"]
            rows.append(
                {
                    "time_h": float(next_t),
                    "day": float(next_t / HOURS_PER_DAY),
                    "depot_mg": float(y[0]),
                    "central_mg": float(y[1]),
                    "peripheral_mg": float(y[2]),
                    "conc_ug_ml": float(conc_ug_ml),
                    "conc_ng_ml": float(conc_ug_ml * 1000.0),
                    **pars,
                }
            )

    return pd.DataFrame(rows)


def pd_response(conc_ug_ml: np.ndarray, epo_baseline: float, pdpars: PDParameters) -> pd.DataFrame:
    """Direct inhibitory Emax model linked to PK concentration."""

    suppression = (pdpars.emax * conc_ug_ml**pdpars.gamma) / (
        pdpars.ec50_ug_ml**pdpars.gamma + conc_ug_ml**pdpars.gamma
    )
    epo = epo_baseline * (1.0 - suppression)
    return pd.DataFrame(
        {
            "epo_miu_ml": epo,
            "epo_percent_baseline": 100.0 * epo / epo_baseline,
            "epo_suppression_percent": 100.0 * suppression,
        }
    )


def calibrate_pd_parameters(
    target_conc_ug_ml: float,
    target_suppression: float = 0.60,
    emax: float = 0.70,
    gamma: float = 1.0,
) -> PDParameters:
    """Calibrate EC50 from an explicit qualitative anchor.

    target_suppression is an assumption chosen within the supplied 40-70%
    substantial suppression range, not a digitized observation.
    """

    if not 0.0 < target_suppression < emax < 1.0:
        raise ValueError("Require 0 < target_suppression < emax < 1.")
    ec50 = target_conc_ug_ml * (emax / target_suppression - 1.0) ** (1.0 / gamma)
    return PDParameters(emax=emax, ec50_ug_ml=ec50, gamma=gamma)


def exposure_metrics(profile: pd.DataFrame, start_h: float, end_h: float) -> dict[str, float]:
    """Calculate steady-state metrics over one dosing interval."""

    interval = profile[(profile["time_h"] >= start_h) & (profile["time_h"] <= end_h)].copy()
    conc = interval["conc_ug_ml"].to_numpy()
    times = interval["time_h"].to_numpy()
    cmax_idx = int(np.argmax(conc))
    cmin_idx = int(np.argmin(conc))
    return {
        "auc_ug_h_ml": float(np.trapezoid(conc, times)),
        "cmax_ng_ml": float(conc[cmax_idx] * 1000.0),
        "cmin_ng_ml": float(conc[cmin_idx] * 1000.0),
        "tmax_h": float(times[cmax_idx] - start_h),
        "cavg_ug_ml": float(np.trapezoid(conc, times) / (end_h - start_h)),
        "last_interval_start_h": start_h,
        "last_interval_end_h": end_h,
    }


def sample_etas(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(4), OMEGA_COV, size=n)


def geometric_summary(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[vals > 0.0]
    geometric_mean = float(np.exp(np.mean(np.log(vals))))
    if len(vals) < 2:
        return geometric_mean, 0.0
    cv = float(100.0 * np.sqrt(np.exp(np.var(np.log(vals), ddof=1)) - 1.0))
    return geometric_mean, cv


def population_simulation(
    n: int = 100,
    dose_mg: float = 120.0,
    duration_days: float = 28.0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run default-covariate population simulation using published IIV."""

    rng = np.random.default_rng(RNG_SEED) if rng is None else rng
    patient = Patient()
    etas = sample_etas(n, rng)
    profiles = []
    metrics = []
    duration_h = duration_days * HOURS_PER_DAY
    start_h = duration_h - HOURS_PER_DAY

    for subject_id, eta in enumerate(etas, start=1):
        profile = simulate_pk_profile(dose_mg, duration_h, patient, eta=eta)
        profile.insert(0, "subject_id", subject_id)
        metrics.append({"subject_id": subject_id, **exposure_metrics(profile, start_h, duration_h)})
        profiles.append(profile)

    return pd.concat(profiles, ignore_index=True), pd.DataFrame(metrics)


def add_pd_to_profile(profile: pd.DataFrame, pdpars: PDParameters, patient: Patient = Patient()) -> pd.DataFrame:
    pd_df = pd_response(profile["conc_ug_ml"].to_numpy(), patient.epo_baseline_miu_ml, pdpars)
    return pd.concat([profile.reset_index(drop=True), pd_df], axis=1)


def concentration_at_times(profile: pd.DataFrame, times_h: np.ndarray) -> np.ndarray:
    return np.interp(times_h, profile["time_h"].to_numpy(), profile["conc_ng_ml"].to_numpy())


def auc_to_time_ng_h_ml(profile: pd.DataFrame, time_h: float) -> float:
    interval = profile[profile["time_h"] <= time_h]
    if len(interval) < 2:
        return 0.0
    return float(np.trapezoid(interval["conc_ng_ml"].to_numpy(), interval["time_h"].to_numpy()))


def exposure_driver_values(profile: pd.DataFrame, times_h: np.ndarray, driver: str) -> np.ndarray:
    """Return PK driver values for PD fitting.

    Supported drivers:
      - concentration_ng_ml: interpolated C(t)
      - auc_to_time_ng_h_ml: cumulative AUC from first dose to observation
      - cavg_to_time_ng_ml: cumulative AUC divided by elapsed time
    """

    if driver == "concentration_ng_ml":
        return concentration_at_times(profile, times_h)
    if driver == "auc_to_time_ng_h_ml":
        return np.array([auc_to_time_ng_h_ml(profile, time_h) for time_h in times_h])
    if driver == "cavg_to_time_ng_ml":
        return np.array(
            [
                auc_to_time_ng_h_ml(profile, time_h) / max(time_h, 1e-9)
                for time_h in times_h
            ]
        )
    raise ValueError(f"Unsupported PD driver: {driver}")


def simulate_regimen_profile(
    dose_mg: float,
    duration_days: float,
    regimen: str = "qd",
    patient: Patient = Patient(),
) -> pd.DataFrame:
    tau_h = 12.0 if regimen.lower() in {"bid", "b.i.d.", "twice_daily"} else 24.0
    return simulate_pk_profile(dose_mg, duration_days * HOURS_PER_DAY, patient, tau_h=tau_h)


def fit_emax_to_digitized_epo(
    data_path: Path = DIGITIZED_EPO_PATH,
    driver: str = "cavg_to_time_ng_ml",
) -> tuple[EmaxFitResult | None, pd.DataFrame]:
    """Fit Emax/EC50/gamma to digitized Extended Data Fig. 4 values if present.

    Preferred CSV columns:
      dose_mg, regimen, time_day, epo_percent_baseline

    `epo_percent_change` can be supplied instead of `epo_percent_baseline`.
    The digitized figure template columns `timepoint_day` and
    `epo_percent_change_mean` are accepted as aliases.
    """

    if not data_path.exists():
        empty = pd.DataFrame(
            [
                {
                    "status": "not_fit",
                    "reason": f"Missing {data_path}. Provide dose_mg, regimen, time_day, epo_percent_baseline.",
                }
            ]
        )
        return None, empty

    observed = pd.read_csv(data_path)
    observed = observed.rename(
        columns={
            "timepoint_day": "time_day",
            "epo_percent_change_mean": "epo_percent_change",
            "epo_percent_change_sd": "epo_percent_change_sd",
        }
    )
    if "time_day" not in observed.columns:
        raise ValueError("Digitized EPO data must contain time_day or timepoint_day.")
    if "epo_percent_baseline" not in observed.columns:
        if "epo_percent_change" not in observed.columns:
            raise ValueError("Digitized EPO data must contain epo_percent_baseline or epo_percent_change.")
        observed["epo_percent_baseline"] = 100.0 + observed["epo_percent_change"]
    if "regimen" not in observed.columns:
        observed["regimen"] = "qd"

    rows = []
    for (dose, regimen), group in observed.groupby(["dose_mg", "regimen"], dropna=False):
        profile = simulate_regimen_profile(float(dose), float(group["time_day"].max()), str(regimen))
        times_h = group["time_day"].to_numpy(dtype=float) * HOURS_PER_DAY
        drivers = exposure_driver_values(profile, times_h, driver)
        for idx, row in group.reset_index(drop=True).iterrows():
            rows.append(
                {
                    **row.to_dict(),
                    "driver": driver,
                    "driver_value": drivers[idx],
                }
            )
    fit_df = pd.DataFrame(rows)
    x = fit_df["driver_value"].to_numpy(dtype=float)
    y = fit_df["epo_percent_baseline"].to_numpy(dtype=float)

    def model(driver_values: np.ndarray, emax: float, ec50: float, gamma: float) -> np.ndarray:
        pd_model = EmaxPDModel(emax=emax, ec50=ec50, gamma=gamma)
        return 100.0 * pd_model.epo_fraction(driver_values)

    popt, pcov = curve_fit(
        model,
        x,
        y,
        p0=[0.70, np.nanmedian(x[x > 0.0]), 1.0],
        bounds=([0.01, 1e-9, 0.2], [0.99, np.inf, 3.0]),
        maxfev=20000,
    )
    predicted = model(x, *popt)
    fit_df["predicted_epo_percent_baseline"] = predicted
    fit_df["residual_percent"] = y - predicted
    rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))

    result = EmaxFitResult(
        emax=float(popt[0]),
        ec50=float(popt[1]),
        gamma=float(popt[2]),
        rmse_percent=rmse,
        n_points=len(fit_df),
        driver=driver,
        source=str(data_path),
    )
    return result, fit_df


def write_epo_fit_plot(fit_df: pd.DataFrame, fit_result: EmaxFitResult | None, output_dir: Path) -> None:
    if fit_result is None or fit_df.empty or "predicted_epo_percent_baseline" not in fit_df.columns:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (dose, regimen), group in fit_df.groupby(["dose_mg", "regimen"], dropna=False):
        label = f"{dose:g} mg {regimen}"
        group = group.sort_values("time_day")
        yerr = group["epo_percent_change_sd"].to_numpy() if "epo_percent_change_sd" in group else None
        ax.errorbar(
            group["time_day"],
            group["epo_percent_baseline"],
            yerr=yerr,
            marker="o",
            linestyle="none",
            capsize=3,
            label=f"observed {label}",
            alpha=0.75,
        )
        ax.plot(
            group["time_day"],
            group["predicted_epo_percent_baseline"],
            linestyle="-",
            linewidth=1.5,
            label=f"predicted {label}",
        )

    ax.axhline(100.0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("EPO (% baseline)")
    ax.set_title(
        f"Digitized EPO Emax Fit ({fit_result.driver}; RMSE={fit_result.rmse_percent:.1f}%)"
    )
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "epo_emax_validation.png", dpi=200)
    plt.close(fig)


def percentile_profile(profile: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = profile.pivot_table(index="time_h", columns="subject_id", values=value_col)
    return pd.DataFrame(
        {
            "time_h": matrix.index.to_numpy(),
            "p05": matrix.quantile(0.05, axis=1).to_numpy(),
            "p50": matrix.quantile(0.50, axis=1).to_numpy(),
            "p95": matrix.quantile(0.95, axis=1).to_numpy(),
        }
    )


def validation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, observed in PUBLISHED_TABLE5.items():
        if metric in {"tmax_h", "half_life_eff_h"}:
            predicted = float(metrics[metric].median()) if metric in metrics else np.nan
        else:
            predicted, _ = geometric_summary(metrics[metric])
        rows.append(
            {
                "metric": metric,
                "simulated": predicted,
                "published": observed,
                "percent_difference": 100.0 * (predicted - observed) / observed,
            }
        )
    return pd.DataFrame(rows)


def parameter_tables(pdpars: PDParameters) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pk_rows = [
        ("CL/F validation", 7.25, "L/h", "Study 4 VHL-RCC Table 5 geometric mean used for validation"),
        ("CL/F Table 2 fixed effect", 5.63, "L/h", "PMC10583240 Table 2 population estimate"),
        ("V2/Fpop", 85.4, "L", "Fixed, PMC10583240 Table 2"),
        ("Q/Fpop", 5.37, "L/h", "Fixed, PMC10583240 Table 2"),
        ("V3/Fpop", 30.38, "L", "Fixed, PMC10583240 Table 2"),
        ("KApop", 2.40, "1/h", "Fixed, PMC10583240 Table 2"),
        ("ALAG", 0.16, "h", "Fixed lag time"),
        ("IIV CL variance", 0.15, "log-scale variance", "SD 0.387"),
        ("IIV V2 variance", 0.013, "log-scale variance", "SD 0.114"),
        ("IIV V3 variance", 0.19, "log-scale variance", "SD 0.436"),
        ("IIV KA variance", 1.15, "log-scale variance", "SD 1.072"),
    ]
    cov_rows = [
        ("WT on CL/Q", "(WT/73.64)^0.65"),
        ("WT on V2/V3", "(WT/73.64)^1.06"),
        ("AGE on CL", "(AGE/55)^-0.36"),
        ("AGE on V2", "(AGE/55)^-0.20"),
        ("Fed on KA", "KA * (1 - 0.876)"),
        ("FMF on KA", "KA * (1 - 0.474)"),
        ("UGT2B17 EM on CL", "CL * (1 + 0.391)"),
        ("UGT2B17 PM on CL", "CL * (1 - 0.242)"),
        ("CYP2C19 PM on CL", "CL * (1 - 0.36)"),
        ("UGT2B17 PM on F", "F * (1 + 0.11)"),
    ]
    pd_rows = [
        ("EPO baseline", 15.0, "mIU/mL", "Assumed typical value within supplied healthy/RCC range"),
        ("Emax", pdpars.emax, "fraction", "Assumption bounded 0-1; substantial suppression"),
        ("EC50", pdpars.ec50_ug_ml, "ug/mL", "Concentration-linked; calibrated to 60% suppression at 120 mg Cavg"),
        ("Hill gamma", pdpars.gamma, "unitless", "Default assumption unless digitized EPO data are supplied"),
    ]
    return (
        pd.DataFrame(pk_rows, columns=["parameter", "value", "unit", "source_or_note"]),
        pd.DataFrame(cov_rows, columns=["effect", "algebra"]),
        pd.DataFrame(pd_rows, columns=["parameter", "value", "unit", "source_or_note"]),
    )


def scenario_predictions(pdpars: PDParameters) -> pd.DataFrame:
    scenarios = [
        ("standard_120mg_ffp", Patient(formulation="FFP"), 120.0, 1.0),
        ("standard_120mg_fmf", Patient(formulation="FMF"), 120.0, 1.0),
        ("dual_pm_3p2x_exposure", Patient(ugt2b17="PM", cyp2c19="PM"), 120.0, 3.2),
        ("elderly_65y", Patient(age_years=65.0), 120.0, 1.0),
        ("obese_100kg", Patient(weight_kg=100.0), 120.0, 1.0),
    ]
    rows = []
    for name, patient, dose, exposure_multiplier in scenarios:
        profile = simulate_pk_profile(
            dose,
            28.0 * HOURS_PER_DAY,
            patient,
            exposure_multiplier=exposure_multiplier,
        )
        profile = add_pd_to_profile(profile, pdpars, patient)
        ss = exposure_metrics(profile, 27.0 * HOURS_PER_DAY, 28.0 * HOURS_PER_DAY)
        nadir_idx = int(profile["epo_percent_baseline"].idxmin())
        below_50 = profile[profile["epo_percent_baseline"] <= 50.0]
        rows.append(
            {
                "scenario": name,
                "auc_ug_h_ml": ss["auc_ug_h_ml"],
                "cmax_ng_ml": ss["cmax_ng_ml"],
                "cmin_ng_ml": ss["cmin_ng_ml"],
                "epo_nadir_percent_baseline": float(profile.loc[nadir_idx, "epo_percent_baseline"]),
                "time_to_epo_nadir_days": float(profile.loc[nadir_idx, "day"]),
                "time_to_50pct_epo_days": float(below_50["day"].iloc[0]) if not below_50.empty else np.nan,
                "grade3_anemia_onset_note": (
                    "Not mechanistically identifiable from EPO alone; Hb turnover and baseline Hb are missing."
                ),
            }
        )
    return pd.DataFrame(rows)


def dose_response(pdpars: PDParameters) -> pd.DataFrame:
    rows = []
    for dose in [80.0, 120.0, 160.0]:
        profile = simulate_pk_profile(dose, 28.0 * HOURS_PER_DAY, Patient())
        profile = add_pd_to_profile(profile, pdpars)
        ss = exposure_metrics(profile, 27.0 * HOURS_PER_DAY, 28.0 * HOURS_PER_DAY)
        rows.append(
            {
                "dose_mg_qd": dose,
                **ss,
                "median_ss_epo_percent_baseline": float(
                    profile[profile["time_h"] >= 27.0 * HOURS_PER_DAY]["epo_percent_baseline"].median()
                ),
                "nadir_epo_percent_baseline": float(profile["epo_percent_baseline"].min()),
            }
        )
    return pd.DataFrame(rows)


def simulate_pkpd_anemia_risk(
    dose_mg: float = 120.0,
    days: float = 180.0,
    patient: Patient = Patient(formulation="FFP"),
    pdpars: PDParameters | None = None,
    baseline_hb_g_dl: float = 9.5,
) -> pd.DataFrame:
    """Simulate PK -> EPO suppression -> Hb timecourse for one patient.

    This uses the concentration-linked Emax model currently calibrated from the
    literature anchor unless digitized Fig. 4 data are supplied and refit.
    """

    pdpars = calibrate_pd_parameters(PUBLISHED_TABLE5["auc_ug_h_ml"] / HOURS_PER_DAY) if pdpars is None else pdpars
    profile = simulate_pk_profile(dose_mg, days * HOURS_PER_DAY, patient, dt_h=6.0)
    epo = pd_response(profile["conc_ug_ml"].to_numpy(), patient.epo_baseline_miu_ml, pdpars)
    hb_model = RBCIndirectResponseModel()
    hb = hb_model.hemoglobin_timecourse(
        profile["day"].to_numpy(),
        epo["epo_percent_baseline"].to_numpy() / 100.0,
        baseline_hb_g_dl,
    )
    result = pd.concat([profile[["time_h", "day", "conc_ng_ml"]].reset_index(drop=True), epo], axis=1)
    result["baseline_hb_g_dl"] = baseline_hb_g_dl
    result["hb_g_dl"] = hb
    result["grade3_anemia"] = hb < hb_model.grade3_threshold_g_dl
    return result


def anemia_population_simulation(
    pdpars: PDParameters,
    n: int = 1000,
    dose_mg: float = 120.0,
    days: float = 180.0,
    baseline_hb_mean: float = 9.5,
    baseline_hb_sd: float = 1.2,
    epo_hb_sensitivity: float = 0.42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Population anemia simulation using sampled baseline Hb.

    PK/EPO is held at the standard validated FFP profile here; baseline Hb
    variability drives population risk. This keeps the safety layer transparent
    until patient-level Hb and rescue-treatment data are available.
    """

    rng = np.random.default_rng(RNG_SEED)
    baseline_hb = np.clip(rng.normal(baseline_hb_mean, baseline_hb_sd, size=n), 6.5, 14.0)
    template = simulate_pkpd_anemia_risk(dose_mg=dose_mg, days=days, pdpars=pdpars, baseline_hb_g_dl=baseline_hb_mean)
    epo_fraction = template["epo_percent_baseline"].to_numpy() / 100.0
    t_days = template["day"].to_numpy()
    hb_model = RBCIndirectResponseModel(epo_hb_sensitivity=epo_hb_sensitivity)
    summaries = []
    sample_profiles = []
    for subject_id, hb0 in enumerate(baseline_hb, start=1):
        hb = hb_model.hemoglobin_timecourse(t_days, epo_fraction, hb0)
        grade3 = hb_model.grade3_anemia(hb)
        below = np.where(hb < hb_model.grade3_threshold_g_dl)[0]
        summaries.append(
            {
                "subject_id": subject_id,
                "baseline_hb_g_dl": hb0,
                "min_hb_g_dl": float(np.nanmin(hb)),
                "grade3_anemia": grade3,
                "time_to_grade3_days": float(t_days[below[0]]) if len(below) else np.nan,
                "epo_hb_sensitivity": epo_hb_sensitivity,
            }
        )
        if subject_id <= 25:
            sample = pd.DataFrame({"subject_id": subject_id, "day": t_days, "hb_g_dl": hb})
            sample_profiles.append(sample)
    return pd.DataFrame(summaries), pd.concat(sample_profiles, ignore_index=True)


def calibrate_anemia_sensitivity(
    pdpars: PDParameters,
    target_incidence: float = 0.27,
    n: int = 1000,
) -> float:
    """Calibrate Hb sensitivity to the published grade 3 anemia incidence."""

    low, high = 0.05, 0.95
    for _ in range(12):
        mid = (low + high) / 2.0
        anemia_df, _ = anemia_population_simulation(pdpars, n=n, epo_hb_sensitivity=mid)
        incidence = float(anemia_df["grade3_anemia"].mean())
        if incidence < target_incidence:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def anemia_summary_table(anemia_df: pd.DataFrame) -> pd.DataFrame:
    grade3_incidence = float(100.0 * anemia_df["grade3_anemia"].mean())
    time_to_grade3 = anemia_df.loc[anemia_df["grade3_anemia"], "time_to_grade3_days"]
    return pd.DataFrame(
        [
            {
                "n": len(anemia_df),
                "grade3_anemia_incidence_percent": grade3_incidence,
                "published_grade3_anemia_percent": 27.0,
                "median_time_to_grade3_days": float(time_to_grade3.median()) if not time_to_grade3.empty else np.nan,
                "mean_baseline_hb_g_dl": float(anemia_df["baseline_hb_g_dl"].mean()),
                "mean_min_hb_g_dl": float(anemia_df["min_hb_g_dl"].mean()),
                "epo_hb_sensitivity": float(anemia_df["epo_hb_sensitivity"].iloc[0]),
                "model_status": "calibrated_to_phase1_grade3_anemia_incidence_requires_patient_hb_rescue_data",
            }
        ]
    )


def write_plots(
    profile: pd.DataFrame,
    pd_profile: pd.DataFrame,
    dose_df: pd.DataFrame,
    output_dir: Path,
    hb_profiles: pd.DataFrame | None = None,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    conc_pct = percentile_profile(profile, "conc_ng_ml")
    epo_pct = percentile_profile(pd_profile, "epo_percent_baseline")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(conc_pct["time_h"] / HOURS_PER_DAY, conc_pct["p05"], conc_pct["p95"], alpha=0.25)
    ax.plot(conc_pct["time_h"] / HOURS_PER_DAY, conc_pct["p50"], color="black", label="median")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Belzutifan concentration (ng/mL)")
    ax.set_title("120 mg q.d. population PK simulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pk_concentration_percentiles.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(epo_pct["time_h"] / HOURS_PER_DAY, epo_pct["p05"], epo_pct["p95"], alpha=0.25)
    ax.plot(epo_pct["time_h"] / HOURS_PER_DAY, epo_pct["p50"], color="black", label="median")
    ax.axhline(50, color="tab:red", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("EPO (% baseline)")
    ax.set_title("Model-implied EPO suppression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "epo_suppression_percentiles.png", dpi=200)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(dose_df["dose_mg_qd"], dose_df["auc_ug_h_ml"], marker="o", label="AUC0-24h")
    ax1.set_xlabel("Dose (mg q.d.)")
    ax1.set_ylabel("AUC0-24h (ug*h/mL)")
    ax2 = ax1.twinx()
    ax2.plot(
        dose_df["dose_mg_qd"],
        100.0 - dose_df["median_ss_epo_percent_baseline"],
        color="tab:red",
        marker="s",
        label="Median EPO suppression",
    )
    ax2.set_ylabel("Median steady-state EPO suppression (%)")
    ax1.set_title("Dose-response check")
    fig.tight_layout()
    fig.savefig(figures_dir / "dose_response.png", dpi=200)
    plt.close(fig)

    if hb_profiles is not None and not hb_profiles.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        hb_pct = (
            hb_profiles.groupby("day")["hb_g_dl"]
            .quantile([0.05, 0.50, 0.95])
            .unstack()
            .rename(columns={0.05: "p05", 0.5: "p50", 0.95: "p95"})
            .reset_index()
        )
        ax.fill_between(hb_pct["day"], hb_pct["p05"], hb_pct["p95"], alpha=0.25)
        ax.plot(hb_pct["day"], hb_pct["p50"], color="black", label="median")
        ax.axhline(7.0, color="tab:red", linestyle="--", linewidth=1, label="Grade 3 threshold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Hemoglobin (g/dL)")
        ax.set_title("Exploratory Hb response simulation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "hemoglobin_response.png", dpi=200)
        plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a simple Markdown table without optional pandas dependencies."""

    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(
            lambda value: f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    header = "| " + " | ".join(text_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def write_report(
    output_dir: Path,
    validation: pd.DataFrame,
    metrics: pd.DataFrame,
    scenario_df: pd.DataFrame,
    pdpars: PDParameters,
    pd_fit_result: EmaxFitResult | None,
    pd_fit_df: pd.DataFrame,
    anemia_summary: pd.DataFrame,
) -> None:
    auc_rmse = float(np.sqrt(np.mean((metrics["auc_ug_h_ml"] - PUBLISHED_TABLE5["auc_ug_h_ml"]) ** 2)))
    cmax_corr_note = "Not computable: only one published aggregate Cmax value is available."
    if pd_fit_result is None:
        vpc_note = (
            "Visual predictive check style comparison is represented by 5th/50th/95th percentile plots; "
            "digitized observed EPO points from Extended Data Fig. 4 were not supplied."
        )
    else:
        vpc_note = (
            "Digitized Extended Data Fig. 4 points are fitted in epo_emax_fit.csv and plotted in "
            "epo_emax_validation.png; percentile simulation plots remain model-based."
        )
    if pd_fit_result is None:
        pd_fit_text = dataframe_to_markdown(pd_fit_df)
    else:
        pd_fit_text = dataframe_to_markdown(
            pd.DataFrame(
                [
                    {
                        "emax": pd_fit_result.emax,
                        "ec50": pd_fit_result.ec50,
                        "gamma": pd_fit_result.gamma,
                        "rmse_percent": pd_fit_result.rmse_percent,
                        "n_points": pd_fit_result.n_points,
                        "driver": pd_fit_result.driver,
                    }
                ]
            )
        )
    report = f"""# Belzutifan PK/PD Model Report

## Sources
- PK: PMC10583240, fixed population PK model parameters from Tables 2, 3, and 5.
- PD: Choueiri et al. Nat Med 2021, PMC9128828. The paper reports concentration-correlated EPO suppression, reductions at all doses, and a plateau at >=120 mg q.d.; exact Extended Data Fig. 4 numeric values were not supplied.

## Validation Scenario
- Study 4 patients received FFP during PK sampling, so FFP is used for direct Table 5 validation.
- CL/F is set to the Table 5 VHL-RCC geometric mean of 7.25 L/h for the validation baseline.
- The FMF formulation penalty on KA is retained in `clinical_scenarios.csv` as `standard_120mg_fmf`.

## PD Calibration Assumption
- Direct inhibitory Emax model: `EPO(t) = EPO_baseline * (1 - Emax*C(t)/(EC50 + C(t)))`.
- Emax: {pdpars.emax:.3f}.
- EC50: {pdpars.ec50_ug_ml:.4f} ug/mL.
- Hill gamma: {pdpars.gamma:.3f}.
- Calibration anchor: 60% EPO suppression at the published 120 mg q.d. geometric mean Cavg (`AUC0-24h/24`), selected within the supplied qualitative 40-70% substantial suppression range. This is not a digitized literature observation.
- If using an AUC-scale EC50 such as 12,000-15,000 h*ng/mL, fit with `driver='auc_to_time_ng_h_ml'` or convert to an average concentration scale before applying the concentration model.

## Extended Data Fig. 4 Fit
{pd_fit_text}

## Validation Against Published Table 5
{dataframe_to_markdown(validation)}

## Goodness-of-Fit Notes
- RMSE versus the single published AUC aggregate for the validation baseline: {auc_rmse:.3f} ug*h/mL.
- Correlation of predicted versus observed Cmax: {cmax_corr_note}
- VPC: {vpc_note}

## Clinical Prediction Limitation
EPO suppression alone cannot determine onset time to grade 3 anemia. A hemoglobin turnover model, baseline hemoglobin distribution, rescue EPO/transfusion rules, and censoring assumptions are required. The anemia module below is calibrated to the 27% phase 1 grade 3 anemia incidence; patient-level Hb and rescue-treatment data would be needed for stronger validation.

## Exploratory Anemia Simulation
{dataframe_to_markdown(anemia_summary)}

## High-Risk Scenario Predictions
{dataframe_to_markdown(scenario_df)}
"""
    (output_dir / "model_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    target_cavg = PUBLISHED_TABLE5["auc_ug_h_ml"] / HOURS_PER_DAY
    pdpars = calibrate_pd_parameters(target_cavg)
    pd_fit_result, pd_fit_df = fit_emax_to_digitized_epo()
    if pd_fit_result is not None and pd_fit_result.driver in {"concentration_ng_ml", "cavg_to_time_ng_ml"}:
        pdpars = PDParameters(
            emax=pd_fit_result.emax,
            ec50_ug_ml=pd_fit_result.ec50 / 1000.0,
            gamma=pd_fit_result.gamma,
        )

    profile, metrics = population_simulation()
    pd_profile = add_pd_to_profile(profile, pdpars)
    dose_df = dose_response(pdpars)
    scenario_df = scenario_predictions(pdpars)
    hb_sensitivity = calibrate_anemia_sensitivity(pdpars)
    anemia_df, hb_profiles = anemia_population_simulation(pdpars, epo_hb_sensitivity=hb_sensitivity)
    anemia_summary = anemia_summary_table(anemia_df)
    validation_profile = simulate_pk_profile(120.0, 28.0 * HOURS_PER_DAY, Patient(formulation="FFP"))
    validation_metrics = pd.DataFrame(
        [exposure_metrics(validation_profile, 27.0 * HOURS_PER_DAY, 28.0 * HOURS_PER_DAY)]
    )
    validation = validation_table(validation_metrics)
    pk_table, cov_table, pd_table = parameter_tables(pdpars)

    percentile_profile(profile, "conc_ng_ml").to_csv(
        output_dir / "population_pk_percentiles.csv", index=False
    )
    percentile_profile(pd_profile, "epo_percent_baseline").to_csv(
        output_dir / "population_epo_percentiles.csv", index=False
    )
    metrics.to_csv(output_dir / "population_pk_metrics.csv", index=False)
    validation.to_csv(output_dir / "validation_table.csv", index=False)
    dose_df.to_csv(output_dir / "dose_response.csv", index=False)
    scenario_df.to_csv(output_dir / "clinical_scenarios.csv", index=False)
    pd_fit_df.to_csv(output_dir / "epo_emax_fit.csv", index=False)
    anemia_df.to_csv(output_dir / "anemia_population.csv", index=False)
    anemia_summary.to_csv(output_dir / "anemia_summary.csv", index=False)
    hb_profiles.to_csv(output_dir / "hemoglobin_profiles_sample.csv", index=False)
    pk_table.to_csv(output_dir / "pk_parameter_definitions.csv", index=False)
    cov_table.to_csv(output_dir / "covariate_effects.csv", index=False)
    pd_table.to_csv(output_dir / "pd_parameter_definitions.csv", index=False)

    write_plots(profile, pd_profile, dose_df, output_dir, hb_profiles=hb_profiles)
    write_epo_fit_plot(pd_fit_df, pd_fit_result, output_dir)
    write_report(
        output_dir,
        validation,
        validation_metrics,
        scenario_df,
        pdpars,
        pd_fit_result,
        pd_fit_df,
        anemia_summary,
    )

    print("Belzutifan PK/PD simulation complete.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
