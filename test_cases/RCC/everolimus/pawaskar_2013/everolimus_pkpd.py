from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HUMAN_PHYSIOLOGY_ONCOLOGY: Dict[str, Any] = {
    "reference_weight_kg": 70,
    "reference_bsa_m2": 1.8,
    "cardiac_output_L_min": 5.0,
    "hematocrit": 0.45,
    "organs": {
        "plasma": {"V_L": 3.5},
        "red_blood_cells": {"V_L": 2.5},
        "lungs": {"V_L": 0.55, "Q_percent_CO": 100.0},
        "liver": {"V_L": 1.8, "Q_percent_CO": 25.0},
        "kidney": {"V_L": 0.3, "Q_percent_CO": 20.0},
        "brain": {"V_L": 1.4, "Q_percent_CO": 14.0},
        "heart": {"V_L": 0.3, "Q_percent_CO": 4.0},
        "pancreas": {"V_L": 0.1, "Q_percent_CO": 1.2},
        "spleen": {"V_L": 0.15, "Q_percent_CO": 1.5},
        "skin": {"V_L": 3.6, "Q_percent_CO": 5.0},
        "muscle": {"V_L": 30.0, "Q_percent_CO": 15.0},
        "adipose": {"V_L": 15.0, "Q_percent_CO": 5.0},
        "bone": {"V_L": 1.0, "Q_percent_CO": 5.0},
        "gi_tract": {"V_L": 1.2, "Q_percent_CO": 3.0},
        "carcass": {"V_L": 15.0, "Q_percent_CO": 1.3},
    },
    "tumor": {
        "baseline_SLD_cm": 14.4,
        "Qtu_mL_min": 1.5,
        "Ktu": 0.480,
        "blood_fraction": 0.05,
    },
}

TUMOR_PARAMS: Dict[str, Any] = {
    "Qtu_mL_min": 1.5,
    "Ktu": 0.480,
    "Vtu_baseline_mL": 50.0,
    "baseline_SLD_cm": 14.4,
    "comments": {
        "Qtu_source": "Literature RCC perfusion 10-50 mL/100g; scaled to 1.5 mL/min for ~100g tumor",
        "Qtu_sensitivity": "Test [0.5, 1.5, 3.0] mL/min in Phase 2 if poor fit",
        "Ktu_source": "PBPK paper; assumes constant (no dose-dependence detected for everolimus)",
        "Vtu_note": "Will implement as patient-level covariate; adjust from baseline SLD",
    },
}

DOSE_CONFIG: Dict[str, Any] = {
    "unit": "mg/day",
    "rationale": "RECORD-1 trial used fixed 10 mg QD; no weight adjustment",
    "clinical_doses": {"5mg_daily": 5.0, "10mg_daily": 10.0},
    "implementation": {
        "input": "dose_mg (e.g., 5.0 or 10.0)",
        "do_not": "Convert to mg/kg or scale by patient weight",
        "reason": "RECORD-1 was fixed-dose, not weight-adjusted; matches clinical practice",
    },
    "phase_2_note": "If needed, can add weight-adjusted variant for sensitivity analysis",
}

EVEROLIMUS_PD_PARAMETERS: Dict[str, Any] = {
    "mechanism": "mTOR inhibitor (S6K1, 4E-BP1 pathways)",
    "EC50_literature": {
        "mTOR_kinase_cellFree_nM": 1.5,
        "pancreatic_cancer_cell_IC50_nM": 20.0,
        "RCC_cancer_cell_IC50_nM": 10.0,
        "tumor_growth_inhibition_xenograft_nM": 5.0,
        "plasma_Css_at_10mg_nM": 3.0,
    },
    "recommended_phase1B": {
        "Emax_1_per_day": 0.05,
        "EC50_nM": 5.0,
        "gamma": 1.0,
    },
    "notes": [
        "EC50 = 5 nM is conservative mid-range for in vivo tumor growth inhibition",
        "If model underfits (tumors grow too much), decrease EC50 (more potent)",
        "If model overfits (tumors shrink too much), increase EC50 or decrease Emax",
        "Phase 2: Calibrate Emax and EC50 simultaneously to match RECORD-1 data",
    ],
}

PBPK_COMPARTMENTS: List[str] = [
    "liver",
    "kidney",
    "brain",
    "heart",
    "pancreas",
    "spleen",
    "skin",
    "muscle",
    "adipose",
    "bone",
    "gi_tract",
    "carcass",
    "tumor",
]

EVEROLIMUS_MW_G_MOL = 958.22


def derive_human_flows_and_volumes(
    physiology: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    organs = physiology["organs"]
    cardiac_output_mL_min = physiology["cardiac_output_L_min"] * 1000.0

    systemic_organs = [name for name in organs if name not in ("plasma", "red_blood_cells", "lungs")]
    systemic_percent = float(sum(organs[name].get("Q_percent_CO", 0.0) for name in systemic_organs))
    if not np.isclose(systemic_percent, 100.0, atol=1e-6):
        raise ValueError(
            f"Systemic organ Q_percent_CO must sum to 100%. Got {systemic_percent:.3f}%."
        )

    q_mL_min: Dict[str, float] = {}
    v_L: Dict[str, float] = {}
    for organ_name, organ_values in organs.items():
        v_L[organ_name] = float(organ_values["V_L"])
        if "Q_percent_CO" in organ_values:
            q_mL_min[organ_name] = (
                float(organ_values["Q_percent_CO"]) / 100.0 * cardiac_output_mL_min
            )

    return v_L, q_mL_min, {"systemic_q_percent_sum": systemic_percent}


@dataclass(frozen=True)
class EverolimusPBPKParams:
    context: Literal["human_clinical", "mouse_preclinical"] = "human_clinical"
    body_weight_kg: float = 70.0
    plasma_volume_L: float = HUMAN_PHYSIOLOGY_ONCOLOGY["organs"]["plasma"]["V_L"]
    blood_volume_L: float = 5.6

    # Distribution and clearance (context-overridden in __post_init__)
    Vd_L: float = 82.3
    CL_total_L_h: float = 1.37
    terminal_half_life_h: float = 41.8
    k_el_h_inv: float = 0.693 / 41.8

    # Source parameter retained from Pawaskar table
    CL_int_mL_h: float = 4.07
    CV_CL: float = 0.358

    # Absorption
    Fa: float = 0.12
    ka_1_h: float = 0.5
    Ka: float = 0.5  # Alias for legacy compatibility
    F: float = 0.12  # Alias for legacy compatibility

    # Partition coefficients
    K_tumor: float = TUMOR_PARAMS["Ktu"]
    K_liver: float = 0.452
    K_kidney: float = 0.435
    K_pancreas: float = 0.582
    K_muscle: float = 0.105
    K_adipose: float = 0.097
    K_brain: float = 0.0186
    K_skin: float = 0.232
    K_spleen: float = 0.361
    K_gi: float = 1.439
    K_lungs: float = 1.0
    K_carcass: float = 5.67
    K_heart: float = 1.0
    K_bone: float = 1.0

    CV_K: Dict[str, float] = field(
        default_factory=lambda: {
            "tumor": 0.453,
            "liver": 0.369,
            "kidney": 0.206,
            "pancreas": 0.505,
            "muscle": 0.0936,
            "adipose": 0.0667,
            "brain": 0.0636,
            "skin": 0.631,
            "spleen": 0.409,
            "gi_tract": 0.903,
        }
    )

    tumor_q_mL_min: float = TUMOR_PARAMS["Qtu_mL_min"]
    tumor_volume_mL: float = TUMOR_PARAMS["Vtu_baseline_mL"]
    distribution_type: str = "lognormal"
    model_type: str = "3-compartment oral PBPK surrogate"

    tissue_volumes_L: Dict[str, float] = field(default_factory=dict)
    tissue_flows_L_h: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context == "mouse_preclinical":
            # Pawaskar-aligned mouse defaults
            object.__setattr__(self, "body_weight_kg", 0.025)
            object.__setattr__(self, "plasma_volume_L", 0.0014)
            object.__setattr__(self, "blood_volume_L", 0.0018)
            object.__setattr__(self, "Vd_L", 0.0294)
            object.__setattr__(self, "CL_total_L_h", self.CL_int_mL_h / 1000.0)
            object.__setattr__(self, "terminal_half_life_h", 5.0)
            object.__setattr__(self, "k_el_h_inv", np.log(2.0) / 5.0)
            object.__setattr__(self, "Fa", 0.12)
            object.__setattr__(self, "ka_1_h", 9.45)
            object.__setattr__(self, "Ka", 9.45)
            object.__setattr__(self, "F", 0.12)
            object.__setattr__(self, "tumor_volume_mL", 0.05)
        else:
            # Human context from allometric scaling of mouse reference
            bw_ratio = 70.0 / 0.025
            vd = 0.0294 * (bw_ratio**1.0)
            cl = (self.CL_int_mL_h / 1000.0) * (bw_ratio**0.75)
            t_half = (0.693 * vd) / cl
            object.__setattr__(self, "body_weight_kg", 70.0)
            object.__setattr__(self, "plasma_volume_L", 3.5)
            object.__setattr__(self, "blood_volume_L", 5.6)
            object.__setattr__(self, "Vd_L", vd)
            object.__setattr__(self, "CL_total_L_h", cl)
            object.__setattr__(self, "terminal_half_life_h", t_half)
            object.__setattr__(self, "k_el_h_inv", np.log(2.0) / t_half)
            object.__setattr__(self, "Fa", 0.12)
            object.__setattr__(self, "ka_1_h", 0.5)
            object.__setattr__(self, "Ka", 0.5)
            object.__setattr__(self, "F", 0.12)

        if self.tissue_volumes_L and self.tissue_flows_L_h:
            return

        volumes, flows_mL_min, _ = derive_human_flows_and_volumes(HUMAN_PHYSIOLOGY_ONCOLOGY)
        tissue_volumes = {comp: float(volumes[comp]) for comp in PBPK_COMPARTMENTS if comp != "tumor"}
        tissue_flows = {
            comp: float(flows_mL_min[comp]) / 1000.0 * 60.0
            for comp in PBPK_COMPARTMENTS
            if comp != "tumor"
        }

        if self.context == "mouse_preclinical":
            # Scale non-tumor volumes roughly by body-weight ratio for mouse output bookkeeping.
            scale = self.body_weight_kg / 70.0
            tissue_volumes = {k: v * scale for k, v in tissue_volumes.items()}
            tissue_flows = {k: v * (scale**0.75) for k, v in tissue_flows.items()}

        tissue_volumes["tumor"] = self.tumor_volume_mL / 1000.0
        tissue_flows["tumor"] = self.tumor_q_mL_min / 1000.0 * 60.0
        object.__setattr__(self, "tissue_volumes_L", tissue_volumes)
        object.__setattr__(self, "tissue_flows_L_h", tissue_flows)

    @property
    def CL_int_L_h(self) -> float:
        return self.CL_int_mL_h / 1000.0

    def k_by_compartment(self) -> Dict[str, float]:
        return {
            "lungs": self.K_lungs,
            "liver": self.K_liver,
            "kidney": self.K_kidney,
            "brain": self.K_brain,
            "heart": self.K_heart,
            "pancreas": self.K_pancreas,
            "spleen": self.K_spleen,
            "skin": self.K_skin,
            "muscle": self.K_muscle,
            "adipose": self.K_adipose,
            "bone": self.K_bone,
            "gi_tract": self.K_gi,
            "carcass": self.K_carcass,
            "tumor": self.K_tumor,
        }


@dataclass(frozen=True)
class PDParams:
    r_per_day: float = 46.0e-3
    E10_per_day: float = 3.9e-3
    E5_per_day: float = 2.3e-3
    baseline_SLD_cm: float = TUMOR_PARAMS["baseline_SLD_cm"]
    theta1: float = 0.4
    theta2: float = -0.7


def _lognormal_from_mean_cv(mean_value: float, cv: float, rng: np.random.Generator) -> float:
    sigma = np.sqrt(np.log1p(cv**2))
    mu = np.log(mean_value) - 0.5 * sigma**2
    return float(rng.lognormal(mu, sigma))


def sample_individual_params(
    base_params: EverolimusPBPKParams, rng: np.random.Generator
) -> EverolimusPBPKParams:
    sampled_values = {
        "CL_int_mL_h": _lognormal_from_mean_cv(base_params.CL_int_mL_h, base_params.CV_CL, rng),
        "K_tumor": _lognormal_from_mean_cv(base_params.K_tumor, base_params.CV_K["tumor"], rng),
        "K_liver": _lognormal_from_mean_cv(base_params.K_liver, base_params.CV_K["liver"], rng),
        "K_kidney": _lognormal_from_mean_cv(base_params.K_kidney, base_params.CV_K["kidney"], rng),
        "K_pancreas": _lognormal_from_mean_cv(base_params.K_pancreas, base_params.CV_K["pancreas"], rng),
        "K_muscle": _lognormal_from_mean_cv(base_params.K_muscle, base_params.CV_K["muscle"], rng),
        "K_adipose": _lognormal_from_mean_cv(base_params.K_adipose, base_params.CV_K["adipose"], rng),
        "K_brain": _lognormal_from_mean_cv(base_params.K_brain, base_params.CV_K["brain"], rng),
        "K_skin": _lognormal_from_mean_cv(base_params.K_skin, base_params.CV_K["skin"], rng),
        "K_spleen": _lognormal_from_mean_cv(base_params.K_spleen, base_params.CV_K["spleen"], rng),
        "K_gi": _lognormal_from_mean_cv(base_params.K_gi, base_params.CV_K["gi_tract"], rng),
    }
    return replace(base_params, **sampled_values)


def mg_per_l_to_nm(concentration_mg_per_L: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(concentration_mg_per_L) * 1e6 / EVEROLIMUS_MW_G_MOL


def mechanistic_pd_link(C_tumor_nM: float, Emax: float, EC50_nM: float, gamma: float = 1.0) -> float:
    """
    Hill equation: E = Emax × C^γ / (EC50^γ + C^γ)

    Args:
        C_tumor_nM: Tumor concentration in nanoMolar
        Emax: Maximum tumor growth inhibition (1/day)
        EC50_nM: Concentration at half-maximal effect
        gamma: Hill coefficient

    Returns:
        E_dose: Drug effect on tumor growth (1/day)
    """
    if C_tumor_nM <= 0:
        return 0.0
    numerator = Emax * (C_tumor_nM**gamma)
    denominator = (EC50_nM**gamma) + (C_tumor_nM**gamma)
    return float(numerator / denominator)


def empirical_pd_link(dose_mg: float) -> float:
    """
    Dose-based empirical link (fallback), direct mapping from RECORD-1.
    """
    if dose_mg >= 10.0:
        return 3.9e-3
    if dose_mg >= 5.0:
        return 2.3e-3
    return 0.0


def interp_concentration(c_tumor_timeseries: pd.DataFrame, t_day: float) -> float:
    return float(
        np.interp(
            t_day,
            c_tumor_timeseries["time_day"].to_numpy(),
            c_tumor_timeseries["C_tumor_nM"].to_numpy(),
        )
    )


def pd_ode_solver(
    y_tumor_SLD: float,
    t_day: float,
    C_tumor_timeseries: pd.DataFrame,
    link_mode: Literal["mechanistic", "empirical"] = "mechanistic",
    link_params: Dict[str, float] | None = None,
) -> float:
    """
    Tumor growth ODE: dy/dt = r - E_dose × y.
    """
    r = 46.0e-3
    C_tumor_nM = interp_concentration(C_tumor_timeseries, t_day)
    params = link_params or {}

    if link_mode == "mechanistic":
        Emax = float(params.get("Emax", EVEROLIMUS_PD_PARAMETERS["recommended_phase1B"]["Emax_1_per_day"]))
        EC50_nM = float(params.get("EC50_nM", EVEROLIMUS_PD_PARAMETERS["recommended_phase1B"]["EC50_nM"]))
        gamma = float(params.get("gamma", EVEROLIMUS_PD_PARAMETERS["recommended_phase1B"]["gamma"]))
        E_dose = mechanistic_pd_link(C_tumor_nM=C_tumor_nM, Emax=Emax, EC50_nM=EC50_nM, gamma=gamma)
    elif link_mode == "empirical":
        dose_mg = float(params.get("dose_mg", 10.0))
        E_dose = empirical_pd_link(dose_mg=dose_mg)
    else:
        raise ValueError(f"Unknown link_mode: {link_mode}")

    return r - E_dose * y_tumor_SLD


def _pbpk_rhs(_t: float, y: np.ndarray, params: EverolimusPBPKParams) -> np.ndarray:
    A_gut = y[0]
    A_plasma = y[1]
    tissue_amounts = y[2:]

    k_map = params.k_by_compartment()
    v_map = params.tissue_volumes_L
    q_map = params.tissue_flows_L_h
    C_plasma = A_plasma / params.plasma_volume_L

    """
    EVEROLIMUS ORAL BIOAVAILABILITY (F = 0.05)

    Decomposition (not explicitly separated in current model):
        Fa (Fraction Absorbed)       = 0.12
        Fg (Gut First-Pass)          = ~0.75
        Fh (Hepatic First-Pass)      = ~0.67

        Net F = Fa × Fg × Fh = 0.12 × 0.75 × 0.67 ≈ 0.06 ≈ 0.05

    Current Implementation: F = 0.05 (aggregate)
        - Represents net oral bioavailability after all first-pass losses
        - Adequate for Phase 1B (population-level predictions)
        - Phase 2 refinement: Separate Fa, Fg, Fh if variability remains unexplained
    """
    F = params.F

    dA_gut = -params.Ka * A_gut
    dA_plasma = F * params.Ka * A_gut
    dA_tissue = np.zeros(len(PBPK_COMPARTMENTS), dtype=float)

    for idx, organ in enumerate(PBPK_COMPARTMENTS):
        C_tissue = tissue_amounts[idx] / v_map[organ]
        flux = q_map[organ] * (C_plasma - C_tissue / k_map[organ])
        dA_tissue[idx] = flux
        dA_plasma -= flux

    liver_idx = PBPK_COMPARTMENTS.index("liver")
    C_liver = tissue_amounts[liver_idx] / v_map["liver"]
    dA_tissue[liver_idx] -= params.CL_int_L_h * C_liver

    return np.concatenate(([dA_gut, dA_plasma], dA_tissue))


def simulate_pbpk(
    params: EverolimusPBPKParams,
    dose_mg: float,
    duration_days: int,
    dosing_interval_h: float = 24.0,
    dt_h: float = 1.0,
    context: Literal["human_clinical", "mouse_preclinical"] = "human_clinical",
    dose_unit: Literal["mg", "mg_per_kg"] = "mg",
    model: Literal["threecompartment", "onecompartment"] = "onecompartment",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Simplified 3-compartment PBPK surrogate:
    - Gut (absorption)
    - Central
    - Peripheral
    """
    total_h = float(duration_days * 24)
    times = np.arange(0.0, total_h + dt_h, dt_h)
    dose_times = set(np.round(np.arange(0.0, total_h + dosing_interval_h, dosing_interval_h), 8))
    if model == "threecompartment":
        y = np.zeros(3, dtype=float)  # [A_gut, A_central, A_peripheral]
    else:
        y = np.zeros(2, dtype=float)  # [A_gut, A_central]
    k_map = params.k_by_compartment()
    v_map = params.tissue_volumes_L
    k_gut = params.ka_1_h

    if model == "threecompartment":
        if context == "mouse_preclinical":
            V_central = 0.004
            V_peripheral = max(params.Vd_L - V_central, 1e-9)
            k_cp = 0.5
            k_pc = k_cp * V_central / V_peripheral
        else:
            V_central = params.plasma_volume_L * 1.3
            V_peripheral = max(params.Vd_L - V_central, 1e-9)
            k_cp = 0.1
            k_pc = k_cp * V_central / V_peripheral
    else:
        V_central = params.Vd_L
        V_peripheral = 0.0
        k_cp = 0.0
        k_pc = 0.0

    expected_c_ss_mg_L = None
    expected_c_tumor_nM = None
    if debug:
        print("\n=== PBPK SIMULATION DEBUG ===")
        print(f"Context: {context}")
        print(f"Input dose: {dose_mg} mg")
        print(f"Fa: {params.Fa}")
        print(f"Dose absorbed per dosing: {dose_mg * params.Fa} mg")
        print(f"CL_int: {params.CL_int_L_h} L/h")
        if context == "mouse_preclinical":
            k_el = params.CL_int_L_h / V_central
            print("\nCompartment volumes:")
            print(f"  V_central: {V_central * 1000:.1f} mL")
            print(f"  V_peripheral: {V_peripheral * 1000:.1f} mL")
            print(f"  V_total: {params.Vd_L * 1000:.1f} mL")
            if model == "threecompartment":
                print("\nDistribution parameters:")
                print(f"  k_cp (central->peripheral): {k_cp:.4f} h^-1")
                print(f"  k_pc (peripheral->central): {k_pc:.4f} h^-1")
                print(f"  Equilibration half-life (k_cp): {0.693 / max(k_cp, 1e-12):.3f} h")
                print(f"  V_central/V_peripheral ratio: {V_central / max(V_peripheral, 1e-12):.6f}")
            print("\nElimination:")
            print(f"  k_el (from central): {k_el:.4f} h^-1")
            print(f"  Central t1/2: {0.693 / k_el:.3f} h")
            dose_absorbed_daily = dose_mg * params.Fa
            expected_c_ss_mg_L = dose_absorbed_daily / (params.CL_int_L_h * 24.0)
            expected_c_tumor_nM = mg_per_l_to_nm(expected_c_ss_mg_L * params.K_tumor)
            print("\nExpected steady-state (from dose balance):")
            print(f"  Dose absorbed/day: {dose_absorbed_daily} mg")
            print(f"  Expected C_central,ss: {expected_c_ss_mg_L:.6f} mg/L")
            print(f"  Expected C_tumor,ss: {float(expected_c_tumor_nM):.1f} nM")

    rows: List[Dict[str, float]] = []
    for i, t_now in enumerate(times):
        if np.round(t_now, 8) in dose_times:
            administered_mg = dose_mg * params.body_weight_kg if dose_unit == "mg_per_kg" else dose_mg
            y[0] += administered_mg * params.Fa

        C_central = y[1] / V_central
        row: Dict[str, float] = {
            "time_h": float(t_now),
            "A_gut_mg": float(y[0]),
            "A_central_mg": float(y[1]),
            "A_peripheral_mg": float(y[2] if model == "threecompartment" else 0.0),
            "A_plasma_mg": float(y[1]),
            "C_plasma_mg_per_L": float(C_central),
        }
        row["C_lungs_mg_per_L"] = row["C_plasma_mg_per_L"]
        row["A_lungs_mg"] = row["C_lungs_mg_per_L"] * HUMAN_PHYSIOLOGY_ONCOLOGY["organs"]["lungs"]["V_L"]

        for comp in PBPK_COMPARTMENTS:
            C_tissue = k_map[comp] * C_central
            row[f"C_{comp}_mg_per_L"] = float(C_tissue)
            row[f"A_{comp}_mg"] = float(C_tissue * v_map[comp])
        rows.append(row)

        if debug and context == "mouse_preclinical" and t_now >= duration_days * 24 - 1:
            C_tumor_nM = float(mg_per_l_to_nm(k_map["tumor"] * C_central))
            ratio = (C_central / expected_c_ss_mg_L) if expected_c_ss_mg_L else np.nan
            print(f"\nDay {t_now / 24:.1f} observed:")
            print(f"  C_central: {C_central:.6f} mg/L")
            print(f"  C_tumor: {C_tumor_nM:.1f} nM")
            print(f"  Ratio (observed/expected): {ratio:.1f}x")

        if i < len(times) - 1:
            dt = float(times[i + 1] - t_now)
            dA_gut = -k_gut * y[0]
            elimination = params.CL_total_L_h * (y[1] / max(V_central, 1e-12))
            if model == "threecompartment":
                dA_central = k_gut * y[0] - elimination - k_cp * y[1] + k_pc * y[2]
                dA_peripheral = k_cp * y[1] - k_pc * y[2]
                y[2] = max(0.0, y[2] + dA_peripheral * dt)
            else:
                dA_central = k_gut * y[0] - elimination

            y[0] = max(0.0, y[0] + dA_gut * dt)
            y[1] = max(0.0, y[1] + dA_central * dt)

    pbpk_df = pd.DataFrame(rows)
    pbpk_df["time_day"] = pbpk_df["time_h"] / 24.0
    pbpk_df["C_tumor_nM"] = mg_per_l_to_nm(pbpk_df["C_tumor_mg_per_L"])
    return pbpk_df


def simulate_pd(
    pbpk_df: pd.DataFrame,
    dose_mg: float,
    pd_params: PDParams,
    link_mode: Literal["mechanistic", "empirical"] = "mechanistic",
    link_params: Dict[str, float] | None = None,
    baseline_sld_cm: float | None = None,
) -> pd.DataFrame:
    times_day = pbpk_df["time_day"].to_numpy()
    y = np.zeros_like(times_day)
    y[0] = baseline_sld_cm if baseline_sld_cm is not None else pd_params.baseline_SLD_cm

    baseline = y[0]
    baseline_factor_growth = (baseline / pd_params.baseline_SLD_cm) ** pd_params.theta1
    baseline_factor_effect = (baseline / pd_params.baseline_SLD_cm) ** pd_params.theta2

    c_timeseries = pbpk_df[["time_day", "C_tumor_nM"]].copy()
    local_link = dict(link_params or {})
    local_link.setdefault("dose_mg", dose_mg)

    e_profile = np.zeros_like(times_day)
    for i, t_day in enumerate(times_day):
        c_now = interp_concentration(c_timeseries, float(t_day))
        if link_mode == "mechanistic":
            e_profile[i] = mechanistic_pd_link(
                C_tumor_nM=c_now,
                Emax=float(local_link.get("Emax", 0.05)),
                EC50_nM=float(local_link.get("EC50_nM", 5.0)),
                gamma=float(local_link.get("gamma", 1.0)),
            )
        else:
            e_profile[i] = empirical_pd_link(float(local_link.get("dose_mg", dose_mg)))

    e_profile = e_profile * baseline_factor_effect
    r_eff = pd_params.r_per_day * baseline_factor_growth

    for i in range(1, len(times_day)):
        dt = times_day[i] - times_day[i - 1]
        dydt = r_eff - e_profile[i - 1] * y[i - 1]
        y[i] = max(0.0, y[i - 1] + dt * dydt)

    return pd.DataFrame(
        {
            "time_h": pbpk_df["time_h"].to_numpy(),
            "time_day": times_day,
            "E_effect_per_day": e_profile,
            "tumor_size_cm": y,
        }
    )


def run_population_simulation(
    dose_mg: float,
    n_subjects: int = 100,
    duration_days: int = 365,
    seed: int = 42,
    link_mode: Literal["mechanistic", "empirical"] = "mechanistic",
    link_params: Dict[str, float] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_pbpk = EverolimusPBPKParams()
    pd_params = PDParams()
    all_subjects: List[pd.DataFrame] = []

    for subject_id in range(n_subjects):
        sampled_pbpk = sample_individual_params(base_pbpk, rng)
        pbpk_df = simulate_pbpk(
            params=sampled_pbpk,
            dose_mg=dose_mg,
            duration_days=duration_days,
        )
        pd_df = simulate_pd(
            pbpk_df=pbpk_df,
            dose_mg=dose_mg,
            pd_params=pd_params,
            link_mode=link_mode,
            link_params=link_params,
        )
        merged = pbpk_df.merge(pd_df, on=["time_h", "time_day"], how="left")
        merged["subject_id"] = subject_id
        all_subjects.append(merged)

    return pd.concat(all_subjects, ignore_index=True)


def summarize_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        pop_df.groupby("time_h")[["C_plasma_mg_per_L", "C_tumor_mg_per_L", "tumor_size_cm"]]
        .agg(["mean", "std", lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)])
        .reset_index()
    )
    summary.columns = [
        "time_h",
        "C_plasma_mean",
        "C_plasma_sd",
        "C_plasma_p5",
        "C_plasma_p95",
        "C_tumor_mean",
        "C_tumor_sd",
        "C_tumor_p5",
        "C_tumor_p95",
        "tumor_mean",
        "tumor_sd",
        "tumor_p5",
        "tumor_p95",
    ]
    return summary


def plot_profiles(summary_df: pd.DataFrame, title_prefix: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_day = summary_df["time_h"] / 24.0

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(t_day, summary_df["C_plasma_mean"], label="Plasma mean")
    ax1.fill_between(t_day, summary_df["C_plasma_p5"], summary_df["C_plasma_p95"], alpha=0.2)
    ax1.plot(t_day, summary_df["C_tumor_mean"], label="Tumor mean")
    ax1.fill_between(t_day, summary_df["C_tumor_p5"], summary_df["C_tumor_p95"], alpha=0.2)
    ax1.set_xlabel("Time (day)")
    ax1.set_ylabel("Concentration (mg/L)")
    ax1.set_title(f"{title_prefix}: PBPK concentration profiles")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{title_prefix}_pk_profiles.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(t_day, summary_df["tumor_mean"], label="Tumor size mean")
    ax2.fill_between(t_day, summary_df["tumor_p5"], summary_df["tumor_p95"], alpha=0.2, label="5-95%")
    ax2.set_xlabel("Time (day)")
    ax2.set_ylabel("SLD (cm)")
    ax2.set_title(f"{title_prefix}: PD tumor trajectory (VPC-style band)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{title_prefix}_pd_vpc.png", dpi=150)
    plt.close(fig2)


def validate_against_record1(
    mean_pct_change: float, dose_mg: float, target_change_5mg: float = 22.4, target_change_10mg: float = -15.7
) -> Dict[str, float]:
    target = target_change_10mg if np.isclose(dose_mg, 10.0) else target_change_5mg
    return {
        "dose_mg": dose_mg,
        "predicted_pct_change_1y": float(mean_pct_change),
        "target_pct_change_1y": float(target),
        "absolute_error_pct_points": float(abs(mean_pct_change - target)),
    }


def run_pipeline(
    doses_mg: Tuple[float, ...] = (5.0, 10.0),
    duration_days: int = 365,
    n_subjects: int = 100,
    link_mode: Literal["mechanistic", "empirical"] = "mechanistic",
    link_params: Dict[str, float] | None = None,
    out_dir: Path = Path("outputs"),
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []

    for dose in doses_mg:
        pop_df = run_population_simulation(
            dose_mg=dose,
            n_subjects=n_subjects,
            duration_days=duration_days,
            link_mode=link_mode,
            link_params=link_params,
        )
        summary_df = summarize_population(pop_df)
        plot_profiles(summary_df, title_prefix=f"dose_{dose:g}mg_{link_mode}", out_dir=out_dir)
        summary_df.to_csv(out_dir / f"summary_dose_{dose:g}mg_{link_mode}.csv", index=False)
        pop_df.to_csv(out_dir / f"population_dose_{dose:g}mg_{link_mode}.csv", index=False)

        subject_end = (
            pop_df.sort_values(["subject_id", "time_h"])
            .groupby("subject_id", as_index=False)
            .tail(1)[["subject_id", "tumor_size_cm"]]
            .rename(columns={"tumor_size_cm": "SLD_final_cm"})
        )
        subject_end["SLD_baseline_cm"] = TUMOR_PARAMS["baseline_SLD_cm"]
        subject_end["SLD_change_pct"] = (
            (subject_end["SLD_final_cm"] - subject_end["SLD_baseline_cm"])
            / subject_end["SLD_baseline_cm"]
            * 100.0
        )
        subject_end.to_csv(out_dir / f"subject_endpoints_dose_{dose:g}mg_{link_mode}.csv", index=False)

        mean_change = float(subject_end["SLD_change_pct"].mean())
        sd_change = float(subject_end["SLD_change_pct"].std(ddof=0))
        validation = validate_against_record1(mean_change, dose_mg=dose)
        rows.append(
            {
                **validation,
                "std_pct_change_1y": sd_change,
                "n_subjects": float(len(subject_end)),
            }
        )

    results_df = pd.DataFrame(rows)
    return results_df


def run_phase1b_smoke_test(out_dir: Path = Path("results_phase1b")) -> Dict[float, Dict[str, float]]:
    """
    Phase 1B validation: 5 mg vs 10 mg, 365 days, mechanistic link.
    """
    print("=" * 80)
    print("PHASE 1B SMOKE TEST: PBPK/PD Model (Mouse -> Human Translation)")
    print("=" * 80)

    results: Dict[float, Dict[str, float]] = {}
    for dose in (5.0, 10.0):
        print(f"\n--- Running {dose} mg daily, 365 days, n=100 subjects ---")
        df = run_pipeline(
            doses_mg=(dose,),
            duration_days=365,
            n_subjects=100,
            link_mode="mechanistic",
            link_params={"Emax": 0.05, "EC50_nM": 5.0, "gamma": 1.0},
            out_dir=out_dir,
        )
        mean_change = float(df["predicted_pct_change_1y"].iloc[0])
        std_change = float(df["std_pct_change_1y"].iloc[0])
        results[dose] = {
            "mean_pct_change": mean_change,
            "std_pct_change": std_change,
            "n_subjects": float(df["n_subjects"].iloc[0]),
        }
        print(f"  Mean SLD change: {mean_change:+.1f}% (SD {std_change:.1f}%)")

    print("\n" + "=" * 80)
    print("RECORD-1 TARGETS vs PHASE 1B RESULTS:")
    print("=" * 80)
    print("\n5 mg Daily:")
    print("  RECORD-1 target:  +22.4% +/- 17.2%")
    print(f"  Phase 1B result:  {results[5.0]['mean_pct_change']:+.1f}% +/- {results[5.0]['std_pct_change']:.1f}%")
    print("\n10 mg Daily:")
    print("  RECORD-1 target:  -15.7% +/- 11.5%")
    print(f"  Phase 1B result:  {results[10.0]['mean_pct_change']:+.1f}% +/- {results[10.0]['std_pct_change']:.1f}%")
    print("\n" + "=" * 80)

    summary_phase1b = pd.DataFrame(
        [
            {
                "dose_mg": dose,
                "predicted_pct_change_1y": values["mean_pct_change"],
                "predicted_sd_pct_change_1y": values["std_pct_change"],
                "target_pct_change_1y": 22.4 if np.isclose(dose, 5.0) else -15.7,
            }
            for dose, values in results.items()
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_phase1b.to_csv(out_dir / "summary_phase1b.csv", index=False)
    return results


def diagnostic_phase1b(
    dose_mg: float = 0.125,
    duration_days: int = 7,
    emax: float = 0.05,
    ec50_nM: float = 5.0,
    gamma: float = 1.0,
    context: Literal["human_clinical", "mouse_preclinical"] = "human_clinical",
    model: Literal["auto", "threecompartment", "onecompartment"] = "auto",
) -> pd.DataFrame:
    selected_model: Literal["threecompartment", "onecompartment"]
    if model == "auto":
        selected_model = "onecompartment"
    else:
        selected_model = model

    if context == "mouse_preclinical":
        mouse_params = EverolimusPBPKParams(context="mouse_preclinical")
        pbpk_df = simulate_pbpk(
            mouse_params,
            dose_mg=dose_mg,
            duration_days=duration_days,
            context="mouse_preclinical",
            dose_unit="mg",
            model=selected_model,
            debug=True,
        )
    else:
        human_params = EverolimusPBPKParams(context="human_clinical")
        pbpk_df = simulate_pbpk(
            human_params,
            dose_mg=dose_mg,
            duration_days=duration_days,
            context="human_clinical",
            dose_unit="mg",
            model=selected_model,
        )
    day_markers = np.arange(0.0, float(duration_days) + 1.0, 1.0)
    diag = (
        pbpk_df[pbpk_df["time_day"].isin(day_markers)][
            ["time_day", "C_plasma_mg_per_L", "C_tumor_mg_per_L", "C_tumor_nM"]
        ]
        .copy()
        .reset_index(drop=True)
    )
    diag["E_mechanistic_1_per_day"] = diag["C_tumor_nM"].apply(
        lambda c: mechanistic_pd_link(c, Emax=emax, EC50_nM=ec50_nM, gamma=gamma)
    )
    diag["E_empirical_1_per_day"] = empirical_pd_link(dose_mg=dose_mg)
    return diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Everolimus PBPK/PD Phase 1B tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run PBPK/PD simulation")
    run_parser.add_argument("--dose", type=float, default=None, help="Absolute dose in mg/day (e.g., 5 or 10)")
    run_parser.add_argument("--duration-days", type=int, default=365)
    run_parser.add_argument("--n-subjects", type=int, default=100)
    run_parser.add_argument("--link-mode", choices=["mechanistic", "empirical"], default="mechanistic")
    run_parser.add_argument("--emax", type=float, default=0.05)
    run_parser.add_argument("--ec50-nm", type=float, default=5.0)
    run_parser.add_argument("--gamma", type=float, default=1.0)
    run_parser.add_argument("--output-dir", type=str, default="results_phase1b")

    smoke_parser = subparsers.add_parser("smoke-test", help="Run 5 mg vs 10 mg Phase 1B smoke test")
    smoke_parser.add_argument("--output-dir", type=str, default="results_phase1b")

    diag_parser = subparsers.add_parser("diagnostic", help="Run Phase 1B diagnostic table")
    diag_parser.add_argument("--dose", type=float, default=None)
    diag_parser.add_argument("--duration-days", type=int, default=7)
    diag_parser.add_argument("--emax", type=float, default=0.05)
    diag_parser.add_argument("--ec50-nm", type=float, default=5.0)
    diag_parser.add_argument("--gamma", type=float, default=1.0)
    diag_parser.add_argument(
        "--model",
        choices=["auto", "threecompartment", "onecompartment"],
        default="auto",
    )
    diag_parser.add_argument(
        "--context",
        choices=["human_clinical", "mouse_preclinical"],
        default="human_clinical",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        doses = (args.dose,) if args.dose is not None else (
            DOSE_CONFIG["clinical_doses"]["5mg_daily"],
            DOSE_CONFIG["clinical_doses"]["10mg_daily"],
        )
        link_params = {"Emax": args.emax, "EC50_nM": args.ec50_nm, "gamma": args.gamma}
        if args.link_mode == "empirical":
            link_params["dose_mg"] = doses[0]
        results = run_pipeline(
            doses_mg=doses,
            duration_days=args.duration_days,
            n_subjects=args.n_subjects,
            link_mode=args.link_mode,
            link_params=link_params,
            out_dir=Path(args.output_dir),
        )
        results.to_csv(Path(args.output_dir) / "validation_report.csv", index=False)
        print(results.to_string(index=False))
    elif args.command == "smoke-test":
        run_phase1b_smoke_test(out_dir=Path(args.output_dir))
    elif args.command == "diagnostic":
        diag_default_dose = 0.125 if args.context == "mouse_preclinical" else 10.0
        diag_df = diagnostic_phase1b(
            dose_mg=(diag_default_dose if args.dose is None else args.dose),
            duration_days=args.duration_days,
            emax=args.emax,
            ec50_nM=args.ec50_nm,
            gamma=args.gamma,
            context=args.context,
            model=args.model,
        )
        print(diag_df.to_string(index=False))
        print(
            f"\nC_tumor_nM range: {diag_df['C_tumor_nM'].min():.3f} to {diag_df['C_tumor_nM'].max():.3f}"
        )


if __name__ == "__main__":
    main()
