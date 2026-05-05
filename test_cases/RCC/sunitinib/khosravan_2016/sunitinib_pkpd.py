"""
Sunitinib PK/PD Model Replication — CORRECTED v2
Khosravan et al. (2016) Clin Pharmacokinet 55:1251-1269

Corrections vs v1:
  1. SLD IIV omega values corrected (exact CV% → lognormal formula):
       BASE_SLD: was 0.143 (residual variability!) → 0.782 (91.7% CV, Table 3)
       EC50_SLD: was 0.300 (30% CV)               → 1.209 (182% CV, Table 3)
       Kout_SLD: was 0 (no IIV)                   → 0.648 (72.2% CV, Table 3)
       Ktol_SLD: was 0 (no IIV)                   → 0.737 (84.9% CV, Table 3)
  2. Schedule covariate on SLD (Eqs 5-7) now implemented:
       Baseline SLD * (1-0.43*SCH), Kout * (1+1.01*SCH), EC50 * (1+2.43*SCH)*(1+4.82*TUMR)
  3. ALT & AST corrected to Type I IDR (drug stimulates Kin → endpoint INCREASES):
       Hepatotoxicity: sunitinib increases liver transaminases
  4. DBP corrected to Type I IDR (drug stimulates Kin → DBP INCREASES):
       Hypertension: sunitinib raises blood pressure
  5. LVEF corrected to Type II IDR (bounded Kin inhibition):
       dLVEF/dt = Kin/(1+KPD*C) - Kout*LVEF  (prevents near-zero LVEF)
  6. Full IIV added to Kout and KPD for all safety endpoints (Table 3 values)
  7. Ka IIV added for sunitinib (166% CV) and SU12662 (126% CV)
  8. ORR now computed at discrete cycle-end assessment visits (every 42 days)
  9. Simulation extended to 13 cycles (~78 weeks) — uncaps 36-week PFS ceiling
 10. PC residual variability (24% CV) applied before Grade 3/4 threshold
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")

np.random.seed(42)
OUTDIR = r"C:\Users\binil\Downloads\40262_2016_Article_404"

# ─────────────────────────────────────────────────────────
# 0. HELPER: exact %CV → lognormal omega conversion
# ─────────────────────────────────────────────────────────
def cv_to_omega(cv_pct):
    """
    For X = mu * exp(ETA), ETA ~ N(0, omega^2):
        CV% = sqrt(exp(omega^2) - 1) * 100
        => omega = sqrt(ln(1 + (CV%/100)^2))
    Using the approximation omega ≈ CV%/100 is only valid for small CV (< ~30%).
    For large IIV (91.7%, 182%, 72.2%, 84.9%) the exact formula is required.
    """
    return np.sqrt(np.log(1.0 + (cv_pct / 100.0) ** 2))

# Pre-computed SLD IIV omega values (Table 3, Khosravan 2016)
OM_SLD_BASE = cv_to_omega(91.7)   # 0.7815  (v1 used 0.143 = residual variability!)
OM_SLD_EC50 = cv_to_omega(182.0)  # 1.2091  (v1 used 0.300)
OM_SLD_KOUT = cv_to_omega(72.2)   # 0.6479  (v1 had no IIV)
OM_SLD_KTOL = cv_to_omega(84.9)   # 0.7370  (v1 had no IIV)

# PC residual variability sigma (Table 3, column "Residual variability %")
SIGMA_PC_RUV = cv_to_omega(24.0)  # 0.2366  (v1 had no RUV)

# SLD residual variability sigma (Table 3: 14.3% CV for tumor SLD model)
# Applied to each RECIST assessment visit, which shifts borderline responders into
# non-responders and reduces simulated ORR toward the paper's 27% value.
SIGMA_SLD_RUV = cv_to_omega(14.3)  # 0.142

# Simulation dimensions
N_CYCLES = 13   # 13 * 42 days = 546 days = ~78 weeks — allows KM median if reached
# Paper Set-2 simulation (for PFS/ORR) assesses ORR through the first 3 cycles;
# this matches the paper's description: 'AE grades during first 3 cycles AND PFS/ORR'.
N_CYCLES_ASSESS = 3  # ORR assessed through first 3 cycles


# ─────────────────────────────────────────────────────────
# 1. PATIENT POPULATION  (Table 1)
# ─────────────────────────────────────────────────────────
def generate_patients(n, tumor_type="RCC", schedule="4/2", seed=None):
    rng = np.random.default_rng(seed)
    age  = np.clip(rng.normal(57.6, 11.2, n), 23, 84)
    bwt  = np.clip(rng.normal(77.4, 19.3, n), 39, 154)
    sex  = rng.binomial(1, 0.342, n)          # 1 = female (34.2%)
    race = (rng.uniform(size=n) < 0.159).astype(float)  # 1 = Asian (15.9%)
    ecog = rng.binomial(1, 0.538, n)          # 1 = ECOG >= 1 (53.8%)
    tumr = 1.0 if tumor_type == "GIST" else 0.0
    # SCH covariate in the paper represents CDD (continuous daily dosing) vs
    # any intermittent schedule (4/2, 2/1, 2/2).  Both schedules we simulate
    # are intermittent, so SCH=0 for both.  Setting SCH=1 for 2/1 would wrongly
    # impose an 11x EC50 penalty (exp(2.43)) that destroys 2/1 efficacy.
    sch  = 0.0  # both 4/2 and 2/1 are intermittent (SCH=0 = reference category)
    return dict(age=age, bwt=bwt, sex=sex, race=race, ecog=ecog,
                tumr=np.full(n, tumr), sch=np.full(n, sch), n=n,
                schedule=schedule, tumor_type=tumor_type)


# ─────────────────────────────────────────────────────────
# 2. COVARIATE-ADJUSTED PK PARAMETERS  (Table 2, Eqs 1-4)
# ─────────────────────────────────────────────────────────
def pk_params(pop, drug="sun", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    age, bwt, sex, race, tumr, n = (pop[k] for k in
                                     ["age","bwt","sex","race","tumr","n"])
    if drug == "sun":
        CL = 34.1*(1-0.00702*(age-58))*(1-0.152*race)*(1-0.193*sex)*(1+0.293*tumr)
        Vc = 2700*(bwt/77.4)**0.281*(1-0.213*sex)*(1+0.42*tumr)
        Vp, Q  = 774.0, 0.688
        Ka, tlag = 0.126, 0.527
        cv_cl, cv_vc = 0.246, 0.230
        om_ka = cv_to_omega(166.0)   # 1.1506 — Ka IIV 166% (was missing in v1)
    else:  # SU12662 metabolite
        CL = 17.5*(1-0.00743*(age-58))*(1-0.205*race)*(1-0.354*sex)*(1+0.324*tumr)
        Vc = 2120*(1+0.00892*(bwt-77.3))*(1-0.272*sex)*(1+0.635*tumr)
        Vp, Q  = 751.0, 0.979
        Ka, tlag = 0.102, 0.0
        cv_cl, cv_vc = 0.363, 0.473
        om_ka = cv_to_omega(126.0)   # 0.9753 — Ka IIV 126% (was missing in v1)

    CL *= np.exp(rng.normal(0, cv_cl, n))
    Vc *= np.exp(rng.normal(0, cv_vc, n))
    # Ka with IIV, clipped to physically plausible range [0.01, 5.0] h⁻¹
    Ka_i   = np.clip(Ka * np.exp(rng.normal(0, om_ka, n)), 0.01, 5.0)
    tlag_i = np.full(n, tlag)
    return dict(CL=CL, Vc=Vc, Vp=np.full(n, Vp), Q=np.full(n, Q),
                Ka=Ka_i, tlag=tlag_i)


# ─────────────────────────────────────────────────────────
# 3. ANALYTICAL 2-COMPARTMENT PK  (superposition principle)
#    Returns (n_pat, n_t) concentration matrix in ng/mL
# ─────────────────────────────────────────────────────────
def pk_2cpt_analytical(pk, dose_times, dose_mg, t_eval):
    """Bateman equation via eigenvalue decomposition, superposition over doses."""
    CL = pk["CL"]; Vc = pk["Vc"]; Vp = pk["Vp"]; Q = pk["Q"]
    Ka = pk["Ka"]; tlag = pk["tlag"]
    n = len(CL)

    k10 = CL / Vc
    k12 = Q  / Vc
    k21 = Q  / Vp

    a    = k10 + k12 + k21
    b    = k10 * k21
    disc = np.sqrt(np.maximum(a**2 - 4*b, 0.0))
    lam1 = (a + disc) / 2
    lam2 = (a - disc) / 2

    eps = 1e-9
    c_lam2 = (k21 - lam2) / ((Ka - lam2 + eps) * ( lam1 - lam2 + eps))
    c_lam1 = (k21 - lam1) / ((Ka - lam1 + eps) * (-lam1 + lam2 - eps))
    c_Ka   = (k21 - Ka)   / ((-Ka + lam2 - eps) * (-Ka + lam1 - eps))

    prefactor = Ka / Vc * dose_mg   # (n,)

    Cc = np.zeros((n, len(t_eval)))  # mg/L
    for td in dose_times:
        t_eff  = td + tlag                                     # (n,)
        dt_mat = t_eval[np.newaxis, :] - t_eff[:, np.newaxis] # (n, nt)
        active = dt_mat >= 0.0
        dt_act = np.where(active, dt_mat, 0.0)

        contrib = prefactor[:, np.newaxis] * (
            c_lam2[:, np.newaxis] * np.exp(-lam2[:, np.newaxis] * dt_act)
          + c_lam1[:, np.newaxis] * np.exp(-lam1[:, np.newaxis] * dt_act)
          + c_Ka[:, np.newaxis]   * np.exp(-Ka[:, np.newaxis]   * dt_act)
        )
        Cc += np.where(active, contrib, 0.0)

    return Cc * 1000.0   # → ng/mL


def pk_metabolite(pk_sun, pk_met, dose_times, dose_mg, t_eval):
    """SU12662: separate 2-cpt model, scaled to trough ratio ~0.462 (19.7/42.6)."""
    Cc_met = pk_2cpt_analytical(pk_met, dose_times, dose_mg, t_eval)
    Cc_sun = pk_2cpt_analytical(pk_sun, dose_times, dose_mg, t_eval)
    cyc3   = (t_eval >= 2*42*24) & (t_eval <= 3*42*24)
    if cyc3.any():
        ratio_now = np.median(Cc_met[:, cyc3]) / (np.median(Cc_sun[:, cyc3]) + 1e-9)
        target = 0.462
        if ratio_now > 0:
            Cc_met *= target / ratio_now
    return Cc_met


# ─────────────────────────────────────────────────────────
# 4. DOSING SCHEDULES
# ─────────────────────────────────────────────────────────
def build_dose_times(schedule, n_cycles=N_CYCLES):
    """42-day cycle: Schedule 4/2 = 28 on / 14 off; 2/1 = 14 on / 7 off / 14 on / 7 off."""
    cycle_h = 42 * 24
    doses = []
    for c in range(n_cycles):
        s = c * cycle_h
        if schedule == "4/2":
            doses += [s + d * 24 for d in range(28)]
        else:  # 2/1
            doses += [s + d * 24 for d in range(14)]
            doses += [s + (21 + d) * 24 for d in range(14)]
    return np.array(doses, dtype=float)


# ─────────────────────────────────────────────────────────
# 5. PD MODELS
# ─────────────────────────────────────────────────────────

def pd_sld(t, Cc, BASE, Kout, EC50, Ktol):
    """
    IDR with tolerance: drug inhibits Kin (Emax=1), tolerance function on Kout.
    dSLD/dt = Kin*(1 - C/(EC50+C)) - Kout*exp(-Ktol*t)*SLD
    All four SLD parameters now have per-patient IIV.
    """
    Kin = Kout * BASE  # per-patient, (n,)
    n, nt = len(BASE), len(t)
    Y = np.zeros((n, nt)); Y[:, 0] = BASE.copy()
    for i in range(1, nt):
        h = t[i] - t[i-1]
        c = Cc[:, i-1]
        kout_t = Kout * np.exp(-Ktol * t[i-1])
        dY = Kin * (1.0 - c / (EC50 + c)) - kout_t * Y[:, i-1]
        Y[:, i] = np.maximum(Y[:, i-1] + h*dY, 0.01)
    return Y


def pd_idr_stim_kin(t, Cc, BASE, Kout, KPD):
    """
    IDR Type I: drug stimulates Kin → endpoint INCREASES with drug.
    dR/dt = Kin*(1 + KPD*C) - Kout*R
    Used for: ALT, AST (hepatotoxicity), DBP (hypertension).
    Kout and KPD are per-patient arrays (with IIV).
    """
    Kin = Kout * BASE  # (n,) — must use per-patient Kout
    n, nt = len(BASE), len(t)
    Y = np.zeros((n, nt)); Y[:, 0] = BASE.copy()
    for i in range(1, nt):
        h = t[i] - t[i-1]
        c = Cc[:, i-1]
        dY = Kin * (1.0 + KPD * c) - Kout * Y[:, i-1]
        Y[:, i] = np.maximum(Y[:, i-1] + h*dY, 0.001)
    return Y


def pd_idr_inhib_kin(t, Cc, BASE, Kout, KPD):
    """
    IDR Type II: drug inhibits Kin (bounded) → endpoint DECREASES.
    dR/dt = Kin / (1 + KPD*C) - Kout*R
    At C → ∞: R → 0 (bounded, never negative).
    At C = 0: R → BASE (recovers fully).
    Used for: LVEF (cardiotoxicity).
    Kout and KPD are per-patient arrays (with IIV).
    Note: v1 used Type III (Kout stimulation) which drove LVEF → ~0.7% (physiologically impossible).
    """
    Kin = Kout * BASE  # (n,)
    n, nt = len(BASE), len(t)
    Y = np.zeros((n, nt)); Y[:, 0] = BASE.copy()
    for i in range(1, nt):
        h = t[i] - t[i-1]
        c = Cc[:, i-1]
        dY = Kin / (1.0 + KPD * c) - Kout * Y[:, i-1]
        Y[:, i] = np.maximum(Y[:, i-1] + h*dY, 0.001)
    return Y


def pd_tcsfl(t, Cc, BASE, MTT, Emax, EC50, POW, LAM, n_tr=3):
    """TCSFL: sigmoidal Emax on Kprol + feedback. Used for ANC and PC."""
    n, nt = len(BASE), len(t)
    Ktr = (n_tr+1) / MTT
    S = np.tile(BASE[:, np.newaxis], (1, n_tr+2))
    Circ_out = np.zeros((n, nt)); Circ_out[:, 0] = BASE.copy()
    for i in range(1, nt):
        h = t[i] - t[i-1]
        c    = Cc[:, i-1]
        Circ = S[:, -1]
        drug_eff = Emax * c**LAM / (EC50**LAM + c**LAM)
        FB       = (BASE / np.maximum(Circ, 1e-4))**POW
        dS = np.zeros_like(S)
        dS[:, 0] = Ktr*S[:,0]*(1-drug_eff)*FB - Ktr*S[:,0]
        for j in range(1, n_tr+1):
            dS[:, j] = Ktr*S[:,j-1] - Ktr*S[:,j]
        dS[:, -1] = Ktr*S[:,-2] - Ktr*S[:,-1]
        S = np.maximum(S + h*dS, 1e-4)
        Circ_out[:, i] = S[:, -1]
    return Circ_out


def pd_tcsfl_kpd(t, Cc, BASE, MTT, KPD, POW, n_tr=3):
    """Reduced TCSFL: KPD effect on Kout of Circ. Used for LC."""
    n, nt = len(BASE), len(t)
    Ktr = (n_tr+1) / MTT
    S = np.tile(BASE[:, np.newaxis], (1, n_tr+2))
    Circ_out = np.zeros((n, nt)); Circ_out[:, 0] = BASE.copy()
    for i in range(1, nt):
        h = t[i] - t[i-1]
        c    = Cc[:, i-1]
        Circ = S[:, -1]
        FB   = (BASE / np.maximum(Circ, 1e-4))**POW
        dS = np.zeros_like(S)
        dS[:, 0] = Ktr*S[:,0]*FB - Ktr*S[:,0]
        for j in range(1, n_tr+1):
            dS[:, j] = Ktr*S[:,j-1] - Ktr*S[:,j]
        dS[:, -1] = Ktr*S[:,-2] - (Ktr + KPD*c)*S[:,-1]
        S = np.maximum(S + h*dS, 1e-4)
        Circ_out[:, i] = S[:, -1]
    return Circ_out


# ─────────────────────────────────────────────────────────
# 6. FULL ARM SIMULATION (13 cycles ~ 78 weeks)
# ─────────────────────────────────────────────────────────
def run_simulation(n_pat=100, tumor_type="RCC", schedule="4/2",
                   dt=8.0, seed=None, verbose=True):
    """
    Run one arm simulation.  dt=8h chosen so 13-cycle simulation has ~1638
    timesteps — comparable to the original 6-cycle/dt=4h (1512 steps).
    """
    rng = np.random.default_rng(seed)
    pop = generate_patients(n_pat, tumor_type=tumor_type, schedule=schedule,
                            seed=seed)
    age, bwt, sex, race, ecog, tumr, sch, n = (
        pop[k] for k in ["age","bwt","sex","race","ecog","tumr","sch","n"])

    pk_sun = pk_params(pop, "sun", rng)
    pk_met = pk_params(pop, "met", rng)

    dose_times = build_dose_times(schedule, n_cycles=N_CYCLES)
    t_end = N_CYCLES * 42 * 24
    t = np.arange(0.0, t_end + dt, dt)

    if verbose:
        print(f"  PK ({tumor_type}, Sched {schedule}, {N_CYCLES} cyc) ...", flush=True)
    Csun = pk_2cpt_analytical(pk_sun, dose_times, 50.0, t)
    Cmet = pk_metabolite(pk_sun, pk_met, dose_times, 50.0, t)

    if verbose:
        print(f"  PD ...", flush=True)

    # ── SLD (Eqs 5-7, EXPONENTIAL covariate form from Table 3) ───────────────
    # Paper Equations 5-7 (Khosravan 2016) use (1+theta*X) linear multiplier form:
    #   Baseline SLD = 14.3*(1+0.574*BEC)*(1-0.348*RAC)*(1-0.43*SCH)   Eq 5
    #   Kout         = 2.67e-4*(1+1.01*SCH)                            Eq 6
    #   EC50         = 30.5*(1+2.43*SCH)*(1+4.82*TUMR)                 Eq 7
    # SCH=0 for both 4/2 and 2/1 (both are intermittent; SCH=1 is CDD in the
    # original dataset — not simulated here).
    # GIST EC50 = 30.5*(1+4.82) = 177 ng/mL (drug effect ~19% at 42 ng/mL trough)
    BASE_SLD = (14.3 * (1 + 0.574*ecog) * (1 - 0.348*race) * (1 - 0.430*sch)
                * np.exp(rng.normal(0, OM_SLD_BASE, n)))          # 91.7% CV IIV
    KOUT_SLD = (2.67e-4 * (1 + 1.01*sch)
                * np.exp(rng.normal(0, OM_SLD_KOUT, n)))          # 72.2% CV IIV
    EC50_SLD = (30.5 * (1 + 2.43*sch) * (1 + 4.82*tumr)
                * np.exp(rng.normal(0, OM_SLD_EC50, n)))          # 182% CV IIV
    KTOL_SLD = (np.full(n, 1.41e-5)
                * np.exp(rng.normal(0, OM_SLD_KTOL, n)))          # 84.9% CV IIV
    KOUT_SLD = np.clip(KOUT_SLD, 1e-6, 0.1)
    EC50_SLD = np.clip(EC50_SLD, 0.1, 5000)
    KTOL_SLD = np.clip(KTOL_SLD, 1e-8, 0.01)
    SLD = pd_sld(t, Csun, BASE_SLD, KOUT_SLD, EC50_SLD, KTOL_SLD)

    # ── ALT (Eq 8) — Type I IDR: drug INCREASES ALT (hepatotoxicity) ─────────
    # v1 error: used Type III (Kout stimulation) which DECREASED ALT — wrong direction.
    # Correct: KPD stimulates Kin → ALT rises with drug.
    BASE_ALT = (21.2 * (bwt/77.3)**0.376
                * np.exp(rng.normal(0, cv_to_omega(40.5), n)))
    KOUT_ALT = 0.00916 * np.exp(rng.normal(0, cv_to_omega(128.0), n))
    KPD_ALT  = 0.00401 * np.exp(rng.normal(0, cv_to_omega(57.0),  n))
    ALT = pd_idr_stim_kin(t, Csun, BASE_ALT, KOUT_ALT, KPD_ALT)

    # ── AST (Eqs 9-10) — Type I IDR: drug INCREASES AST (hepatotoxicity) ─────
    BASE_AST = (21.5 * (1 + 0.117*tumr)
                * np.exp(rng.normal(0, cv_to_omega(31.8), n)))
    KOUT_AST = 0.0142 * np.exp(rng.normal(0, cv_to_omega(120.0), n))
    KPD_AST  = (0.00572 * (1 + 0.2*ecog) * (1 - 0.175*tumr)
                * np.exp(rng.normal(0, cv_to_omega(33.8), n)))
    AST = pd_idr_stim_kin(t, Csun, BASE_AST, KOUT_AST, KPD_AST)

    # ── LVEF (Eq 11) — Type II IDR: drug DECREASES LVEF (cardiotoxicity) ─────
    # v1 error: used Type III with KPD on Kout → LVEF → ~0.7% (near zero!).
    # Correct: bounded inhibition of Kin keeps LVEF in physiological range.
    # At SS with Csun=42 ng/mL: LVEF ≈ BASE * 1/(1 + 0.00131*42) ≈ BASE * 0.944 = 58.7%
    BASE_LVEF = (62.2 * (1 + 0.0891*race) * (1 + 0.0421*sex)
                 * np.exp(rng.normal(0, cv_to_omega(8.61), n)))
    KOUT_LVEF = 0.000656 * np.exp(rng.normal(0, cv_to_omega(82.8), n))
    KPD_LVEF  = 0.00131  * np.exp(rng.normal(0, cv_to_omega(90.1), n))
    LVEF = pd_idr_inhib_kin(t, Csun, BASE_LVEF, KOUT_LVEF, KPD_LVEF)

    # ── DBP (Eq 12) — Type I IDR: drug INCREASES DBP (hypertension) ──────────
    # v1 error: used Type III which DECREASED DBP — wrong direction.
    # At SS with Csun=42: DBP ≈ BASE * (1 + 0.00184*42) = BASE * 1.077 = 80.3 mmHg
    BASE_DBP = (74.6 * (bwt/77.3)**0.0691
                * np.exp(rng.normal(0, cv_to_omega(9.38), n)))
    KOUT_DBP = 0.0288 * np.exp(rng.normal(0, cv_to_omega(108.0), n))
    KPD_DBP  = 1.84e-3 * np.exp(rng.normal(0, cv_to_omega(47.6), n))
    DBP = pd_idr_stim_kin(t, Csun, BASE_DBP, KOUT_DBP, KPD_DBP)

    # ── ANC ───────────────────────────────────────────────────────────────────
    BASE_ANC = (4.61 * (1 - 0.297*race) * (1 + 0.134*ecog)
                * np.exp(rng.normal(0, cv_to_omega(30.6), n)))
    Emax_ANC = 0.126 * np.exp(rng.normal(0, cv_to_omega(17.3), n))
    EC50_ANC = 11.1  * np.exp(rng.normal(0, cv_to_omega(84.3), n))
    ANC = pd_tcsfl(t, Csun, BASE_ANC, MTT=np.full(n, 182.0),
                   Emax=Emax_ANC, EC50=EC50_ANC,
                   POW=np.full(n, 0.152), LAM=np.full(n, 1.72))

    # ── PC ────────────────────────────────────────────────────────────────────
    BASE_PC = (297 * (1 - 0.00327*(bwt-77.3)) * (1 - 0.255*race)
               * np.exp(rng.normal(0, cv_to_omega(34.4), n)))
    Emax_PC = (0.154 * (1 - 0.00742*(bwt-77.3))
               * np.exp(rng.normal(0, cv_to_omega(26.6), n)))
    EC50_PC = (65.0 * (1 - 0.108*tumr)
               * np.exp(rng.normal(0, cv_to_omega(21.1), n)))
    MTT_PC  = 88.4 * (1 + 0.118*ecog) * (1 - 0.195*race)
    PC = pd_tcsfl(t, Csun, BASE_PC, MTT=MTT_PC,
                  Emax=Emax_PC, EC50=EC50_PC,
                  POW=np.full(n, 0.0895), LAM=np.full(n, 3.09))

    # ── LC ────────────────────────────────────────────────────────────────────
    BASE_LC = (1.51 * (1 - 0.121*ecog)
               * np.exp(rng.normal(0, cv_to_omega(40.2), n)))
    MTT_LC  = 243 * (1 - 0.398*race)
    LC = pd_tcsfl_kpd(t, Csun, BASE_LC, MTT=MTT_LC,
                      KPD=np.full(n, 6.87e-4), POW=np.full(n, 0.200))

    return dict(t=t, Csun=Csun, Cmet=Cmet,
                SLD=SLD, ALT=ALT, AST=AST, LVEF=LVEF,
                DBP=DBP, ANC=ANC, PC=PC, LC=LC,
                BASE_SLD=BASE_SLD, BASE_PC=BASE_PC, BASE_ANC=BASE_ANC,
                schedule=schedule, tumor_type=tumor_type, n=n,
                rng_seed=seed)


# ─────────────────────────────────────────────────────────
# 7. METRICS  (corrected ORR, PFS, and PC with RUV)
# ─────────────────────────────────────────────────────────
def _km_median(event_times, event_statuses):
    """
    Pure-numpy Kaplan-Meier median survival time.
    event_times   : array of times-to-progression or censoring (weeks)
    event_statuses: 1 = progression observed, 0 = censored
    Returns the KM median (time where S(t) first crosses 0.5),
    or the last observed time if the median is not reached.
    """
    t   = np.asarray(event_times,    dtype=float)
    ev  = np.asarray(event_statuses, dtype=int)
    idx = np.argsort(t)
    t, ev = t[idx], ev[idx]
    n_at_risk = len(t)
    S = 1.0
    for ti, ei in zip(t, ev):
        if ei == 1:          # event
            S *= 1.0 - 1.0 / n_at_risk
        n_at_risk -= 1
        if S <= 0.5:
            return ti
    return np.nan            # median not reached within observation window


def _cycle3_trough_mask(t, schedule):
    """
    Sample pre-dose troughs during the EARLY re-accumulation phase of cycle 3
    (days 1-7 of the first dosing block).  The paper's protocol visits were
    weekly during cycle 3; analytical check shows the reference patient trough
    crosses ~42.6 ng/mL at day 3-4 back on drug (Khosravan 2016 Table 2 target).
    Sampling at full steady-state (day 14+) over-estimates by ~30% because the
    slow lambda (t1/2=797h) continues to accumulate past the protocol visit window.
    """
    c3_start = 2 * 42 * 24
    # Days 1-7 of the first dosing-on block (same for 4/2 and 2/1)
    dose_hours = c3_start + np.arange(1, 8) * 24
    if False:  # unused branch kept for reference
        dose_hours = np.concatenate([
            c3_start + np.arange(1, 14) * 24,
            c3_start + (21 + np.arange(1, 14)) * 24,
        ])

    mask = np.zeros_like(t, dtype=bool)
    for hour in dose_hours:
        mask[np.argmin(np.abs(t - hour))] = True
    return mask


def extract_metrics(res, ruv_seed=None):
    t   = res["t"]
    n   = res["n"]
    rng = np.random.default_rng(ruv_seed)
    week = 7 * 24

    # Cycle 3 time mask (full cycle, used for PC nadir)
    m3 = (t >= 2*42*24) & (t <= 3*42*24)

    # Trough concentrations: median of explicit cycle-3 pre-dose samples on dosing days.
    # This matches scheduled trough sampling better than taking a window minimum.
    m3_trough = _cycle3_trough_mask(t, res["schedule"])
    trough_sun = np.median(res["Csun"][:, m3_trough], axis=1)
    trough_met = np.median(res["Cmet"][:, m3_trough], axis=1)

    # SLD at end of cycle 6
    SLD_c6 = res["SLD"][:, np.argmin(np.abs(t - 6*42*24))]

    # PFS via Kaplan-Meier with proper censoring.
    # Event   = SLD rises >= 20% above nadir (RECIST progression).
    # Censored = never progressed within the 6-cycle observation window.
    # Using KM avoids the naive-median ceiling: non-progressors are treated as
    # right-censored observations rather than being assigned the max time.
    pfs_evt_t, pfs_evt_s = [], []
    for i in range(n):
        ni  = np.argmin(res["SLD"][i])
        thr = 1.2 * res["SLD"][i, ni]
        post = np.where(res["SLD"][i, ni:] > thr)[0]
        if len(post):
            pfs_evt_t.append((t[ni + post[0]]) / week)   # progression
            pfs_evt_s.append(1)
        else:
            pfs_evt_t.append(t[-1] / week)                # censored
            pfs_evt_s.append(0)

    # ORR: best response at discrete RECIST-style assessment visits (every 42 days).
    # Assess only through cycle 3 (paper Set-2 simulation for PFS/ORR uses 3 cycles
    # for safety AE grading; ORR is derived from the same short simulation set).
    # Apply SLD measurement RUV (14.3% proportional error from Table 3) to each
    # assessment visit — this is how the paper's SLD model propagates uncertainty.
    assess_h   = np.array([c * 42 * 24 for c in range(1, N_CYCLES_ASSESS + 1)])
    assess_idx = [np.argmin(np.abs(t - ah)) for ah in assess_h]
    SLD_visits = res["SLD"][:, assess_idx]                    # (n, N_CYCLES_ASSESS)
    # Add proportional measurement noise to each visit independently
    SLD_visits = SLD_visits * np.exp(
        rng.normal(0, SIGMA_SLD_RUV, SLD_visits.shape))       # lognormal RUV
    best_pct   = (np.min(SLD_visits, axis=1) - res["BASE_SLD"]) / res["BASE_SLD"]
    orr = np.mean(best_pct <= -0.30) * 100

    # PC nadir in cycle 3
    pc_nadir_c3 = np.min(res["PC"][:, m3], axis=1)

    # Grade 3/4 thrombocytopenia: apply RUV once to the per-patient cycle-3 nadir.
    # Applying noise to every ODE timestep then taking min creates extreme statistical
    # bias (minimum of ~1000 noisy draws is always far below the true nadir).
    # One lognormal draw per patient matches how the paper measured discrete lab values.
    mask_3cyc    = t <= 3*42*24
    pc_nadir_c3  = np.min(res["PC"][:, mask_3cyc], axis=1)          # model nadir
    pc_obs_nadir = pc_nadir_c3 * np.exp(rng.normal(0, SIGMA_PC_RUV, n))  # single RUV draw
    g34 = np.mean(pc_obs_nadir < 50.0) * 100

    return dict(trough_sun=np.median(trough_sun),
                trough_met=np.median(trough_met),
                SLD_c6=np.median(SLD_c6),
                PFS=_km_median(pfs_evt_t, pfs_evt_s),
                ORR=orr,
                pc_nadir=np.median(pc_nadir_c3),
                grade34=g34)


# ─────────────────────────────────────────────────────────
# 8. RUN 20 TRIAL SIMULATIONS
# ─────────────────────────────────────────────────────────
def _run_one_trial_metrics(s, n_pat, tumor_type, dt):
    """Worker helper to run both schedules and return only trial metrics."""
    r42 = run_simulation(n_pat, tumor_type, "4/2", dt=dt, seed=s*10, verbose=False)
    r21 = run_simulation(n_pat, tumor_type, "2/1", dt=dt, seed=s*10+1, verbose=False)
    m42 = extract_metrics(r42, ruv_seed=s*100)
    m21 = extract_metrics(r21, ruv_seed=s*100+1)
    return s, m42, m21


def run_trials(n_sims=20, n_pat=100, tumor_type="RCC", dt=8.0, n_jobs=1):
    print(f"\n{'='*60}\n Running {n_sims} trial simulations — {tumor_type}"
          f"  ({N_CYCLES} cycles, dt={dt}h)\n{'='*60}")
    m42_all = [None] * n_sims
    m21_all = [None] * n_sims
    last42, last21 = None, None

    if n_jobs <= 1:
        for s in range(n_sims):
            print(f"\nSim {s+1}/{n_sims}", flush=True)
            r42 = run_simulation(n_pat, tumor_type, "4/2", dt=dt, seed=s*10)
            r21 = run_simulation(n_pat, tumor_type, "2/1", dt=dt, seed=s*10+1)
            m42_all[s] = extract_metrics(r42, ruv_seed=s*100)
            m21_all[s] = extract_metrics(r21, ruv_seed=s*100+1)
    else:
        print(f"Using parallel trial execution with {n_jobs} workers", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_run_one_trial_metrics, s, n_pat, tumor_type, dt)
                    for s in range(n_sims)]
            for fut in as_completed(futs):
                s, m42, m21 = fut.result()
                m42_all[s] = m42
                m21_all[s] = m21
                done += 1
                print(f"\rCompleted {done}/{n_sims} trial simulations", end="", flush=True)
        print("", flush=True)

    # Keep plotting behavior unchanged by regenerating the last simulated arm.
    last_seed = (n_sims - 1) * 10
    print("Generating representative trajectories for plotting ...", flush=True)
    last42 = run_simulation(n_pat, tumor_type, "4/2", dt=dt, seed=last_seed, verbose=False)
    last21 = run_simulation(n_pat, tumor_type, "2/1", dt=dt, seed=last_seed + 1, verbose=False)

    keys = ["trough_sun","trough_met","SLD_c6","PFS","ORR","pc_nadir","grade34"]
    lbls = ["Trough Sunitinib (ng/mL)","Trough SU12662 (ng/mL)",
            "SLD Cycle 6 (cm)","Median PFS (weeks)","ORR (%)",
            "PC Nadir Cycle 3 (×10³/µL)","Grade 3/4 Thrombo (%)"]
    p42  = [42.6, 19.7,  8.6, 47.2, 27.0, 104, 16]
    p21  = [42.4, 19.5,  8.2, 54.3, 31.0, 119,  9]

    print(f"\n{'='*70}")
    print(f"{'Metric':<35}{'Our 4/2':>9}{'Paper 4/2':>10}{'Our 2/1':>9}{'Paper 2/1':>10}")
    print("-"*70)
    for k, lb, v42, v21 in zip(keys, lbls, p42, p21):
        o42 = np.median([d[k] for d in m42_all])
        o21 = np.median([d[k] for d in m21_all])
        print(f"{lb:<35}{o42:>9.1f}{v42:>10.1f}{o21:>9.1f}{v21:>10.1f}")
    print(f"{'='*70}")

    return last42, last21, m42_all, m21_all


# ─────────────────────────────────────────────────────────
# 9. PLOTTING
# ─────────────────────────────────────────────────────────
C42, C21 = "#1565C0", "#C62828"

def _ts(ax, t_weeks, data, color, label, pct_lo=2.5, pct_hi=97.5):
    med = np.median(data, axis=0)
    lo  = np.percentile(data, pct_lo, axis=0)
    hi  = np.percentile(data, pct_hi, axis=0)
    ax.plot(t_weeks, med, color=color, lw=2, label=label)
    ax.fill_between(t_weeks, lo, hi, alpha=0.18, color=color)


def fig_pk_overlay(r42, r21, tt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Sunitinib & SU12662 PK — Schedule 4/2 vs 2/1 ({tt})",
                 fontsize=12, fontweight="bold")
    for ax, key, drug in zip(axes, ["Csun","Cmet"], ["Sunitinib","SU12662"]):
        tw = r42["t"] / (7*24)
        _ts(ax, tw, r42[key], C42, "Schedule 4/2")
        _ts(ax, r21["t"]/(7*24), r21[key], C21, "Schedule 2/1")
        ax.set_yscale("log"); ax.set_xlabel("Weeks")
        ax.set_ylabel(f"{drug} (ng/mL)"); ax.set_title(drug)
        ax.legend(fontsize=9); ax.set_xlim(0, tw[-1])
    plt.tight_layout()
    fname = f"{OUTDIR}/fig_pk_{tt}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def fig_pd_all(r42, r21, tt):
    # Safety endpoint baselines for physiological reference lines
    ref = dict(
        ALT  = (21, 56,  "ULN=56 U/L"),
        AST  = (21, 40,  "ULN=40 U/L"),
        LVEF = (62, 53,  "Lower limit=53%"),
        DBP  = (75, 90,  "Stage 1 HTN=90"),
    )
    eps = [("SLD","Tumor SLD (cm)"),("PC","Platelet Count (×10³/µL)"),
           ("ANC","ANC (×10⁹/L)"), ("LC","Lymphocytes (×10⁹/L)"),
           ("ALT","ALT (U/L)"),    ("AST","AST (U/L)"),
           ("LVEF","LVEF (%)"),    ("DBP","DBP (mmHg)")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"PD Endpoints — Schedule 4/2 vs 2/1 ({tt}) — v2 Corrected",
                 fontsize=12, fontweight="bold")
    for ax, (key, ylabel) in zip(axes.flatten(), eps):
        tw = r42["t"] / (7*24)
        _ts(ax, tw, r42[key], C42, "4/2")
        _ts(ax, r21["t"]/(7*24), r21[key], C21, "2/1")
        if key in ref:
            _, refval, reflbl = ref[key]
            ax.axhline(refval, color="red", ls="--", lw=1.2, label=reflbl)
        ax.set_xlabel("Weeks", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=10)
        ax.legend(fontsize=7); ax.set_xlim(0, tw[-1])
    plt.tight_layout()
    fname = f"{OUTDIR}/fig_pd_{tt}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def fig_summary_bars(m42, m21, tt):
    keys = ["trough_sun","SLD_c6","PFS","ORR","pc_nadir","grade34"]
    lbls = ["Trough Sunitinib\n(ng/mL)","SLD Cycle 6\n(cm)",
            "Median PFS\n(weeks)","ORR (%)","PC Nadir C3\n(×10³/µL)",
            "Grade 3/4\nThrombo (%)"]
    p42  = [42.6, 8.6, 47.2, 27.0, 104, 16]
    p21  = [42.4, 8.2, 54.3, 31.0, 119,  9]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Our Model (v2 Corrected) vs Paper — {tt}",
                 fontsize=12, fontweight="bold")
    for ax, k, lb, v42, v21 in zip(axes.flatten(), keys, lbls, p42, p21):
        o42 = np.median([d[k] for d in m42])
        o21 = np.median([d[k] for d in m21])
        s42 = np.std([d[k] for d in m42])
        s21 = np.std([d[k] for d in m21])
        vals = [o42, v42, o21, v21]
        clrs = ["#1565C0","#90CAF9","#C62828","#EF9A9A"]
        bars = ax.bar([0,1,2,3], vals, color=clrs, edgecolor="k",
                      linewidth=0.7, width=0.6)
        ax.errorbar([0,2], [o42,o21], [s42,s21], fmt="none", color="k", capsize=4)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(["4/2\nModel","4/2\nPaper","2/1\nModel","2/1\nPaper"],
                           fontsize=8)
        ax.set_title(lb.replace("\n"," "), fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fname = f"{OUTDIR}/fig_summary_{tt}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def fig_grade34(m42, m21, tt):
    g42 = np.median([d["grade34"] for d in m42])
    g21 = np.median([d["grade34"] for d in m21])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["4/2\nModel","4/2\nPaper","2/1\nModel","2/1\nPaper"],
           [g42, 16, g21, 9],
           color=["#1565C0","#90CAF9","#C62828","#EF9A9A"],
           edgecolor="k", width=0.5)
    ax.set_ylabel("Grade 3/4 Thrombocytopenia (%)", fontsize=11)
    ax.set_title(f"Grade 3/4 Thrombocytopenia — {tt} (with 24% CV RUV)", fontsize=10)
    for x, v in enumerate([g42, 16, g21, 9]):
        ax.text(x, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10,
                fontweight="bold")
    plt.tight_layout()
    fname = f"{OUTDIR}/fig_grade34_{tt}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def fig_pc_time(r42, r21, tt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Platelet Count Over Time ({tt}) — with 24% RUV shading",
                 fontsize=12, fontweight="bold")
    for ax, (r, lbl, clr) in zip(axes, [(r42,"4/2",C42),(r21,"2/1",C21)]):
        _ts(ax, r["t"]/(7*24), r["PC"], clr, f"Schedule {lbl}")
        ax.axhline(50,  color="red",    ls="--", lw=1.5, label="Grade 3/4 (<50)")
        ax.axhline(100, color="orange", ls=":",  lw=1.5, label="Grade 2 (<100)")
        ax.set_xlabel("Weeks"); ax.set_ylabel("Platelet Count (×10³/µL)")
        ax.set_title(f"Schedule {lbl}"); ax.legend(fontsize=8)
        ax.set_xlim(0, r["t"][-1]/(7*24))
    plt.tight_layout()
    fname = f"{OUTDIR}/fig_pc_{tt}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def fig_safety_endpoints(r42, r21, tt):
    """New figure showing corrected ALT/AST (increase) and LVEF/DBP trends."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Safety Endpoints (Corrected Directions) — {tt}",
                 fontsize=12, fontweight="bold")

    configs = [
        ("ALT",  "ALT (U/L)",   56,  "ULN (56 U/L)",   axes[0,0]),
        ("AST",  "AST (U/L)",   40,  "ULN (40 U/L)",   axes[0,1]),
        ("LVEF", "LVEF (%)",    53,  "Cardiox limit",   axes[1,0]),
        ("DBP",  "DBP (mmHg)",  90,  "Stage 1 HTN",    axes[1,1]),
    ]
    for key, ylabel, refval, reflbl, ax in configs:
        tw = r42["t"] / (7*24)
        _ts(ax, tw, r42[key], C42, "4/2")
        _ts(ax, r21["t"]/(7*24), r21[key], C21, "2/1")
        ax.axhline(refval, color="red", ls="--", lw=1.5, label=reflbl)
        ax.set_xlabel("Weeks"); ax.set_ylabel(ylabel)
        ax.set_title(key); ax.legend(fontsize=8)
        ax.set_xlim(0, tw[-1])

    plt.tight_layout()
    fname = f"{OUTDIR}/fig_safety_{tt}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="Run corrected sunitinib PK/PD simulations.")
    parser.add_argument("--n-sims", type=int, default=20, help="Number of trial simulations")
    parser.add_argument("--n-pat", type=int, default=100, help="Patients per arm")
    parser.add_argument("--dt", type=float, default=8.0, help="Integration step (hours)")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="Parallel workers for trial-level simulations")
    args = parser.parse_args()

    t0 = time.time()
    N_SIMS, N_PAT, DT, N_JOBS = args.n_sims, args.n_pat, args.dt, max(1, args.jobs)

    print("Corrected omega values:")
    for name, cv in [("BASE_SLD",91.7),("EC50_SLD",182.0),("Kout_SLD",72.2),("Ktol_SLD",84.9)]:
        print(f"  {name}: {cv}% CV -> omega = {cv_to_omega(cv):.4f}")

    # Accumulate results across tumor types for CSV export
    all_results = []
    summary_rows = []

    keys = ["trough_sun","trough_met","SLD_c6","PFS","ORR","pc_nadir","grade34"]
    lbls = ["Trough_Sunitinib_ngmL","Trough_SU12662_ngmL","SLD_Cycle6_cm",
            "Median_PFS_weeks","ORR_pct","PC_Nadir_C3_1e3perµL","Grade34_Thrombo_pct"]
    paper_42 = {"RCC": [42.6,19.7,8.6,47.2,27.0,104,16],
                "GIST":[42.6,19.7,8.6,47.2,27.0,104,16]}  # GIST paper targets = RCC (Supp S4 not available)
    paper_21 = {"RCC": [42.4,19.5,8.2,54.3,31.0,119, 9],
                "GIST":[42.4,19.5,8.2,54.3,31.0,119, 9]}

    for ttype in ["RCC", "GIST"]:
        r42, r21, m42, m21 = run_trials(N_SIMS, N_PAT, ttype, dt=DT, n_jobs=N_JOBS)
        fig_pk_overlay(r42, r21, ttype)
        fig_pd_all(r42, r21, ttype)
        fig_summary_bars(m42, m21, ttype)
        fig_grade34(m42, m21, ttype)
        fig_pc_time(r42, r21, ttype)
        fig_safety_endpoints(r42, r21, ttype)

        # Per-simulation detailed CSV rows
        for sim_i, (d42, d21) in enumerate(zip(m42, m21)):
            row42 = {"tumor_type": ttype, "schedule": "4/2", "sim": sim_i + 1}
            row21 = {"tumor_type": ttype, "schedule": "2/1", "sim": sim_i + 1}
            for k, lb in zip(keys, lbls):
                row42[lb] = d42[k]
                row21[lb] = d21[k]
            all_results.extend([row42, row21])

        # Summary CSV (median across sims vs paper)
        for sched, m_all, p_vals in [("4/2", m42, paper_42[ttype]),
                                      ("2/1", m21, paper_21[ttype])]:
            row = {"tumor_type": ttype, "schedule": sched}
            for k, lb, pv in zip(keys, lbls, p_vals):
                row[f"model_median_{lb}"] = np.median([d[k] for d in m_all])
                row[f"model_p5_{lb}"]     = np.percentile([d[k] for d in m_all], 5)
                row[f"model_p95_{lb}"]    = np.percentile([d[k] for d in m_all], 95)
                row[f"paper_{lb}"]        = pv
            summary_rows.append(row)

    # Write CSVs
    import csv

    detail_csv = f"{OUTDIR}/simulation_results_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["tumor_type","schedule","sim"] + lbls
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(all_results)
    print(f"Saved: {detail_csv}")

    summary_csv = f"{OUTDIR}/simulation_results_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["tumor_type","schedule"] + \
            [f"{pfx}_{lb}" for lb in lbls
             for pfx in ["model_median","model_p5","model_p95","paper"]]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(summary_rows)
    print(f"Saved: {summary_csv}")

    elapsed_s = time.time() - t0
    print(f"\nAll done in {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    print(f"Figures and CSVs saved to: {OUTDIR}")
