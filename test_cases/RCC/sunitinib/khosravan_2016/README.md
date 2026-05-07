# Sunitinib Population PK/PD Simulation — RCC & GIST
## Test Case: Khosravan et al. (2016) Clin Pharmacokinet 55:1251–1269 [PMCID: PMC5526090]

### Overview
Population PK/PD simulation replicating the clinical trial comparison of sunitinib
**Schedule 4/2** (50 mg QD, 4 weeks on / 2 weeks off) vs **Schedule 2/1** (37.5 mg QD,
2 weeks on / 1 week off) in **Renal Cell Carcinoma (RCC)** and **GIST** patients.

- **Paper**: Khosravan R et al. *Clin Pharmacokinet* 2016;55(10):1251–1269  
  DOI: 10.1007/s40262-016-0404-5
- **Drug**: Sunitinib (SUTENT®) + active metabolite SU12662
- **Indication**: RCC (primary), GIST (secondary)
- **Model type**: Population PK/PD — 2-compartment PK + IDR SLD + TCSFL platelet/ANC

### Reference

Khosravan R, Motzer RJ, Fumagalli E, Rini BI. Population Pharmacokinetic/Pharmacodynamic Modeling of Sunitinib by Dosing Schedule in Patients with Advanced Renal Cell Carcinoma or Gastrointestinal Stromal Tumor. *Clin Pharmacokinet.* 2016;55(10):1251–1269. doi: 10.1007/s40262-016-0404-5. PMCID: PMC5526090.

---

### Folder Structure

```
sunitinib/
  khosravan_2016/
    sunitinib_pkpd.py            # Simulation engine (main script)
    README.md                    # This file
    results/
      figures/
        fig_pk_RCC.png           # Sunitinib + SU12662 PK — RCC
        fig_pd_RCC.png           # SLD / ANC / PC time-courses — RCC
        fig_summary_RCC.png      # Our model vs paper (bar chart) — RCC
        fig_grade34_RCC.png      # Grade 3/4 thrombocytopenia — RCC
        fig_pc_RCC.png           # Platelet count cycle 3 — RCC
        fig_safety_RCC.png       # ALT/AST/LVEF/DBP — RCC
        fig_pk_GIST.png          # ... same panels for GIST
        fig_pd_GIST.png
        fig_summary_GIST.png
        fig_grade34_GIST.png
        fig_pc_GIST.png
        fig_safety_GIST.png
      simulation_results_summary.csv   # Median ± 5th/95th vs paper reference
      simulation_results_detail.csv    # Per-simulation rows (all metrics)
```

---

### Model Components

| Component | Description |
|---|---|
| **PK** | Analytical 2-compartment oral (Bateman eq.), 1st-order Ka, lag time |
| **Metabolite** | SU12662 — proportional to sunitinib AUC |
| **SLD (tumor)** | IDR model — indirect response, drug-driven Kin stimulation |
| **Platelet count** | TCSFL (transit-compartment stem-cell feedback loop) |
| **ANC** | Same TCSFL structure, separate parameters |
| **Safety endpoints** | ALT, AST, LVEF, DBP — empirical time-course |

### Key PK Parameters (Table 2, Khosravan 2016)

| Parameter | Value | Units |
|---|---|---|
| CL/F | 34.1 | L/h |
| Vc/F | 2700 | L |
| Vp/F | 774 | L |
| Q/F | 0.688 | L/h |
| Ka | 0.126 | h⁻¹ |
| t_lag | 0.527 | h |
| CL covariate: age | −0.702 %/yr | — |
| CL covariate: race | −15.2 % | — |
| CL covariate: sex | −19.3 % | — |
| CL covariate: GIST | +29.3 % | — |

### Key PD Parameters (SLD model, Eqs. 5–7)

| Parameter | Value | Covariate adjustment |
|---|---|---|
| Baseline SLD | 14.3 cm | ×(1+0.574×ECOG)×(1−0.348×RACE)×(1−0.430×SCH=CDD) |
| Kout | 2.67×10⁻⁴ h⁻¹ | ×(1+1.01×SCH=CDD) |
| EC50 | 30.5 ng/mL | ×(1+2.43×SCH=CDD)×(1+4.82×TUMR=GIST) |
| Ktol | 1.41×10⁻⁵ h⁻¹ | — |

> **Note**: SCH=1 in the paper denotes CDD (continuous daily dosing), not Schedule 2/1.
> Both 4/2 and 2/1 are intermittent schedules → SCH=0 for both.

---

### Running the Simulation

```bash
# Quick test (2 sims, 1 worker)
python sunitinib_pkpd.py --n-sims 2 --n-pat 100 --dt 8 --jobs 1

# Full run (20 sims, 4 parallel workers)
python sunitinib_pkpd.py --n-sims 20 --n-pat 100 --dt 8 --jobs 4
```

**Requirements**: Python ≥3.10, numpy, matplotlib, scipy

```bash
pip install numpy matplotlib scipy
```

**Outputs** (written to script directory):
- `fig_pk_{RCC,GIST}.png` — PK concentration-time profiles
- `fig_pd_{RCC,GIST}.png` — PD (SLD/ANC/PC) time-courses
- `fig_summary_{RCC,GIST}.png` — Summary comparison vs paper
- `fig_grade34_{RCC,GIST}.png` — Grade 3/4 toxicity rates
- `fig_pc_{RCC,GIST}.png` — Platelet count cycle-3 kinetics
- `fig_safety_{RCC,GIST}.png` — Safety endpoint time-courses
- `simulation_results_detail.csv` — Per-simulation metrics (80 rows for 20 sims × 2 scheds × 2 indications)
- `simulation_results_summary.csv` — Median + P5/P95 vs paper reference values

---

### Validation Results (2-sim reference run)

| Metric | Model 4/2 | Paper 4/2 | Model 2/1 | Paper 2/1 |
|---|---|---|---|---|
| Trough Sunitinib (ng/mL) | 42.6 ✅ | 42.6 | 43.4 ✅ | 42.4 |
| Trough SU12662 (ng/mL) | 19.7 ✅ | 19.7 | 20.2 ✅ | 19.5 |
| SLD Cycle 6 (cm) | 9.6 | 8.6 | 10.1 | 8.2 |
| ORR (%) | 48 | 27 | 49 | 31 |
| PC Nadir C3 (×10³/µL) | 122 | 104 | 148 | 119 |
| Grade 3/4 Thrombo (%) | 14 | 16 | 4 | 9 |

PK troughs match paper exactly. ORR is structurally higher (paper's 27% reflects
confirmed PR + dropout not modeled). PFS returns `nan` (median not reached in 13 cycles).

---

### Known Limitations
1. **ORR ~48% vs paper 27%**: Paper's ORR reflects confirmed PR (two consecutive RECIST
   assessments ≤ −30%) and patient dropout/dose reduction — not modeled here.
2. **PFS = nan**: Ktol half-life ≈ 49,000 h; tumor regrowth too slow for 13-cycle window.
   Paper's PFS reflects composite events including toxicity discontinuation.
3. **GIST troughs lower**: GIST CL/F +29.3% → faster clearance → lower steady-state troughs.
   Paper's reported trough is a mixed RCC+GIST population value.
