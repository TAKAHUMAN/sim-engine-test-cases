# PT2385 PK/PD Simulation - RCC
## Test Case: Courtney et al. (2018), JCO

### Overview

This test case implements a semi-mechanistic PK/PD validation workflow for PT2385, an oral HIF-2alpha antagonist evaluated in adults with advanced clear cell renal cell carcinoma.

Model blocks:

- PK: 1-compartment oral model with first-order absorption depot, linear elimination, and dose-dependent oral bioavailability to reproduce the exposure plateau above 800 mg BID.
- PD: indirect-response Type I EPO model, where PT2385 inhibits EPO production through HIF-2alpha transcriptional blockade.
- PD fitting: weighted least-squares fit of EPO IC50 to digitized Courtney et al. 2018 Figure 2B data, with Kout fixed to a physiological EPO half-life.
- Exposure-efficacy visualization: clinical Cmin,ss threshold of 0.5 ug/mL linked to stable disease >= 4 months probability.

### Reference

Courtney KD et al. PT2385, a first-in-class HIF-2alpha antagonist, in patients with previously treated advanced clear cell renal cell carcinoma. *Journal of Clinical Oncology*. 2018.

Additional PD turnover source:

Jelkmann W. Regulation of erythropoietin production. *Physiological Reviews*. 2011;91(4):1277-1337.

### Source Split

- PK and exposure-efficacy source: Courtney et al. 2018, including Day 15 800 mg BID PK summary, accumulation ratio, and 0.5 ug/mL Cmin,ss clinical activity threshold.
- EPO PD response source: digitized Courtney et al. 2018 Figure 2B.
- EPO Kout source: fixed physiological endogenous EPO half-life of 5 h from literature; not fit from sparse trial timepoints.

### Folder Structure

```text
pt2385/
  courtney_2018/
    pt2385_pkpd_model.py
    fit_epo_pd.py
    requirements.txt
    README.md
    results/
      fitted_pd_parameters.txt
      pt2385_day15_summary.csv
      figures/
        epo_exposure_response_steady_state.png
        epo_fit_observed_vs_predicted.png
        pt2385_day15_trough_threshold.png
        pt2385_epo_dose_response.png
        pt2385_epo_time.png
        pt2385_exposure_efficacy.png
        pt2385_pk_day1_day15.png
```

### Files

- `pt2385_pkpd_model.py`: Forward PK/PD simulation, dose comparison, verification checks, and publication-style figures.
- `fit_epo_pd.py`: Weighted least-squares EPO PD fitting workflow for IC50, including sensitivity analysis over fixed EPO half-life values.
- `requirements.txt`: Minimal Python package requirements.
- `results/*.csv` and `results/*.txt`: Current deterministic output tables and fitted PD parameter report.
- `results/figures/*.png`: Current generated validation and diagnostic plots.

### Current Parameterization

PK parameters:

| Parameter | Value | Units | Source |
|---|---:|---|---|
| ke | 0.04076 | h^-1 | Courtney et al. 2018, T1/2 = 17 h |
| ka | 1.98305 | h^-1 | Back-calculated from Tmax = 2 h |
| Vd/F | 265,305 | mL | Back-calculated from estimated single-dose Cmax after bioavailability model |
| F0 | 1.0 | fraction | Assumed low-dose oral bioavailability limit |
| Fmax bioavailability floor | 0.10 | fraction | Assumed high-dose bioavailability floor |
| Kd bioavailability | 500,000 | ug | Assumed dose-dependence parameter |

PD parameters:

| Parameter | Value | Units | Source |
|---|---:|---|---|
| Kout | 0.138600 | h^-1 | Fixed endogenous EPO T1/2 = 5 h |
| IC50 | 0.475676 | ug/mL | Fitted from digitized Courtney et al. 2018 Figure 2B |
| Imax | 1.0 | fraction | Fixed |
| Hill coefficient | 1.0 | unitless | Fixed due sparse timepoints |

IC50 sensitivity across fixed EPO half-life values:

| EPO T1/2 | IC50 | Weighted SSR |
|---:|---:|---:|
| 4 h | 0.5071 ug/mL | 82.8773 |
| 5 h | 0.4757 ug/mL | 88.5030 |
| 6 h | 0.4441 ug/mL | 93.6008 |
| 8 h | 0.3904 ug/mL | 103.3822 |

### How To Run

From repository root:

```bash
python test_cases/RCC/pt2385/courtney_2018/pt2385_pkpd_model.py
python test_cases/RCC/pt2385/courtney_2018/fit_epo_pd.py
```

The first command regenerates the deterministic PK/PD outputs and figures. The second command regenerates the fitted EPO PD parameter report and diagnostic fitting figures.

### Validation Checkpoints

Current deterministic 800 mg BID results:

| Metric | Value | Target |
|---|---:|---:|
| Day 15 Cmax | 3.27 ug/mL | 3.0-3.2 ug/mL |
| Day 1 Tmax | 2.0 h | 1.5-2.5 h |
| AUC accumulation ratio | 2.7 | 2.3-2.7 |
| Day 15 Cmin,ss | 2.18 ug/mL | >= 0.5 ug/mL |
| Day 15 EPO trough | 0.156 normalized | < 0.2 |
| 1800/800 absorbed amount ratio | 1.49 | < 1.6 |
| 1800/800 Cmax ratio | 1.49 | < 1.5 |

Dose comparison at Day 15:

| Dose | Cmax | Cmin | AUC0-12h | EPO trough |
|---:|---:|---:|---:|---:|
| 100 mg BID | 0.779 ug/mL | 0.518 ug/mL | 7.859 ug*h/mL | 0.436 |
| 200 mg BID | 1.361 ug/mL | 0.906 ug/mL | 13.737 ug*h/mL | 0.307 |
| 400 mg BID | 2.199 ug/mL | 1.464 ug/mL | 22.191 ug*h/mL | 0.216 |
| 800 mg BID | 3.270 ug/mL | 2.177 ug/mL | 33.002 ug*h/mL | 0.156 |
| 1200 mg BID | 4.009 ug/mL | 2.669 ug/mL | 40.466 ug*h/mL | 0.131 |
| 1800 mg BID | 4.875 ug/mL | 3.246 ug/mL | 49.206 ug*h/mL | 0.110 |

### Notes

- The fitted EPO IC50 of 0.475676 ug/mL agrees closely with the independently reported 0.5 ug/mL exposure-efficacy threshold, supporting the PD model linkage.
- The 800 mg Day 15 Cmax is slightly above the 3.0-3.2 ug/mL target window. This is left transparent rather than retuned with unsupported free parameters.
- The bioavailability model is an assumed structural approximation to the observed PT2385 exposure plateau above 800 mg BID and should be refit if individual PK data become available.
- The EPO fit uses digitized mean and SD values only. Confidence intervals require bootstrap or individual-level data and are not estimated here.
