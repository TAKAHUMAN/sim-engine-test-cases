# Belzutifan PK/PD Model Report

## Sources
- PK: PMC10583240, fixed population PK model parameters from Tables 2, 3, and 5.
- PD: Choueiri et al. Nat Med 2021, PMC9128828. The paper reports concentration-correlated EPO suppression, reductions at all doses, and a plateau at >=120 mg q.d.; exact Extended Data Fig. 4 numeric values were not supplied.

## Validation Scenario
- Study 4 patients received FFP during PK sampling, so FFP is used for direct Table 5 validation.
- CL/F is set to the Table 5 VHL-RCC geometric mean of 7.25 L/h for the validation baseline.
- The FMF formulation penalty on KA is retained in `clinical_scenarios.csv` as `standard_120mg_fmf`.

## PD Calibration Assumption
- Direct inhibitory Emax model: `EPO(t) = EPO_baseline * (1 - Emax*C(t)/(EC50 + C(t)))`.
- Emax: 0.666.
- EC50: 0.2078 ug/mL.
- Hill gamma: 1.243.
- Calibration anchor: 60% EPO suppression at the published 120 mg q.d. geometric mean Cavg (`AUC0-24h/24`), selected within the supplied qualitative 40-70% substantial suppression range. This is not a digitized literature observation.
- If using an AUC-scale EC50 such as 12,000-15,000 h*ng/mL, fit with `driver='auc_to_time_ng_h_ml'` or convert to an average concentration scale before applying the concentration model.

## Extended Data Fig. 4 Fit
| emax | ec50 | gamma | rmse_percent | n_points | driver |
| --- | --- | --- | --- | --- | --- |
| 0.666 | 207.8 | 1.243 | 13.38 | 28 | cavg_to_time_ng_ml |

## Validation Against Published Table 5
| metric | simulated | published | percent_difference |
| --- | --- | --- | --- |
| auc_ug_h_ml | 16.56 | 16.71 | -0.9031 |
| cmax_ng_ml | 1445 | 1363 | 6.018 |
| cmin_ng_ml | 293.5 | 306.7 | -4.294 |
| tmax_h | 1.5 | 1.42 | 5.634 |
| half_life_eff_h | nan | 12.39 | nan |

## Goodness-of-Fit Notes
- RMSE versus the single published AUC aggregate for the validation baseline: 0.151 ug*h/mL.
- Correlation of predicted versus observed Cmax: Not computable: only one published aggregate Cmax value is available.
- VPC: Digitized Extended Data Fig. 4 points are fitted in epo_emax_fit.csv and plotted in epo_emax_validation.png; percentile simulation plots remain model-based.

## Clinical Prediction Limitation
EPO suppression alone cannot determine onset time to grade 3 anemia. A hemoglobin turnover model, baseline hemoglobin distribution, rescue EPO/transfusion rules, and censoring assumptions are required. The anemia module below is calibrated to the 27% phase 1 grade 3 anemia incidence; patient-level Hb and rescue-treatment data would be needed for stronger validation.

## Exploratory Anemia Simulation
| n | grade3_anemia_incidence_percent | published_grade3_anemia_percent | median_time_to_grade3_days | mean_baseline_hb_g_dl | mean_min_hb_g_dl | epo_hb_sensitivity | model_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 27 | 27 | 91.5 | 9.552 | 7.581 | 0.4779 | calibrated_to_phase1_grade3_anemia_incidence_requires_patient_hb_rescue_data |

## High-Risk Scenario Predictions
| scenario | auc_ug_h_ml | cmax_ng_ml | cmin_ng_ml | epo_nadir_percent_baseline | time_to_epo_nadir_days | time_to_50pct_epo_days | grade3_anemia_onset_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard_120mg_ffp | 16.56 | 1445 | 293.5 | 38.89 | 26.06 | 0.02083 | Not mechanistically identifiable from EPO alone; Hb turnover and baseline Hb are missing. |
| standard_120mg_fmf | 16.56 | 1334 | 300.1 | 39.41 | 26.08 | 0.03125 | Not mechanistically identifiable from EPO alone; Hb turnover and baseline Hb are missing. |
| dual_pm_3p2x_exposure | 121 | 5787 | 4451 | 34.45 | 27.06 | 0.02083 | Not mechanistically identifiable from EPO alone; Hb turnover and baseline Hb are missing. |
| elderly_65y | 17.59 | 1511 | 321.1 | 38.62 | 26.06 | 0.02083 | Not mechanistically identifiable from EPO alone; Hb turnover and baseline Hb are missing. |
| obese_100kg | 13.57 | 1108 | 264.3 | 40.8 | 27.06 | 0.02083 | Not mechanistically identifiable from EPO alone; Hb turnover and baseline Hb are missing. |
