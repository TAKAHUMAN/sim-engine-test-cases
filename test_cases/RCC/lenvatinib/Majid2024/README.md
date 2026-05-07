# Lenvatinib — integrated PK / biomarker PD / TGI (RCC-adapted) [PMCID: PMC11179699]

Deterministic implementation of the **Majid et al. (2024)** integrated PopPK / biomarker / TGI model (RR-DTC), with **RCC CL/F × 0.851** applied to clearance for RCC-style simulations.

**Not for clinical decision-making.**

## Reference (Methods)

Majid O, Hayato S, Sreerama Reddy SH, Lalovic B, Hihara T, Hoshi T, Funahashi Y, Aluri J, Takenaka O, Yasuda S, Hussein Z. Population pharmacokinetic-pharmacodynamic modeling of serum biomarkers as predictors of tumor dynamics following lenvatinib treatment in patients with radioiodine-refractory differentiated thyroid cancer (RR-DTC). *CPT Pharmacometrics Syst Pharmacol.* 2024;13(6):954–969. doi: [10.1002/psp4.13130](https://doi.org/10.1002/psp4.13130). PMCID: PMC11179699.

## Run

```bash
cd test_cases/RCC/lenvatinib/Majid2024
pip install -r requirements.txt
python lenvatinib_rcc_pkpd.py
```

## Results and figures (default paths)

| Path | Content |
|------|---------|
| `results/lenvatinib_rcc_timeseries.csv` | `C_ng_mL`, **`AUC_cum_biomarker_ng_h_mL`**, **`AUC_interval_tumor_ng_h_mL`**, tumor mm, % change biomarkers/tumor |
| `results/figures/figure4_style_pk_pd_tgi.png` | Biomarkers, tumor % change, concentration + interval AUC |
| `results/validation_checks.txt` | Figure 4–style checks on the **primary** patient; Section **C** only: >35% shrink (73.2 kg, 70.2 mm) |
| `results/assumptions_block.txt` | Assumptions A1–A7 |

## Key CLI flags

| Flag | Role |
|------|------|
| `--tumor-auc-window-weeks` | Trailing **interval AUC** window for TGI Hill (default **8** wk, RECIST tumor assessments every 8 wk in randomization phase). |
| `--dt-h` | Euler step (h); default **0.1** (stiff Ang-2 Hill γ). |
| `--bw-kg` | Default **75.1** (Figure 4 simulation patient). |
| `--y0-tumor-mm` | Default **59.5** mm SLD (Figure 4 median from studies 303+211). Shrinkage **>35%** caption uses **73.2 kg / 70.2 mm** in `validation_checks.txt` Section C only. |
| `--biomarker-dpslope` | Optional Table 2 DPslope (1/h); default **0** for long treated-only runs. |
| `--pk-legacy-split` + `--f-zero-order` | Legacy fraction-split ZO/FO; default is single-depot Majid-style PK. |

## Model summary

- **PK:** Table 1; QD; single absorption depot per dose; simultaneous ZO + FO with **mass-conserving** flux cap per step.
- **Biomarkers:** **Cumulative** AUC drives Hill terms.
- **TGI:** **Interval** AUC over the trailing **W** weeks drives the tumor Hill; **W = 8** matches assessment cadence in Majid et al.
- **Tumor ODE:** Second pass on stored PK/PD (no tumor → biomarker feedback).
