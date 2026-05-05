# Pazopanib PK/PD Simulation - mRCC and STS
## Test Case: Tan et al. (2025)

### Overview
This test case captures the pazopanib model chain from:

- Tan Z, Voller S, Yin A, et al. Clin Pharmacokinet. 2025.
- "Model-Informed Dose Optimization of Pazopanib in Real-World Patients with Cancer"

Model blocks:

- PK: 1-compartment oral model, first-order absorption/elimination, dose-nonlinear F1
- Toxicity: Gompertz time-to-event model with Cmin,ss threshold at 34 mg/L
- Tumor dynamics: semimechanistic SLD model for mRCC and STS with primary/acquired resistance

### Folder Structure

```text
pazopanib/
  tan_2025/
    pazopanib_pkpd.py
    pazopanib_workflow.py
    README.md
    results/
      pazopanib_dose_comparison_summary.csv
      figures/
        pazopanib_cmin_dose_comparison.png
        pazopanib_toxicity_probability_comparison.png
        pazopanib_tumor_dynamics_comparison.png
        pazopanib_cmin_ss_hist_600mg_n200.png
```

### Files
- `pazopanib_pkpd.py`: Core PK + toxicity + tumor dynamics implementation.
- `pazopanib_workflow.py`: Multi-dose plotting workflow (400/600/800 mg).
- `results/pazopanib_dose_comparison_summary.csv`: Current summary table from corrected run.
- `results/figures/*.png`: Current generated validation plots.

### Current Parameterization
The PK and PD defaults are aligned to the supplementary NONMEM structure used in the paper:

- CL/F = 0.497 L/h
- V/F = 46.1 L
- Ka = 0.976 1/h (fixed)
- F1 = TVF1 * (200/DOSE)^0.42 * exp(eta)
- Toxicity Gompertz parameters: lambda 0.0021/day, shape -0.012/day, HR 3.35 for Cmin,ss > 34 mg/L
- Tumor dynamics parameters for mRCC and STS match reported KG/KD/lambda values

### Why Current Results Can Differ from Paper Figure Values
The current outputs are expected to be directionally consistent, but exact point estimates can differ for practical reasons:

1. **Simulation sampling size and protocol**
   - Current generated set uses n=200 per dose arm for rapid validation.
   - Paper qualification often uses larger simulation batches (e.g., 1000 simulation workflows for VPC-level checks).

2. **Simulation context vs real-world treatment course**
   - This test case runs fixed-dose 400/600/800 mg QD arms.
   - The observed real-world cohort includes dose modifications over treatment and mixed follow-up windows.

3. **Summary metric conventions**
   - This package reports model-based daily trough summaries at fixed simulation days.
   - Published summaries can mix event-day/censor-day exposure extraction and additional NONMEM post-processing conventions.

4. **Validation scope**
   - This repository stores deterministic validation artifacts for reproducibility.
   - It is not a full rerun of the original end-to-end estimation + qualification pipeline from the publication environment.

### How To Run
From repository root:

```bash
python test_cases/RCC/pazopanib/tan_2025/pazopanib_workflow.py pazopanib_plots
```

This writes three figure outputs and a summary CSV to the chosen output directory.

