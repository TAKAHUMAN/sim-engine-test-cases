# Sunitinib PK/PD Simulation — mRCC
## Test Case: Diekstra et al. (2017)

### Overview
Population PK/PD test case for sunitinib and active metabolite SU12662 in mRCC,
including dual biomarker dynamics for sVEGFR-2 and sVEGFR-3.

- Reference: Diekstra et al. 2017
- Drug: Sunitinib + SU12662
- Indication: RCC (mRCC)
- Model type: Semiphysiological PK + indirect-response PD

### Key Structural Notes
- Parent PK uses an enzyme compartment linked by QH (liver blood flow), with metabolite formation from that compartment.
- PD follows inverse-linear inhibition of production:
  - INH = ACu / (Kd + ACu)
  - dR/dt = kin * (1 / (1 + alpha * INH)) - kout * R
- ACu uses full active metabolite unbound contribution (weight = 1.0).

### Files
- sunitinib_pkpd.py: PK/PD model and simulation utilities
- sunitinib_paper_workflow.py: parameter loading, baseline simulation, VPC/GOF helpers
- results/figures/: output location for generated plots

### Notes
This case was added as the second RCC sunitinib test case in this repository.
