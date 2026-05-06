# Sorafenib — RCC PK/PD (Jain 2011 PK + Wilhelm 2008 RCC PD)

Exploratory **population PK** (Jain et al. 2011, *Br J Clin Pharmacol* 72:294-305) linked to **RCC-only xenograft PD** (786-O, Renca) from Wilhelm et al. 2008, *Mol Cancer Ther* 7(10):3129-3140, plus a **VEGFR-2 IC50 anchor** (Wilhelm et al. 2004, *Cancer Res* 64:7099).

**Not for clinical decision-making.** See assumptions file and parent repository guidelines.

## How to run

```bash
cd test_cases/RCC/sorafenib/Jain2011_Wilhelm2008
pip install -r requirements.txt
python sorafenib_rcc_pkpd.py
```

Default runtime is several minutes at `n_subjects=1000` and `dose_sweep_subjects=300`. For a quick smoke test:

```bash
python sorafenib_rcc_pkpd.py --n-subjects 100 --dose-sweep-subjects 30
```

## Outputs (after run)

| Path | Content |
|------|---------|
| `results/output1_pk_400mg_bid_summary.csv` | PK summary @ 400 mg BID steady state |
| `results/output4_checkpoint_table.txt` | QA checkpoint table |
| `results/output5_assumptions_block.txt` | Assumptions A1–A14 |
| `results/figures/*.png` | Calibration, dose–response, combined, `fu_mouse` sensitivity |

## Validation hooks

- **`verify_ehc_mass_balance()`** runs first: single-subject AUC ratio (EHC on/off) must fall in **1.7–2.3** (target ~2× for ~50% EHC contribution in the QA scenario).
- **`run_sanity_checks()`** asserts basic PK/Hill invariants.

## PD translation (3 approaches)

1. **Total AUC** (paper-consistent with Wilhelm’s cross-species **total** AUC narrative).  
2. **Free AUC** — scenarios **2a** (`fu_mouse=0.030`) and **2b** (`fu_mouse=0.005`).  
3. **VEGFR-2 IC50 anchor** on human **free Cavg** with fixed EC50 = **0.090 µM** (free).
