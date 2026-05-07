# Sorafenib — RCC PK/PD (Jain 2011 PK + Wilhelm 2008 RCC PD) [PMCIDs: PMC3162659, PMC12261297]

Exploratory **population PK** (Jain et al. 2011, *Br J Clin Pharmacol* 72:294-305) linked to **RCC-only xenograft PD** (786-O, Renca) from Wilhelm et al. 2008, *Mol Cancer Ther* 7(10):3129-3140, plus a **VEGFR-2 IC50 anchor** (Wilhelm et al. 2004, *Cancer Res* 64:7099).

**Not for clinical decision-making.** See assumptions file and parent repository guidelines.

## Source Split (PK vs PD)

- PK source paper: Jain et al. 2011 (PMCID: PMC3162659)
- PD source paper: Wilhelm et al. 2008 RCC xenograft preclinical PD (PMCID: PMC12261297)

## References

Jain L, Woo S, Gardner ER, Dahut WL, Kohn EC, Kummar S, Mould DR, Giaccone G, Yarchoan R, Venitz J, Figg WD. Population pharmacokinetic analysis of sorafenib in patients with solid tumours. *Br J Clin Pharmacol.* 2011;72(2):294–305. doi: 10.1111/j.1365-2125.2011.03963.x. PMCID: PMC3162659.

Wilhelm SM, Adnane L, Newell P, Villanueva A, Llovet JM, Lynch M. Preclinical overview of sorafenib, a multikinase inhibitor that targets both Raf and VEGF and PDGF receptor tyrosine kinase signaling. *Mol Cancer Ther.* 2008;7(10):3129–3140. doi: 10.1158/1535-7163.MCT-08-0013. PMCID: PMC12261297.

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
