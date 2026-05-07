# Everolimus PBPK/PD Simulation - RCC
## Test Case: Pawaskar et al. (2013) + RECORD-1 linkage [PMCID: PMC3755750]

### Overview
This test case contains an everolimus PBPK/PD workflow for RCC translation:

- PBPK: oral absorption + one-compartment body pool (default) with tissue partition mapping
- Link: tumor concentration (`C_tumor`) mapped to PD effect via mechanistic or empirical mode
- PD: tumor-size dynamics using RECORD-1 style growth/effect equation
- Validation outputs: 5 mg and 10 mg summaries for mechanistic and empirical runs

### References

Pawaskar DK, Straubinger RM, Fetterly GJ, Hylander BH, Repasky EA, Ma WW, Jusko WJ. Physiologically based pharmacokinetic models for everolimus and sorafenib in mice. *Cancer Chemother Pharmacol.* 2013;71(5):1219–1229. doi: 10.1007/s00280-013-2116-y. PMCID: PMC3755750.

Motzer RJ, Escudier B, Oudard S, Hutson TE, Porta C, Bracarda S, Grünwald V, Thompson JA, Figlin RA, Hollaender N, Urbanowitz G, Berg WJ, Kay A, Lebwohl D, Ravaud A, RECORD-1 Study Group. Efficacy of everolimus in advanced renal cell carcinoma: a double-blind, randomised, placebo-controlled phase III trial. *Lancet.* 2008;372(9637):449–456. doi: 10.1016/S0140-6736(08)61039-9. PMCID: not available in Europe PMC.

### Source Split (PK vs PD)

- PK source paper: Pawaskar et al. 2013 (PMCID: PMC3755750)
- PD/clinical linkage source: RECORD-1 (Motzer et al. 2008; no PMCID in Europe PMC)

### Files
- `everolimus_pkpd.py`: Core implementation and CLI
- `results/summary_phase1b.csv`: Phase 1B comparison table
- `results/summary_dose_5mg_mechanistic.csv`
- `results/summary_dose_10mg_mechanistic.csv`
- `results/summary_dose_5mg_empirical.csv`
- `results/summary_dose_10mg_empirical.csv`

### How To Run
From repository root:

```bash
python test_cases/RCC/everolimus/pawaskar_2013/everolimus_pkpd.py diagnostic --dose 0.125 --context mouse_preclinical --model onecompartment --duration-days 7
python test_cases/RCC/everolimus/pawaskar_2013/everolimus_pkpd.py diagnostic --dose 10 --context human_clinical --model onecompartment --duration-days 365
```

### Notes
- Mouse mode uses Pawaskar-aligned oral absorption and partition assumptions.
- Human mode is allometrically scaled from mouse reference parameters.
- One-compartment mode is the default due better consistency with current validation checks.
