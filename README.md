# Simulation Engine Test Cases

Validation test cases for the TAKAHUMAN simulation engine, organized by indication and drug.

## Repository Structure

```
test_cases/
  <INDICATION>/
    <drug_name>/
      <author_year>/
        sunitinib_pkpd.py   # Simulation script
        README.md            # Model description & validation
        results/
          figures/           # Output plots
          *.csv              # Simulation result tables
```

## Indications

| Indication | Drug | Reference | Status |
|---|---|---|---|
| RCC | Sunitinib | Khosravan 2016 | ✅ Complete |
| RCC | Sunitinib | (next model) | 🔄 Planned |

## Adding a New Test Case

1. Create folder: `test_cases/<INDICATION>/<drug>/<author_year>/`
2. Add simulation script and `README.md`
3. Run simulation, save figures to `results/figures/` and CSVs to `results/`
4. Submit a PR to `main`

## Current Coverage

- **RCC (Renal Cell Carcinoma):** 1 model
  - Sunitinib PK/PD — Khosravan et al., *Clin Pharmacokinet* 2016;55:1251–1269
