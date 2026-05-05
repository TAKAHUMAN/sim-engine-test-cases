# Simulation Engine Test Cases

Validation test cases for the TAKAHUMAN simulation engine, organized by indication and drug.

## Repository Structure

```
test_cases/
  <INDICATION>/
    <drug_name>/
      <author_year>/
        sunitinib_pkpd.py       # Simulation script
        README.md               # Model description and validation
        results/
          figures/              # Output plots
          *.csv                 # Simulation result tables
```

## Indications

| Indication | Drug | Reference | Status |
|---|---|---|---|
| RCC | Sunitinib | Khosravan 2016 | Complete |
| RCC | Sunitinib | Diekstra 2017 | Complete |
| RCC | Axitinib | Schindler 2017 | Complete |

## Adding a New Test Case

1. Create folder: `test_cases/<INDICATION>/<drug>/<author_year>/`
2. Add simulation script and `README.md`
3. Run simulation, save figures to `results/figures/` and CSVs to `results/`
4. Submit a PR to `main`

## Current Coverage

- RCC (Renal Cell Carcinoma): 3 models
  - Sunitinib PK/PD - Khosravan et al., Clin Pharmacokinet 2016;55:1251-1269
  - Sunitinib PK/PD - Diekstra et al. 2017
  - Axitinib PK/PD chain - Schindler et al. 2017
