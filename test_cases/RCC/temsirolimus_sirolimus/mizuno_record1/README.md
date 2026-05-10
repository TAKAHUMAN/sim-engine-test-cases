# Population PK/PD Model of Sirolimus and Temsirolimus in Adult RCC

This repository contains a mechanistic population pharmacokinetic and
pharmacodynamic simulation engine for temsirolimus, sirolimus, mTOR target
engagement, and progression-free survival (PFS) in adult renal cell carcinoma
(RCC).

The model links:

1. Plasma temsirolimus and sirolimus concentrations.
2. Molar-summed mTOR-active drug concentration.
3. pS6K1 inhibition as a pharmacodynamic biomarker.
4. Tumor log-growth with acquired resistance.
5. Individual and cohort-level PFS outcomes.

The current clinical-use candidate is the acquired-resistance model calibrated
to RECORD-1 aggregate PFS benchmarks.

## Source Evidence

The implementation is based on three published evidence layers:

1. PK: Mizuno et al. 2016, pediatric parent-metabolite temsirolimus and
   sirolimus population PK model.
2. Biomarker: O'Donnell et al. 2008, everolimus mTOR target engagement and
   pS6K1 suppression.
3. Efficacy: George and Bukowski 2009 / RECORD-1, everolimus PFS benchmarks in
   advanced RCC after VEGF-targeted therapy.

All important departures from those papers are documented in
[DEVIATIONS.md](DEVIATIONS.md).

## Validation Status

Final calibrated acquired-resistance model:

```text
lambda_kill(t) = lambda_kill_0 * exp(-t / tau_resist)

lambda_kill_0 = 0.02455 /day
tau_resist    = 39.50 days
```

Validation against RECORD-1 using a 500-patient simulated adult RCC cohort:

| Endpoint | Predicted | Target | Status |
| --- | ---: | ---: | --- |
| Median PFS | 4.01 mo | 4.0-5.5 mo | PASS |
| 6-month PFS | 29.8% | 22.1-29.9% | PASS |
| 12-month PFS | 7.4% | 5-10% | PASS |
| Median S6K1 inhibition | 79.9% | >=80% | NEAR PASS |

All simulations use `numpy.random.default_rng(seed=20240501)` unless otherwise
specified.

## Repository Layout

```text
.
├── README.md
├── METHODS.md
├── DEVIATIONS.md
├── LICENSE
├── requirements.txt
├── model/
│   ├── pk_model.py
│   ├── pk_interface.py
│   ├── biomarker_model.py
│   ├── tumor_growth_model.py
│   ├── pd_resistance.py
│   └── pd_mixture.py
├── simulation/
│   ├── pk_simulate.py
│   ├── pk_validate.py
│   ├── pd_simulate.py
│   ├── pd_validate.py
│   └── main.py
├── tests/
├── outputs/
├── notebooks/
└── docs/
```

Large regenerated CSV outputs are ignored by git. Lightweight validation
reports and figures under `outputs/` are kept for review.

## Quick Start

```bash
pip install -r requirements.txt
pytest tests -q -p no:cacheprovider
python -m simulation.main
```

Primary calibrated outputs:

```text
outputs/pd_calibrated_validation_report.txt
outputs/pd_calibrated_pfs_validation.png
outputs/pd_validation_report.txt
outputs/pd_pfs_validation.png
outputs/pcvpc.png
```

## Model Summary

### Pharmacokinetics

The PK layer reproduces the Mizuno et al. combined parent-metabolite model:

- Temsirolimus: three compartments with IV zero-order infusion.
- Sirolimus: two compartments formed from temsirolimus clearance through `Fm`.
- Molecular-weight correction is applied during metabolite formation.
- Amounts are in micrograms, volumes are in liters, and concentrations are
  therefore micrograms/L, numerically equal to ng/mL.

Adult RCC adaptation:

- Dose: 25 mg IV weekly.
- Infusion duration: 30 minutes.
- Adult conversion fraction: `Fm = 0.70`.
- Week-4 steady-state approximation: week-1 profile scaled by accumulation
  factor 1.5.

### Biomarker

The biomarker model uses molar-summed active concentration:

```text
C_um = TEM_ng_ml / 1030.3 + SIR_ng_ml / 914.2
```

S6K1 inhibition:

```text
E(t) = Emax * C(t)^gamma / (EC50^gamma + C(t)^gamma)
```

with `Emax = 0.95`, `EC50 = 0.010 uM`, and `gamma = 1.2`.

### Tumor Dynamics

Tumor size is modeled on the natural log scale:

```text
dL/dt = lambda_growth - lambda_kill(t) * E(t)
lambda_kill(t) = lambda_kill_0 * exp(-t / tau_resist)
```

Progression is the first time tumor log-size increases by `log(1.20)` from
baseline. Patients not progressing by 365 days are administratively censored.

## Calibration

The calibrated resistance model jointly fits:

- Median PFS target: 4.9 months.
- 6-month PFS target: 26%.
- 12-month PFS target: 7.5%.

The objective uses relative squared error across these endpoints. Because
survival probabilities and medians are step-like in a finite simulated cohort,
the calibrator uses coarse-grid seeding before optional Powell local
optimization.

The committed calibrated report was generated with:

```text
lambda_kill_0 = 0.02455 /day
tau_resist    = 39.50 days
N             = 500
seed          = 20240501
```

## Diagnostic Models

Two diagnostic refinements are retained:

- Constant-efficacy tumor model: matches median PFS but overpredicts the long
  PFS tail.
- Responder/resistant mixture model: tested and rejected as a primary structure
  because the supplied quick-path assumptions were internally inconsistent and
  failed to recover RECORD-1 median PFS.

See [DEVIATIONS.md](DEVIATIONS.md) for the scientific interpretation.

## Testing

```bash
pytest tests -q -p no:cacheprovider
```

Current status:

```text
26 passed
```

## Limitations

This is a mechanistic simulation and calibration exercise, not a clinically
validated dosing tool. It requires prospective adult RCC PK/PD validation with
serial temsirolimus, sirolimus, and pS6K1 measurements before clinical
decision support use.

Known limitations include:

- Everolimus biomarker parameters are transposed to sirolimus/temsirolimus.
- Adult `Fm = 0.70` is inferred from pediatric-to-adult AUC-ratio discussion.
- Resistance kinetics are calibrated from aggregate RECORD-1 endpoints, not
  individual patient data.
- S6K1 inhibition is near the 80% target but falls just below it in the seeded
  cohort.

## References

1. Mizuno T, Fukuda T, Christians U, et al. Population pharmacokinetics of
   temsirolimus and sirolimus in children with recurrent solid tumours. Br J
   Clin Pharmacol. 2017;83(5):1097-1107.
2. O'Donnell A, Faivre S, Burris HA, et al. Phase I pharmacokinetic and
   pharmacodynamic study of everolimus. J Clin Oncol. 2008;26(10):1588-1595.
3. George S, Bukowski RM. Role of everolimus in the treatment of renal cell
   carcinoma. Ther Clin Risk Manag. 2009;5:699-706.
4. Motzer RJ, Escudier B, Oudard S, et al. Efficacy of everolimus in advanced
   renal cell carcinoma. Lancet. 2008;372(9637):449-456.

## License

MIT. See [LICENSE](LICENSE).

