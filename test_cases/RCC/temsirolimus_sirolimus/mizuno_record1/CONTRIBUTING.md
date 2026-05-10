# Contributing

This repository contains scientific simulation code. Contributions should keep
the model auditable and reproducible.

## Expectations

- Document every new parameter source.
- Add or update tests for model behavior.
- Do not silently change published parameters.
- If a parameter is calibrated rather than published, state the target, loss
  function, cohort size, seed, and resulting value.
- Keep generated large CSV outputs out of git.

## Validation

Run before opening a pull request:

```bash
pytest tests -q -p no:cacheprovider
python -m simulation.main
```

