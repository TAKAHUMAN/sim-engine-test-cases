import numpy as np
import pandas as pd
from pathlib import Path

from simulation.pd_validate import format_pd_validation_report, kaplan_meier_curve, validate_against_record1


def test_kaplan_meier_curve_decreases_on_events() -> None:
    times, survival = kaplan_meier_curve(
        np.array([1.0, 2.0, 2.0, 3.0]), np.array([True, True, False, True])
    )
    assert times[0] == 0.0
    assert survival[0] == 1.0
    assert survival[-1] < survival[0]


def test_record1_validation_schema() -> None:
    df = pd.DataFrame(
        {
            "PFS_days": [120.0, 150.0, 365.0],
            "PFS_months": [120.0 / 30.44, 150.0 / 30.44, 365.0 / 30.44],
            "event_observed": [True, True, False],
            "median_s6k1_inhibition": [0.8, 0.82, 0.9],
        }
    )
    checks = validate_against_record1(df, output_dir=Path("outputs/test_pd_validate"))
    assert {"target", "simulated", "reference", "pass"}.issubset(checks.columns)
    report = format_pd_validation_report(df, checks)
    assert "Model refinement" in report
    assert "Supervisor summary" in report
