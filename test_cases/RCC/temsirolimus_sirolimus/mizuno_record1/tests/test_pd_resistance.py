import pandas as pd
from pathlib import Path

from model.pd_resistance import (
    calibrate_tau_grid,
    joint_calibration_loss,
    resistance_endpoints,
    run_joint_calibration,
    run_resistance_refined_pipeline,
)


def test_tau_grid_calibration_schema_small_run() -> None:
    grid = calibrate_tau_grid(tau_grid_days=(60.0, 70.0), n_individuals=2)
    assert {"tau_resist_days", "combined_score", "pfs_6m", "pfs_12m"}.issubset(grid.columns)
    assert len(grid) == 2


def test_resistance_refined_pipeline_small_run() -> None:
    result = run_resistance_refined_pipeline(
        n_individuals=2,
        tau_resist_days=69.0,
        output_dir=Path("outputs/test_pd_resistance"),
    )
    assert result.tau_resist_days == 69.0
    assert "lambda_kill(t)" in result.report


def test_joint_calibration_helpers() -> None:
    df = pd.DataFrame(
        {
            "PFS_days": [100.0, 200.0, 365.0],
            "PFS_months": [100.0 / 30.44, 200.0 / 30.44, 365.0 / 30.44],
        }
    )
    endpoints = resistance_endpoints(df)
    assert set(endpoints) == {"median", "pfs_6m", "pfs_12m"}
    assert joint_calibration_loss(endpoints) >= 0.0


def test_joint_calibration_small_run() -> None:
    result = run_joint_calibration(
        n_calibration=2,
        n_validation=2,
        maxiter=1,
        coarse_lambda_grid=(0.015,),
        coarse_tau_grid=(69.0,),
        output_dir=Path("outputs/test_pd_resistance_calibration"),
    )
    assert result.lambda_kill_0_median > 0.0
    assert result.tau_resist_days > 0.0
    assert "jointly calibrated" in result.report
