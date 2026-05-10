from pathlib import Path

from model.pd_mixture import (
    calibrate_mixture_grid,
    estimate_responder_fraction_from_anchors,
    run_mixture_refined_pipeline,
)


def test_estimate_responder_fraction_from_anchors() -> None:
    pi, responder_rate, resistant_rate = estimate_responder_fraction_from_anchors()
    assert 0.0 <= pi <= 1.0
    assert responder_rate < resistant_rate


def test_mixture_grid_schema_small_run() -> None:
    grid = calibrate_mixture_grid(
        responder_fraction_grid=(0.35,),
        tau_grid_days=(69.0,),
        n_individuals=2,
    )
    assert {"responder_fraction", "median_pfs_months", "combined_score"}.issubset(
        grid.columns
    )


def test_mixture_refined_pipeline_small_run() -> None:
    result = run_mixture_refined_pipeline(
        n_individuals=2,
        responder_fraction=0.35,
        output_dir=Path("outputs/test_pd_mixture"),
        output_prefix="test_pd_mixture",
    )
    assert result.responder_fraction == 0.35
    assert "responder/resistant mixture" in result.report
