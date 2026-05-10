import pandas as pd

from simulation.main import build_one_page_report


def test_build_one_page_report_contains_required_sections() -> None:
    recovered = pd.DataFrame(
        {
            "parameter": ["CL_TEM"],
            "estimated": [4.31],
            "paper": [4.31],
            "deviation_pct": [0.0],
            "method": ["NCA"],
        }
    )
    auc_summary = pd.DataFrame(
        {
            "dose_mg_m2": [15.0],
            "auc_ratio_mean": [1.21],
            "auc_ratio_median": [1.0],
            "auc_ratio_sd": [1.18],
        }
    )
    ci_checks = pd.DataFrame(
        {
            "parameter": ["CL_TEM"],
            "typical_value": [4.31],
            "ci_low": [2.40],
            "ci_high": [6.16],
            "pass": [True],
        }
    )

    report = build_one_page_report(recovered, auc_summary, ci_checks, pcvpc_path="pcvpc.png")

    assert "Estimated typical parameters vs paper" in report
    assert "Sirolimus:temsirolimus AUC0-inf ratio" in report
    assert "PASS" in report
