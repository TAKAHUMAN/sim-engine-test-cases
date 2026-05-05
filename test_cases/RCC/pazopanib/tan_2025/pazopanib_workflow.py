"""Python plotting workflow for pazopanib PK/PD dose comparison.

Run from repo root:

    python microservice/pazopanib_workflow.py pazopanib_plots
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from microservice.pazopanib_pkpd import PazopanibPKPDConfig, simulate_population_dose_scenario


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_dose_comparison(
    *,
    doses_mg: tuple[float, ...] = (400.0, 600.0, 800.0),
    n_patients: int = 250,
    t_end_days: int = 365,
    seed: int = 11,
) -> dict[float, dict[str, np.ndarray]]:
    cfg = PazopanibPKPDConfig()
    out: dict[float, dict[str, np.ndarray]] = {}
    for i, d in enumerate(doses_mg):
        out[d] = simulate_population_dose_scenario(
            cfg,
            dose_mg=d,
            n_patients=n_patients,
            t_end_days=t_end_days,
            seed=seed + i,
        )
    return out


def summarize_results(results: dict[float, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d, r in sorted(results.items()):
        rows.append(
            {
                "dose_mg": d,
                "cmin_day365_p50_mg_L": float(r["cmin_p50"][-1]),
                "cmin_day365_p05_mg_L": float(r["cmin_p05"][-1]),
                "cmin_day365_p95_mg_L": float(r["cmin_p95"][-1]),
                "pct_cmin_ge_20_5": float(r["pct_cmin_ge_20_5"][0]),
                "pct_cmin_gt_34": float(r["pct_cmin_gt_34"][0]),
                "toxicity_prob_day365_p50": float(r["tox_p50"][-1]),
            }
        )
    return rows


def save_pazopanib_comparison_pngs(
    output_dir: str | Path,
    *,
    doses_mg: tuple[float, ...] = (400.0, 600.0, 800.0),
    n_patients: int = 250,
    t_end_days: int = 365,
    seed: int = 11,
) -> list[Path]:
    """Generate Python plots comparing dose scenarios."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    results = run_dose_comparison(doses_mg=doses_mg, n_patients=n_patients, t_end_days=t_end_days, seed=seed)
    days = next(iter(results.values()))["day"]
    colors = {400.0: "C0", 600.0: "C1", 800.0: "C3"}

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for d, r in sorted(results.items()):
        c = colors.get(float(d), None)
        ax1.fill_between(days, r["cmin_p05"], r["cmin_p95"], alpha=0.15, color=c)
        ax1.plot(days, r["cmin_p50"], lw=2, color=c, label=f"{int(d)} mg")
    ax1.axhline(20.0, ls="--", color="darkgreen", lw=1.2, label="Target lower (20)")
    ax1.axhline(34.0, ls="--", color="red", lw=1.2, label="Toxicity threshold (34)")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Cmin (mg/L)")
    ax1.set_title("Pazopanib Cmin trajectories (median and 90% PI)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    fig1.tight_layout()
    p1 = out_dir / "pazopanib_cmin_dose_comparison.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for d, r in sorted(results.items()):
        c = colors.get(float(d), None)
        ax2.plot(days, r["tox_p50"], lw=2, color=c, label=f"{int(d)} mg")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Cumulative liver toxicity probability")
    ax2.set_title("Predicted CTCAE>=2 liver toxicity risk")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    p2 = out_dir / "pazopanib_toxicity_probability_comparison.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for d, r in sorted(results.items()):
        c = colors.get(float(d), None)
        axes[0].plot(days, r["sld_mrcc_p50"], lw=2, color=c, label=f"{int(d)} mg")
        axes[1].plot(days, r["sld_sts_p50"], lw=2, color=c, label=f"{int(d)} mg")
    axes[0].set_title("mRCC tumor dynamics (SLD)")
    axes[1].set_title("STS tumor dynamics (SLD)")
    for ax in axes:
        ax.set_xlabel("Day")
        ax.set_ylabel("SLD (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig3.tight_layout()
    p3 = out_dir / "pazopanib_tumor_dynamics_comparison.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    rows = summarize_results(results)
    lines = [
        "dose_mg,cmin_day365_p50_mg_L,cmin_day365_p05_mg_L,cmin_day365_p95_mg_L,pct_cmin_ge_20_5,pct_cmin_gt_34,toxicity_prob_day365_p50"
    ]
    for r in rows:
        lines.append(
            f"{int(r['dose_mg'])},{r['cmin_day365_p50_mg_L']:.4f},{r['cmin_day365_p05_mg_L']:.4f},"
            f"{r['cmin_day365_p95_mg_L']:.4f},{r['pct_cmin_ge_20_5']:.2f},{r['pct_cmin_gt_34']:.2f},{r['toxicity_prob_day365_p50']:.4f}"
        )
    csv_path = out_dir / "pazopanib_dose_comparison_summary.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [p1, p2, p3, csv_path]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pazopanib_plots")
    paths = save_pazopanib_comparison_pngs(out)
    for p in paths:
        print(p.resolve())
