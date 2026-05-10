from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .model import PDParameters, PKParameters, population_prediction_grid, simulate_pd, simulate_pk


sns.set_theme(style="whitegrid", context="notebook")


def save_observed_vs_predicted(sparse: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(
        data=sparse,
        x="ipred_concentration_ng_ml",
        y="observed_concentration_ng_ml",
        hue="occasion",
        ax=ax,
    )
    lim = max(sparse["ipred_concentration_ng_ml"].max(), sparse["observed_concentration_ng_ml"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--")
    ax.set(xlabel="Individual prediction (ng/mL)", ylabel="Observed pseudo-concentration (ng/mL)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_residuals_vs_predicted(sparse: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.scatterplot(data=sparse, x="ipred_concentration_ng_ml", y="cwres", hue="occasion", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(2.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-2.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Individual prediction (ng/mL)", ylabel="CWRES")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_random_effects(eta: pd.DataFrame, path: Path) -> None:
    g = sns.displot(
        data=eta,
        x="eta",
        col="parameter",
        col_wrap=4,
        bins=18,
        facet_kws={"sharex": False, "sharey": False},
        height=2.2,
    )
    g.set_axis_labels("ETA", "Count")
    g.figure.tight_layout()
    g.figure.savefig(path, dpi=180)
    plt.close(g.figure)


def save_concentration_profiles(params: PKParameters, population_grid: pd.DataFrame, path: Path) -> None:
    typical = simulate_pk(params, end_h=360.0, dt_h=0.1)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = {"day1": "#1f77b4", "day15": "#d62728"}
    for occasion, start_h in [("day1", 0.0), ("day15", 14.0 * 24.0)]:
        window = typical[(typical["time_h"] >= start_h) & (typical["time_h"] <= start_h + 24.0)].copy()
        window["time_after_dose_h"] = window["time_h"] - start_h
        quant = (
            population_grid[population_grid["occasion"] == occasion]
            .groupby("time_after_dose_h")["concentration_ng_ml"]
            .quantile([0.05, 0.95])
            .unstack()
            .reset_index()
        )
        ax.fill_between(
            quant["time_after_dose_h"],
            quant[0.05],
            quant[0.95],
            color=colors[occasion],
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            window["time_after_dose_h"],
            window["concentration_ng_ml"],
            color=colors[occasion],
            label=f"{occasion} typical",
            linewidth=2,
        )
    ax.set(xlabel="Time after dose (h)", ylabel="Dovitinib concentration (ng/mL)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_vpc(population_grid: pd.DataFrame, paper_targets: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, occasion in zip(axes, ["day1", "day15"]):
        subset = population_grid[population_grid["occasion"] == occasion]
        quant = (
            subset.groupby("time_after_dose_h")["concentration_ng_ml"]
            .quantile([0.05, 0.5, 0.95])
            .unstack()
            .reset_index()
        )
        ax.fill_between(quant["time_after_dose_h"], quant[0.05], quant[0.95], color="#8ecae6", alpha=0.35)
        ax.plot(quant["time_after_dose_h"], quant[0.5], color="#023047", linewidth=2, label="Median")
        target = paper_targets[(paper_targets["occasion"] == occasion) & (paper_targets["metric"] == "Cmax")].iloc[0]
        ax.scatter([6.0], [target["value"]], color="#d00000", zorder=4, label="Paper Cmax")
        ax.set_title(occasion)
        ax.set_xlabel("Time after dose (h)")
        ax.set_xlim(0, 24)
    axes[0].set_ylabel("Concentration (ng/mL)")
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_individual_predictions(sparse: pd.DataFrame, params: PKParameters, path: Path) -> None:
    selected = sparse["patient_id"].drop_duplicates().head(12).tolist()
    typical = simulate_pk(params, end_h=360.0, dt_h=0.2)
    fig, axes = plt.subplots(3, 4, figsize=(12.0, 7.8), sharex=True, sharey=True)
    for ax, patient_id in zip(axes.flat, selected):
        obs = sparse[sparse["patient_id"] == patient_id]
        for occasion, start_h, color in [("day1", 0.0, "#1f77b4"), ("day15", 14.0 * 24.0, "#d62728")]:
            window = typical[(typical["time_h"] >= start_h) & (typical["time_h"] <= start_h + 24.0)].copy()
            ax.plot(window["time_h"] - start_h, window["concentration_ng_ml"], color=color, alpha=0.9)
            point = obs[obs["occasion"] == occasion]
            ax.scatter(point["time_after_dose_h"], point["observed_concentration_ng_ml"], color=color, s=18)
        ax.set_title(f"ID {patient_id}", fontsize=9)
        ax.set_xlim(0, 24)
    fig.supxlabel("Time after dose (h)")
    fig.supylabel("Concentration (ng/mL)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pd_response(pk_params: PKParameters, pd_parameters: pd.DataFrame, pd_targets: pd.DataFrame, path: Path) -> None:
    pk_profile = simulate_pk(pk_params, end_h=26.0 * 24.0, dt_h=0.5)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for marker, group in pd_parameters.groupby("marker"):
        values = {row.parameter: row.estimate for row in group.itertuples(index=False)}
        mode = "stimulation" if marker in {"FGF23", "VEGF"} else "inhibition"
        baseline = values.get("baseline", 100.0)
        baseline_unit = "pg/mL" if marker == "FGF23" else "percent"
        params = PDParameters(
            marker=marker,
            mode=mode,
            emax=values["emax"],
            ec50_ng_ml=values["ec50_ng_ml"],
            kout_h=values["kout_h"],
            escape_max=values.get("escape_max", 0.0),
            escape_k_h=values.get("escape_k_h", 0.0),
            baseline=baseline,
            baseline_unit=baseline_unit,
        )
        profile = simulate_pd(pk_profile, params, end_h=26.0 * 24.0, dt_h=1.0)
        ax.plot(profile["day"], profile["percent_change_from_baseline"], linewidth=2, label=marker)
    sns.scatterplot(
        data=pd_targets,
        x="day",
        y="percent_change_from_baseline",
        hue="marker",
        style="marker",
        s=70,
        legend=False,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="Study day", ylabel="Change from baseline (%)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
