"""Indirect-response tumor growth and PFS model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

DEFAULT_RESPONDER_FRACTION = 0.35


class TumorGrowthModel:
    """Tumor log-size model driven by pS6K1 inhibition."""

    def __init__(
        self,
        lambda_growth: float,
        lambda_kill: float,
        biomarker_model,
        times: NDArray[np.float64],
        pS6K1_array: NDArray[np.float64],
        pS6K1_0: float,
        resistance_tau_days: float | None = None,
    ):
        self.lambda_growth = float(lambda_growth)
        self.lambda_kill = float(lambda_kill)
        self.biomarker_model = biomarker_model
        self.times = np.asarray(times, dtype=float)
        self.pS6K1_array = np.asarray(pS6K1_array, dtype=float)
        self.pS6K1_0 = float(pS6K1_0)
        self.cycle_length = float(self.times[-1])
        self.resistance_tau_days = resistance_tau_days

    def inhibition_at(self, t_day: float) -> float:
        """Interpolate weekly steady-state fractional pS6K1 inhibition at a day."""

        if self.cycle_length <= 0.0:
            return 0.0
        t_cycle = float(np.mod(t_day, self.cycle_length))
        pS6K1_t = float(np.interp(t_cycle, self.times, self.pS6K1_array))
        return float(np.clip(1.0 - (pS6K1_t / self.pS6K1_0), 0.0, 1.0))

    def growth_rate(self, t: float, L: NDArray[np.float64]) -> list[float]:
        """dL/dt = lambda_growth - lambda_kill times E(t)."""

        _ = L
        e_t = self.inhibition_at(t)
        if self.resistance_tau_days is None:
            effective_kill = self.lambda_kill
        else:
            effective_kill = self.lambda_kill * np.exp(-t / self.resistance_tau_days)
        dL_dt = self.lambda_growth - effective_kill * e_t
        return [dL_dt]

    def simulate_pfs(self, L_0: float, t_max: float = 365.0) -> tuple[float, bool]:
        """Integrate until a 20% increase from baseline or administrative censoring."""

        L_progression = float(L_0 + np.log(1.20))

        def progression_event(t: float, y: NDArray[np.float64]) -> float:
            _ = t
            return float(y[0] - L_progression)

        progression_event.terminal = True  # type: ignore[attr-defined]
        progression_event.direction = 1.0  # type: ignore[attr-defined]

        solution = solve_ivp(
            self.growth_rate,
            (0.0, float(t_max)),
            [float(L_0)],
            events=progression_event,
            dense_output=False,
            method="RK45",
            max_step=1.0,
            rtol=1e-6,
            atol=1e-8,
        )
        if not solution.success:
            raise RuntimeError(f"Tumor ODE solve failed: {solution.message}")
        if solution.t_events and solution.t_events[0].size > 0:
            return float(solution.t_events[0][0]), True
        return float(t_max), False


class MixtureResponderModel:
    """Responder/resistant tumor mixture model.

    Responders use the drug-effect tumor ODE. Resistant tumors use baseline
    drug-independent log-linear growth.
    """

    def __init__(self, responder_fraction: float = DEFAULT_RESPONDER_FRACTION):
        if not 0.0 <= responder_fraction <= 1.0:
            raise ValueError("responder_fraction must be between 0 and 1")
        self.pi = float(responder_fraction)

    def sample_status(self, rng: np.random.Generator) -> bool:
        """Return True if the individual is an S6K1-sensitive responder."""

        return bool(rng.random() < self.pi)

    def simulate_pfs_mixture(
        self,
        *,
        L_0: float,
        lambda_growth: float,
        lambda_kill_0: float,
        biomarker_model,
        times: NDArray[np.float64],
        pS6K1_array: NDArray[np.float64],
        pS6K1_0: float,
        rng: np.random.Generator,
        tau_resist_days: float | None,
        t_max: float = 365.0,
    ) -> tuple[float, bool, bool]:
        """Simulate PFS after sampling responder status.

        Returns PFS days, event flag, and responder flag.
        """

        is_responder = self.sample_status(rng)
        if is_responder:
            tumor = TumorGrowthModel(
                lambda_growth,
                lambda_kill_0,
                biomarker_model,
                times,
                pS6K1_array,
                pS6K1_0,
                resistance_tau_days=tau_resist_days,
            )
            pfs_days, progressed = tumor.simulate_pfs(L_0, t_max=t_max)
            return pfs_days, progressed, True

        pfs_days = float(np.log(1.20) / lambda_growth)
        if pfs_days <= t_max:
            return pfs_days, True, False
        return float(t_max), False, False
