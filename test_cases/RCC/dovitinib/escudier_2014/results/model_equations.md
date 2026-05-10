# Dovitinib Aggregate PK/PD Model Equations

## PK Model

Gut amount: `A_gut`, central amount: `A_c`, peripheral amount: `A_p`.

`C = A_c / Vc`

`CL(t) = CL_day1 + (CL_day15 - CL_day1) * (1 - exp(-kaut * t))`

`Vmax(t) = CL(t) * Km`

`dA_gut/dt = -Ka * A_gut`

`dA_c/dt = F * Ka * A_gut - (Q/Vc) * A_c + (Q/Vp) * A_p - Vmax(t) * C / (Km + C)`

`dA_p/dt = (Q/Vc) * A_c - (Q/Vp) * A_p`

The model is parameterized in apparent oral terms because absolute oral
bioavailability is not identifiable from aggregate oral summaries.

## Residual Error

`Y = IPRED * (1 + eps_prop) + eps_add`

## PD Turnover Model

For inhibition markers:

`I(C) = Emax * C / (EC50 + C)`

`dR/dt = kout * (1 - I(C) + escape(t) - R)`

For stimulation markers:

`S(C) = Emax * C / (EC50 + C)`

`dR/dt = kout * (1 + S(C) + escape(t) - R)`

`escape(t) = escape_max * (1 - exp(-escape_k * t))`

Reported PD output is `(R - 1) * 100`, the percent change from baseline.
