# GTA VI Launch-Day Twitch Forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aboylabs/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aboylabs/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day, 19 November 2026 (UTC), hours 2–24, as
five quantiles per hour (levels 0.05, 0.25, 0.5, 0.75, 0.95).

## How it works

`main.py` is a thin entry point: it reads every `*.in` case in the
input directory, calls the model in `src/model.py`, and writes one
`<id>.out` JSON file per case. Python 3.13, standard library only, no
network, no randomness, no clock — two runs over the same input are
byte-identical.

The model ("Blend2") conditions the whole forecast on the case's known
hour-1 print `v1` and on the launch profiles in that case's
`comparables`. For each hour `h = 2..24` it works in log-ratios
`r_h = ln(viewers_h / viewers_1)` and combines two median estimates:

1. **Scale regression (K).** An ordinary least-squares fit of `r_h`
   against centred `ln v1`, trained on clean-tier comparables with
   hour-1 viewers above 10,000. Release-day shapes depend strongly on
   hour-1 scale (correlation about −0.94 by hour 24), so a pooled
   median would overstate the rise for a launch as big as GTA VI.
   Spread comes from the residual standard deviation (floored at 0.25).
2. **Rolling-big empirical median (RB).** The median log-ratio of the
   rolling-midnight, big launches — tier "comparable", midnight-local
   unlock, hour-1 viewers above 50,000 — the closest analogues to GTA
   VI's own unlock. Spread comes from the inter-quartile range
   (divided by 1.349; a fixed 0.5 when fewer than four values).

The reported median is the average of the two medians. The reported
spread is the larger of the two spreads (floored at 0.30), widened in
quadrature by half the absolute disagreement between the two medians,
so the band grows exactly where the two views diverge — most of all
for hour-1 prints far outside the training range. Quantiles are
`v1 * exp(median + z_tau * sigma)`.

Early-access-contaminated and weak / free-to-play launches are excluded
from all training sets, and the raw pooled median profile is never used
on its own: on leave-one-out backtests over the 37 comparable launches
it scores worse than the house baseline for big launches. The target
launch's own pre-release history is not used for the median — it sits
orders of magnitude below launch scale.

## Fallbacks

If the regression set has fewer than 8 launches or the rolling-big set
fewer than 3, the model falls back to the scale regression alone on all
clean-tier comparables. If no usable comparable exists at all, it falls
back to the house persistence baseline
(`q = v1 * exp(z_tau * 0.15 * sqrt(h - 1))`). A failure while handling
one case never aborts the run: `main.py` catches it per case and still
writes a valid baseline forecast for that case.

## Backtest

Leave-one-out over the 37 comparable launches with known release-day
profiles (each launch forecast from all the others, scored with the
contest's pinball loss on the log scale):

| model | loss, rolling-big (n=6) | loss, v1 > 50k (n=20) |
|---|---|---|
| house baseline | 0.250 | 0.251 |
| Blend2 (this entry) | 0.188 | 0.236 |

See `../analysis/` (outside this repository) for the validation
scripts and the full analysis.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
