"""Forecast model for the GTA VI launch-day Twitch CCU bounty.

Method (Blend2): for each hour h = 2..24 the forecast is a log-normal
band around a median log-ratio r_h = ln(viewers_h / viewers_1), where
viewers_1 is the case's known hour-1 print. Two independent estimates of
the median are averaged:

  * K  -- a scale regression on clean comparables with a large hour 1
          (tier "comparable", hour-1 viewers above 10,000):
          r_h = a_h + b_h * (ln v1 - mean ln v1), fitted by least
          squares per hour, with sigma from the residual spread.
  * RB -- the empirical median log-ratio of the rolling-midnight,
          big launches (tier "comparable", rolling midnight unlock,
          hour-1 viewers above 50,000), with sigma from its
          inter-quartile range.

The reported median is the mean of the two medians. The reported sigma
is the larger of the two sigmas (floored at 0.30), widened by half the
absolute disagreement between the two medians, combined in quadrature.
Quantiles are v1 * exp(median + z_tau * sigma) for the standard normal
z values of the five contest levels.

Early-access-contaminated and weak / free-to-play launches are excluded
from every training set: their release-day shapes are distorted by
early-access peaks and live-service patterns and do not resemble a
rolling-midnight console launch. The raw pooled median profile is never
used on its own: it overstates the rise for large launches because the
profile shape depends strongly on hour-1 scale.

Fallbacks, in order: if the regression set has fewer than 8 launches or
the rolling-big set fewer than 3, use the scale regression alone on all
clean-tier comparables; if no usable comparable exists at all, use the
house persistence baseline. The target's own pre-release history is
never used for the median -- it sits orders of magnitude below launch
scale and carries no information about the release-day shape.

Only the Python standard library is used. There is no randomness, no
clock, and no iteration order that depends on hashing: every list that
feeds the output is sorted explicitly, so two runs are byte-identical.
"""

import math
import statistics
from datetime import datetime, timezone

# Quantile levels scored by the contest, in row order.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Standard normal quantiles at those levels.
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# Training-set cutoffs (viewers in hour 1).
REGRESSION_MIN_V1 = 10000
ROLLING_BIG_MIN_V1 = 50000

# Minimum training-set sizes for the blended model; below them the
# regression-only fallback is used instead.
MIN_REGRESSION_SET = 8
MIN_ROLLING_BIG_SET = 3

# Spread floors, in log units.
SIGMA_FLOOR_REGRESSION = 0.25
SIGMA_FLOOR_BLEND = 0.30
SIGMA_FALLBACK_ROLLING = 0.5

# Hours of the release day that are forecast.
FIRST_HOUR = 2
LAST_HOUR = 24

# Baseline persistence spread (house baseline), log units per root hour.
BASELINE_SIGMA = 0.15

# A launch needs hour 1 plus at least this many release-day hours to
# count as a usable profile.
MIN_PROFILE_HOURS = 20

# Output clamps.
MAX_VIEWERS = 1e9


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp; return None when unparsable."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _percentile(sorted_values, q):
    """Linear-interpolation percentile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def _tier_of(comparable):
    """Tier label carried in the unlock text (Tier: comparable)."""
    text = comparable.get("unlock") or ""
    if "Tier:" not in text:
        return "unknown"
    rest = text.split("Tier:", 1)[1]
    return rest.split(".", 1)[0].strip()


def _unlock_kind(comparable):
    """Unlock shape: rolling-midnight, simultaneous, or something else.

    Early-access-contaminated and weak launches are reported under
    their own kinds so callers can exclude them by tier anyway.
    """
    tier = _tier_of(comparable)
    if tier == "early-access-contaminated":
        return "early-access"
    if tier == "weak":
        return "weak"
    text = (comparable.get("unlock") or "").lower()
    if ("simultaneous" in text or "single global unlock" in text
            or "single simultaneous" in text):
        return "simultaneous"
    if "midnight local" in text or "rolling" in text:
        return "rolling-midnight"
    if "global unlock" in text:
        return "simultaneous"
    return "unverified"


def _release_day_series(comparable):
    """Map hour (1..24) to viewers for the comparable's release day."""
    day_start = _parse_time(comparable.get("day_start"))
    if day_start is None:
        return {}
    series = {}
    hourly = comparable.get("hourly") or []
    for row in hourly:
        if not isinstance(row, dict):
            continue
        moment = _parse_time(row.get("from"))
        viewers = row.get("viewers")
        if moment is None or not isinstance(viewers, (int, float)):
            continue
        if isinstance(viewers, bool) or viewers <= 0:
            continue
        delta = (moment - day_start).total_seconds()
        if 0 <= delta < 24 * 3600:
            hour = int(delta // 3600) + 1
            if hour not in series:
                series[hour] = float(viewers)
    return series


def build_profiles(comparables):
    """Turn raw comparable entries into deduplicated launch profiles.

    Each profile is a dict with the launch name, release date, its
    release-day hourly series, its tier and its unlock kind. Launches
    without an hour-1 print or with fewer than MIN_PROFILE_HOURS
    release-day hours are dropped. Duplicates (the same launch listed
    in several cases) collapse to the first occurrence in sorted order,
    so the result does not depend on input ordering.
    """
    profiles = {}
    ordered = sorted(
        (c for c in (comparables or []) if isinstance(c, dict)),
        key=lambda c: (str(c.get("name", "")),
                       str(c.get("day_start", ""))))
    for comparable in ordered:
        key = (str(comparable.get("name", "")),
               str(comparable.get("day_start", ""))[:10])
        if key in profiles:
            continue
        series = _release_day_series(comparable)
        if 1 not in series or len(series) < MIN_PROFILE_HOURS:
            continue
        profiles[key] = {
            "name": str(comparable.get("name", "")),
            "day": str(comparable.get("day_start", ""))[:10],
            "series": series,
            "tier": _tier_of(comparable),
            "unlock": _unlock_kind(comparable),
        }
    return [profiles[key] for key in sorted(profiles)]


def _hour1_viewers(case):
    """The case's known hour-1 print, or None when it is unusable."""
    known = case.get("known") or []
    for row in known:
        if isinstance(row, dict) and row.get("hour") == 1:
            value = row.get("viewers")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value) and value > 0:
                    return float(value)
    return None


def _ols(points):
    """Least-squares fit y = a + b*x; returns (a, b, residual sd).

    Points are (x, y) pairs. With fewer than two points, or with no
    spread in x, the slope is zero and the intercept is the mean of y.
    The residual spread is the population standard deviation.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0.0, SIGMA_FALLBACK_ROLLING
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    var_x = sum((p[0] - mean_x) ** 2 for p in points)
    if var_x <= 0:
        slope = 0.0
    else:
        cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
        slope = cov / var_x
    intercept = mean_y - slope * mean_x
    if n > 1:
        resid = [p[1] - (intercept + slope * p[0]) for p in points]
        spread = statistics.pstdev(resid)
    else:
        spread = SIGMA_FALLBACK_ROLLING
    return intercept, slope, spread


def _regression_forecast(v1, train, hour):
    """Scale-regression forecast for one hour.

    Fits r = ln(v_h / v_1) against centred ln v1 over the training
    launches that have hour h. When too few launches have that hour,
    the previous hour's fit is reused (walking back towards hour 1).
    Returns (median log-ratio, sigma) or None when nothing fits.
    """
    if v1 <= 0 or not train:
        return None
    log_v1s = [math.log(p["series"][1]) for p in train]
    centre = sum(log_v1s) / len(log_v1s)
    target_x = math.log(v1) - centre
    look_hour = hour
    while look_hour >= FIRST_HOUR:
        points = []
        for profile in train:
            series = profile["series"]
            if look_hour in series:
                points.append((math.log(series[1]) - centre,
                               math.log(series[look_hour] / series[1])))
        if len(points) >= 5:
            intercept, slope, spread = _ols(points)
            median = intercept + slope * target_x
            sigma = max(spread, SIGMA_FLOOR_REGRESSION)
            return median, sigma
        look_hour -= 1
    if len(train) >= 2:
        # No hour had enough points: fall back to a flat persistence
        # median with a wide spread rather than failing.
        return 0.0, 1.0
    return None


def _rolling_median(profile_set, hour):
    """Empirical median log-ratio and IQR sigma for one hour.

    Returns (median, sigma) or None when the set has no value at all
    for that hour. With fewer than four values the IQR is unreliable,
    so a fixed wide sigma is used instead.
    """
    values = sorted(
        math.log(p["series"][hour] / p["series"][1])
        for p in profile_set if hour in p["series"])
    if not values:
        return None
    median = _percentile(values, 0.5)
    if len(values) >= 4:
        iqr = _percentile(values, 0.75) - _percentile(values, 0.25)
        sigma = iqr / 1.349
    else:
        sigma = SIGMA_FALLBACK_ROLLING
    return median, sigma


def _round_half_up(value):
    """Round to 2 decimals, halves away from zero, as a float."""
    scaled = math.floor(value * 100.0 + 0.5) / 100.0
    return scaled


def _clean_quantiles(raw):
    """Clamp one hour's five raw quantiles into a valid, ordered row."""
    cleaned = []
    previous = 1.0
    for value in raw:
        if not math.isfinite(value):
            value = previous
        value = max(1.0, min(MAX_VIEWERS, value))
        value = _round_half_up(value)
        if value < previous:
            value = previous
        cleaned.append(value)
        previous = value
    return cleaned


def baseline_forecast(v1):
    """House persistence baseline: flat median, widening log spread."""
    anchor = v1 if (v1 is not None and math.isfinite(v1) and v1 >= 1) else 1.0
    rows = {}
    for hour in range(FIRST_HOUR, LAST_HOUR + 1):
        spread = BASELINE_SIGMA * math.sqrt(hour - 1)
        raw = [anchor * math.exp(z * spread) for z in Z_VALUES]
        rows[hour] = _clean_quantiles(raw)
    return rows


def forecast_from_profiles(v1, profiles):
    """Blend2 forecast for hour-1 print v1 given training profiles.

    Returns {hour: [q05, q25, q50, q75, q95]} for hours 2..24.
    """
    clean = [p for p in profiles if p["tier"] == "comparable"]
    reg_set = [p for p in clean if p["series"][1] > REGRESSION_MIN_V1]
    rolling_big = [p for p in clean
                   if p["unlock"] == "rolling-midnight"
                   and p["series"][1] > ROLLING_BIG_MIN_V1]
    rolling_all = [p for p in clean if p["unlock"] == "rolling-midnight"]

    use_blend = (len(reg_set) >= MIN_REGRESSION_SET
                 and len(rolling_big) >= MIN_ROLLING_BIG_SET)

    rows = {}
    for hour in range(FIRST_HOUR, LAST_HOUR + 1):
        if use_blend:
            reg = _regression_forecast(v1, reg_set, hour)
            roll = _rolling_median(rolling_big, hour)
            if roll is None:
                roll = _rolling_median(rolling_all, hour)
            if reg is None or roll is None:
                # One leg failed for this hour: fall back to whichever
                # leg is available, widened; both missing means the
                # regression-only path below.
                if reg is not None:
                    median, sigma = reg
                elif roll is not None:
                    median, sigma = roll
                else:
                    rows[hour] = baseline_forecast(v1)[hour]
                    continue
                raw = [v1 * math.exp(median + z * sigma) for z in Z_VALUES]
                rows[hour] = _clean_quantiles(raw)
                continue
            med_k, sig_k = reg
            med_rb, sig_rb = roll
            median = (med_k + med_rb) / 2.0
            base = max(sig_k, sig_rb, SIGMA_FLOOR_BLEND)
            sigma = math.sqrt(base * base
                              + (0.5 * abs(med_k - med_rb)) ** 2)
            raw = [v1 * math.exp(median + z * sigma) for z in Z_VALUES]
            rows[hour] = _clean_quantiles(raw)
        else:
            reg = _regression_forecast(v1, clean, hour)
            if reg is None:
                rows[hour] = baseline_forecast(v1)[hour]
                continue
            median, sigma = reg
            raw = [v1 * math.exp(median + z * sigma) for z in Z_VALUES]
            rows[hour] = _clean_quantiles(raw)
    return rows


def forecast_case(case):
    """Full forecast for one parsed case dict: {hour: [5 quantiles]}."""
    v1 = _hour1_viewers(case)
    if v1 is None:
        return baseline_forecast(1.0)
    profiles = build_profiles(case.get("comparables") or [])
    if not profiles:
        return baseline_forecast(v1)
    return forecast_from_profiles(v1, profiles)
