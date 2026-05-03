#!/usr/bin/env python3
"""Loss visualization script for pixel-space training logs."""

import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

DEFAULT_CSVS = {
    "sprint": "pixel-space-stable-rms-baseline/ckpts/loss_log.csv",
    "classic":      "pixel-space-stable-rms/ckpts/loss_log.csv",
}


def remove_outliers(values, iqr_factor=3.0):
    """Replace outliers (beyond iqr_factor * IQR from Q1/Q3) with NaN."""
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    mask = (values < q1 - iqr_factor * iqr) | (values > q3 + iqr_factor * iqr)
    cleaned = values.copy().astype(float)
    cleaned[mask] = np.nan
    n = mask.sum()
    if n:
        print(f"  Removed {n} outlier(s) from series")
    return cleaned


def gsmooth(values, sigma):
    """Gaussian smooth, ignoring NaNs via interpolation."""
    if sigma <= 0:
        return values
    x = values.copy()
    nans = np.isnan(x)
    if nans.all():
        return x
    # fill NaNs by linear interpolation before smoothing
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    smoothed = gaussian_filter1d(x, sigma=sigma)
    smoothed[nans] = np.nan
    return smoothed


def _power_law(x, a, b, c):
    """Power-law model: a * x^b + c  (x must be > 0)."""
    return  a / (x + b) + c #a * np.exp(-b * x) + c


def power_law_fit_forecast(steps, values, forecast_steps=0):
    """Fit a power law  a*x^b+c  to non-NaN data and optionally forecast.

    x is normalised to [1, 2] internally so it is always positive and
    well-conditioned for fractional exponents.

    Returns
    -------
    fit_steps  : 1-D array covering the training range
    fit_values : model evaluated over fit_steps
    fore_steps : 1-D array covering the forecast range (empty if forecast_steps=0)
    fore_values: model evaluated over fore_steps
    popt       : (a, b, c) best-fit parameters, or None on failure
    """
    mask = ~np.isnan(values)
    if mask.sum() < 4:
        empty = np.array([])
        return steps, np.full_like(steps, np.nan), empty, empty, None

    x_min, x_max = steps[mask].min(), steps[mask].max()
    # map to [1, 2] so x is always positive and exponentiation is stable
    x_norm = 1.0 + (steps[mask] - x_min) / (x_max - x_min)
    y      = values[mask]

    # Initial guess: a ~ range (negative for decay), b ~ -0.5, c ~ min
    y_range = y.max() - y.min()
    p0 = (-y_range, -0.5, y.max())

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _power_law, x_norm, y, p0=p0,
                bounds=([-np.inf, -np.inf, -np.inf], [np.inf, 0, np.inf]),
                maxfev=20_000,
            )
    except Exception as exc:
        print(f"  power_law fit failed: {exc}")
        empty = np.array([])
        return steps, np.full_like(steps, np.nan), empty, empty, None

    def _eval(s):
        xn = 1.0 + (s - x_min) / (x_max - x_min)
        return _power_law(xn, *popt)

    fit_values = _eval(steps)

    if forecast_steps > 0:
        step_size   = (steps[-1] - steps[0]) / (len(steps) - 1) if len(steps) > 1 else 1
        fore_steps  = np.arange(1, forecast_steps + 1) * step_size + steps[-1]
        fore_values = _eval(fore_steps)
    else:
        fore_steps  = np.array([])
        fore_values = np.array([])

    return steps, fit_values, fore_steps, fore_values, popt


def _add_power_overlay(ax, steps, values, forecast_steps, color, label_prefix):
    """Compute and draw power-law fit + forecast on *ax*."""
    fit_s, fit_v, fore_s, fore_v, popt = power_law_fit_forecast(steps, values, forecast_steps)
    if popt is None:
        return
    a, b, c = popt
    fit_label = f"{label_prefix} power law  (b={b:.3f})"
    ax.plot(fit_s, fit_v, color=color, linewidth=1.4, linestyle="--", label=fit_label)
    if len(fore_s):
        ax.plot(fore_s, fore_v, color=color, linewidth=1.2, linestyle=":",
                label=f"{label_prefix} forecast (+{forecast_steps:,} steps)")
        ax.axvspan(fore_s[0], fore_s[-1], alpha=0.06, color=color)


def main():
    parser = argparse.ArgumentParser(description="Visualize training loss CSV(s).")
    parser.add_argument(
        "csv", nargs="*",
        help=(
            "One or more CSV paths, optionally labelled as  label=path.csv. "
            "Defaults to the built-in dict when omitted."
        ),
    )
    parser.add_argument("--sigma", type=float, default=150, metavar="S",
                        help="Gaussian smoothing sigma (default: 150, 0 = off)")
    parser.add_argument("--iqr", type=float, default=3.0, metavar="F",
                        help="IQR outlier removal factor (default: 3.0)")
    parser.add_argument("--forecast", type=int, default=0, metavar="N",
                        help="Forecast N steps beyond the last recorded step (default: 0)")
    parser.add_argument("--no-poly", action="store_true",
                        help="Disable power-law fit / forecast overlay")
    parser.add_argument("--out", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    # ── Build label→path mapping ───────────────────────────────────────────────
    if args.csv:
        run_map = {}
        for entry in args.csv:
            if "=" in entry:
                label, path = entry.split("=", 1)
            else:
                label = entry
            run_map[label] = path
    else:
        run_map = DEFAULT_CSVS

    # ── Load all CSVs ─────────────────────────────────────────────────────────
    runs = {}
    for label, path in run_map.items():
        df = pd.read_csv(path)
        print(f"[{label}] Loaded {len(df):,} rows  |  steps {df['step'].min()}–{df['step'].max()}")
        runs[label] = {
            "steps": df["step"].to_numpy(dtype=float),
            "loss":  remove_outliers(df["loss"].to_numpy(dtype=float), args.iqr),
            "mse":   remove_outliers(df["mse"].to_numpy(dtype=float),  args.iqr),
            "dino":  remove_outliers(df["dino"].to_numpy(dtype=float), args.iqr),
            "lr":    df["lr"].to_numpy(dtype=float),
        }

    # Colour palette — one hue family per run, cycling if needed
    _palettes = [
        {"raw": "#aaaaaa", "line": "#e05c5c", "fit": "#ff9944"},  # red/orange
        {"raw": "#aaaaaa", "line": "#5c9ee0", "fit": "#44aaff"},  # blue
        {"raw": "#aaaaaa", "line": "#5ce07a", "fit": "#44dd88"},  # green
        {"raw": "#aaaaaa", "line": "#c97de0", "fit": "#dd88ff"},  # purple
        {"raw": "#aaaaaa", "line": "#e0c05c", "fit": "#ffdd44"},  # yellow
    ]
    palette_for = {lbl: _palettes[i % len(_palettes)] for i, lbl in enumerate(runs)}

    s = args.sigma
    show_poly = not args.no_poly
    forecast  = args.forecast

    all_steps = np.concatenate([r["steps"] for r in runs.values()])
    global_step_min = all_steps.min()
    global_step_max = all_steps.max()

    def xlim_end(base_end):
        if show_poly and forecast > 0:
            step_size = (all_steps[-1] - all_steps[0]) / max(len(all_steps) - 1, 1)
            return base_end + forecast * step_size
        return base_end

    fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=False)
    fig.suptitle("Pixel-Space Training Loss", fontsize=14, fontweight="bold")

    panel_cfg = [
        (axes[0], "loss", "Loss",      "Total Loss"),
        (axes[1], "mse",  "MSE Loss",  "MSE Component"),
        (axes[2], "dino", "DINO Loss", "DINO Component"),
    ]

    for ax, key, ylabel, title in panel_cfg:
        for label, run in runs.items():
            pal   = palette_for[label]
            steps = run["steps"]
            vals  = run[key]
            ax.plot(steps, vals, color=pal["raw"], alpha=0.15, linewidth=0.5)
            ax.plot(steps, gsmooth(vals, s),
                    linewidth=1.8, label=f"{label} (σ={s})", color=pal["line"])
            if show_poly:
                _add_power_overlay(ax, steps, vals, forecast, pal["fit"], label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(global_step_min, xlim_end(global_step_max))

    # ── Panel 4: learning rate ─────────────────────────────────────────────────
    ax = axes[3]
    for label, run in runs.items():
        pal = palette_for[label]
        ax.plot(run["steps"], run["lr"], linewidth=1.4,
                color=pal["line"], label=label)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title("Learning Rate Schedule")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(global_step_min, global_step_max)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
