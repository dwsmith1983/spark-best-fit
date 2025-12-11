"""Visualization utilities for fitted distributions."""

from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

if TYPE_CHECKING:
    from .config import PlotConfig
    from .results import DistributionFitResult


def plot_distribution(
    result: "DistributionFitResult",
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    config: "PlotConfig",
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "Density",
    save_path: Optional[str] = None,
) -> Tuple:
    """Plot fitted distribution against data histogram.

    Creates a matplotlib figure showing the data histogram and the
    fitted probability density function overlaid.

    Args:
        result: Fitted distribution result
        y_hist: Histogram density values
        x_hist: Histogram bin centers
        config: Plot configuration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure

    Returns:
        Tuple of (figure, axis)

    Example:
        >>> from spark_dist_fit import DistributionFitter, PlotConfig
        >>> fitter = DistributionFitter()
        >>> results = fitter.fit(df, 'value')
        >>> best = results.best(n=1)[0]
        >>>
        >>> # Plot with defaults
        >>> fitter.plot(best)
        >>>
        >>> # Custom plot config
        >>> plot_config = PlotConfig(figsize=(16, 10), dpi=300)
        >>> fitter.plot(best, config=plot_config, title='Distribution Fit')
    """
    # Get scipy distribution and parameters
    dist = getattr(st, result.distribution)
    params = result.parameters

    # Extract shape, loc, scale
    arg = params[:-2] if len(params) > 2 else ()
    loc = params[-2]
    scale = params[-1]

    # Generate smooth x values for PDF
    try:
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale)
    except (ValueError, RuntimeError, FloatingPointError):
        # Fallback if ppf fails
        start = x_hist.min()
        end = x_hist.max()

    # Ensure valid range
    if not np.isfinite(start):
        start = x_hist.min()
    if not np.isfinite(end):
        end = x_hist.max()

    x_pdf = np.linspace(start, end, 1000)
    y_pdf = dist.pdf(x_pdf, *arg, loc=loc, scale=scale)

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize)

    # Plot PDF
    ax.plot(
        x_pdf,
        y_pdf,
        "r-",
        lw=config.pdf_linewidth,
        label="Fitted PDF",
        zorder=3,
    )

    # Plot histogram
    if config.show_histogram:
        # Convert histogram density to bar plot
        bin_width = x_hist[1] - x_hist[0] if len(x_hist) > 1 else 1.0
        ax.bar(
            x_hist,
            y_hist,
            width=bin_width * 0.9,
            alpha=config.histogram_alpha,
            label="Data Histogram",
            color="skyblue",
            edgecolor="navy",
            linewidth=0.5,
            zorder=2,
        )

    # Format parameter string
    param_names = (dist.shapes + ", loc, scale").split(", ") if dist.shapes else ["loc", "scale"]
    param_str = ", ".join([f"{k}={v:.4f}" for k, v in zip(param_names, params)])

    dist_title = f"{result.distribution}({param_str})"
    sse_str = f"SSE: {result.sse:.6f}"

    if result.aic is not None and result.bic is not None:
        metrics_str = f"{sse_str}, AIC: {result.aic:.2f}, BIC: {result.bic:.2f}"
    else:
        metrics_str = sse_str

    # Set title
    full_title = f"{title}\n{dist_title}\n{metrics_str}" if title else f"{dist_title}\n{metrics_str}"

    ax.set_title(full_title, fontsize=config.title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=config.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=config.label_fontsize)

    # Configure legend
    ax.legend(fontsize=config.legend_fontsize, loc="best", framealpha=0.9)

    # Configure grid
    ax.grid(alpha=config.grid_alpha, linestyle="--", linewidth=0.5, zorder=1)

    # Improve layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, format=config.save_format, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax


def plot_comparison(
    results: list,
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    config: "PlotConfig",
    title: str = "Distribution Comparison",
    xlabel: str = "Value",
    ylabel: str = "Density",
    save_path: Optional[str] = None,
) -> Tuple:
    """Plot multiple fitted distributions for comparison.

    Args:
        results: List of DistributionFitResult objects
        y_hist: Histogram density values
        x_hist: Histogram bin centers
        config: Plot configuration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure

    Returns:
        Tuple of (figure, axis)

    Example:
        >>> from spark_dist_fit.plotting import plot_comparison
        >>> top_3 = results.best(n=3)
        >>> plot_comparison(top_3, y_hist, x_hist, plot_config)
    """
    if not results:
        raise ValueError("Must provide at least one result to plot")

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize)

    # Plot histogram
    if config.show_histogram:
        bin_width = x_hist[1] - x_hist[0] if len(x_hist) > 1 else 1.0
        ax.bar(
            x_hist,
            y_hist,
            width=bin_width * 0.9,
            alpha=config.histogram_alpha,
            label="Data Histogram",
            color="skyblue",
            edgecolor="navy",
            linewidth=0.5,
            zorder=1,
        )

    # Define colors for multiple distributions
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot each distribution
    for i, result in enumerate(results):
        dist = getattr(st, result.distribution)
        params = result.parameters

        arg = params[:-2] if len(params) > 2 else ()
        loc = params[-2]
        scale = params[-1]

        # Generate PDF
        try:
            start = dist.ppf(0.01, *arg, loc=loc, scale=scale)
            end = dist.ppf(0.99, *arg, loc=loc, scale=scale)
        except (ValueError, RuntimeError, FloatingPointError):
            start = x_hist.min()
            end = x_hist.max()

        if not np.isfinite(start):
            start = x_hist.min()
        if not np.isfinite(end):
            end = x_hist.max()

        x_pdf = np.linspace(start, end, 1000)
        y_pdf = dist.pdf(x_pdf, *arg, loc=loc, scale=scale)

        # Plot with label
        label = f"{result.distribution} (SSE={result.sse:.4f})"
        ax.plot(
            x_pdf,
            y_pdf,
            lw=config.pdf_linewidth,
            label=label,
            color=colors[i],
            zorder=2 + i,
        )

    # Configure plot
    ax.set_title(title, fontsize=config.title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=config.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=config.label_fontsize)
    ax.legend(fontsize=config.legend_fontsize, loc="best", framealpha=0.9)
    ax.grid(alpha=config.grid_alpha, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.dpi, format=config.save_format, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax
