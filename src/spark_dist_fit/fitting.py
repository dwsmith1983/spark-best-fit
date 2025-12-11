"""Distribution fitting using Pandas UDFs for efficient parallel processing."""

import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

# Define output schema for Pandas UDF
# Note: Pandas infers all columns as nullable, so we match that here
FIT_RESULT_SCHEMA = StructType(
    [
        StructField("distribution", StringType(), True),
        StructField("parameters", ArrayType(FloatType()), True),
        StructField("sse", FloatType(), True),
        StructField("aic", FloatType(), True),
        StructField("bic", FloatType(), True),
    ]
)


def create_fitting_udf(histogram_broadcast: Any, data_sample_broadcast: Any):
    """Factory function to create Pandas UDF with broadcasted data.

    This is the KEY optimization: The histogram and data sample are
    broadcasted once to all executors, then the Pandas UDF processes
    batches of distributions efficiently using vectorized operations.

    Args:
        histogram_broadcast: Broadcast variable containing (y_hist, x_hist)
        data_sample_broadcast: Broadcast variable containing data sample

    Returns:
        Pandas UDF function for fitting distributions

    Example:
        >>> # In DistributionFitter:
        >>> hist_bc = spark.sparkContext.broadcast((y_hist, x_hist))
        >>> data_bc = spark.sparkContext.broadcast(data_sample)
        >>> fitting_udf = create_fitting_udf(hist_bc, data_bc)
        >>> results = df.select(fitting_udf(col('distribution_name')))
    """

    @pandas_udf(FIT_RESULT_SCHEMA)
    def fit_distributions_batch(distribution_names: pd.Series) -> pd.DataFrame:
        """Vectorized UDF to fit multiple distributions in a batch.

        This function processes a batch of distribution names, fitting each
        against the broadcasted histogram and data sample. Uses Apache Arrow
        for efficient data transfer.

        Args:
            distribution_names: Series of scipy distribution names to fit

        Returns:
            DataFrame with columns: distribution, parameters, sse, aic, bic
        """
        # Get broadcasted data (no serialization overhead!)
        y_hist, x_hist = histogram_broadcast.value
        data_sample = data_sample_broadcast.value

        # Fit each distribution in the batch
        results = []
        for dist_name in distribution_names:
            result = fit_single_distribution(
                dist_name=dist_name,
                data_sample=data_sample,
                x_hist=x_hist,
                y_hist=y_hist,
            )
            results.append(result)

        # Create DataFrame with explicit schema compliance
        df = pd.DataFrame(results)
        # Ensure non-nullable columns have no None values
        df["distribution"] = df["distribution"].astype(str)
        df["sse"] = df["sse"].astype(float)
        return df

    return fit_distributions_batch


def fit_single_distribution(
    dist_name: str, data_sample: np.ndarray, x_hist: np.ndarray, y_hist: np.ndarray
) -> Dict[str, Any]:
    """Fit a single distribution and compute goodness-of-fit metrics.

    Args:
        dist_name: Name of scipy.stats distribution
        data_sample: Sample of raw data for parameter fitting
        x_hist: Histogram bin centers
        y_hist: Histogram density values

    Returns:
        Dictionary with keys: distribution, parameters, sse, aic, bic
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Get distribution object
            dist = getattr(st, dist_name)

            # Fit distribution to data sample
            params = dist.fit(data_sample)

            # Evaluate PDF at histogram bin centers
            pdf_values = evaluate_pdf(dist, params, x_hist)

            # Compute Sum of Squared Errors
            sse = np.sum((y_hist - pdf_values) ** 2.0)

            # Compute information criteria
            aic, bic = compute_information_criteria(dist, params, data_sample)

            return {
                "distribution": dist_name,
                "parameters": [float(p) for p in params],
                "sse": float(sse),
                "aic": float(aic),
                "bic": float(bic),
            }

    except (ValueError, RuntimeError, FloatingPointError, AttributeError):
        # Return sentinel values for failed fits
        # Use [np.nan] for parameters to maintain non-empty list
        return {
            "distribution": dist_name,
            "parameters": [float(np.nan)],
            "sse": float(np.inf),
            "aic": float(np.inf),
            "bic": float(np.inf),
        }


def evaluate_pdf(dist: Any, params: Tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """Evaluate probability density function at given points.

    Args:
        dist: scipy.stats distribution object
        params: Distribution parameters (shape params, loc, scale)
        x: Points at which to evaluate PDF

    Returns:
        PDF values at x
    """
    # Extract shape, loc, scale from params
    arg = params[:-2]  # Shape parameters
    loc = params[-2]  # Location
    scale = params[-1]  # Scale

    # Evaluate PDF
    pdf = dist.pdf(x, *arg, loc=loc, scale=scale)

    # Handle potential numerical issues
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

    return pdf


def compute_information_criteria(dist: Any, params: Tuple[float, ...], data: np.ndarray) -> Tuple[float, float]:
    """Compute AIC and BIC information criteria.

    These criteria help compare model complexity vs fit quality.
    Lower values indicate better models.

    Args:
        dist: scipy.stats distribution object
        params: Fitted distribution parameters
        data: Original data sample

    Returns:
        Tuple of (aic, bic)
    """
    try:
        n = len(data)
        k = len(params)  # Number of parameters

        # Compute log-likelihood
        log_likelihood = np.sum(dist.logpdf(data, *params))

        # Handle numerical issues
        if not np.isfinite(log_likelihood):
            return np.inf, np.inf

        # Akaike Information Criterion
        aic = 2 * k - 2 * log_likelihood

        # Bayesian Information Criterion
        bic = k * np.log(n) - 2 * log_likelihood

        return aic, bic

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf, np.inf


def create_sample_data(data_full: np.ndarray, sample_size: int = 10_000, random_seed: int = 42) -> np.ndarray:
    """Create a sample of data for distribution fitting.

    Most scipy distributions can be fit accurately with ~10k samples,
    avoiding the need to pass entire large datasets to UDFs.

    Args:
        data_full: Full dataset
        sample_size: Target sample size
        random_seed: Random seed for reproducibility

    Returns:
        Sampled data (or full data if smaller than sample_size)
    """
    if len(data_full) <= sample_size:
        return data_full

    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(data_full), size=sample_size, replace=False)
    return data_full[indices]
