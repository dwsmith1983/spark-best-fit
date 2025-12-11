"""spark-dist-fit: Modern Spark 4 distribution fitting library.

Efficiently fits ~100 scipy.stats distributions to data using Spark's
parallel processing with optimized Pandas UDFs and broadcast variables.

Example:
    >>> from pyspark.sql import SparkSession
    >>> from spark_dist_fit import DistributionFitter, FitConfig
    >>>
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.createDataFrame([(float(x),) for x in data], ['value'])
    >>>
    >>> # Fit distributions
    >>> fitter = DistributionFitter(spark)
    >>> results = fitter.fit(df, column='value')
    >>>
    >>> # Get best distribution
    >>> best = results.best(n=1)[0]
    >>> print(f"Best: {best.distribution} with SSE={best.sse:.6f}")
    >>>
    >>> # Plot
    >>> fitter.plot(best, title='Best Fit Distribution')
"""

from spark_dist_fit._version import __version__
from spark_dist_fit.config import FitConfig, PlotConfig
from spark_dist_fit.core import DistributionFitter
from spark_dist_fit.distributions import DistributionRegistry
from spark_dist_fit.results import DistributionFitResult, FitResults
from spark_dist_fit.utils import SparkSessionWrapper

__author__ = "Dustin Smith"
__email__ = "dustin.william.smith@gmail.com"

__all__ = [
    # Main classes
    "DistributionFitter",
    # Configuration
    "FitConfig",
    "PlotConfig",
    # Results
    "FitResults",
    "DistributionFitResult",
    # Distribution management
    "DistributionRegistry",
    # Utilities
    "SparkSessionWrapper",
    # Version
    "__version__",
]
