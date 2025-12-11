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

from .config import FitConfig, PlotConfig
from .core import DistributionFitter
from .distributions import DistributionRegistry
from .results import DistributionFitResult, FitResults
from .utils import SparkSessionWrapper

__version__ = "2.0.0"
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
