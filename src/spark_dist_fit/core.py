"""Core distribution fitting engine for Spark."""

import logging
from typing import Optional

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import NumericType

from spark_dist_fit.config import FitConfig, PlotConfig, SparkConfig
from spark_dist_fit.distributions import DistributionRegistry
from spark_dist_fit.fitting import FITTING_SAMPLE_SIZE, create_fitting_udf
from spark_dist_fit.histogram import HistogramComputer
from spark_dist_fit.results import DistributionFitResult, FitResults
from spark_dist_fit.utils import SparkSessionWrapper

logger = logging.getLogger(__name__)


class DistributionFitter(SparkSessionWrapper):
    """Modern Spark distribution fitting engine.

    Efficiently fits ~100 scipy.stats distributions to data using Spark's
    parallel processing capabilities. Uses broadcast variables and Pandas UDFs
    to avoid data collection and minimize serialization overhead.

    Key optimizations:
    - Computes histogram distributedly (no collect of raw data)
    - Broadcasts tiny histogram (~1KB) to all executors
    - Uses Pandas UDFs with Apache Arrow for 10x faster processing
    - Zero cross joins in the entire pipeline
    - Adaptive strategy for variable data sizes

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from spark_dist_fit import DistributionFitter, FitConfig
        >>>
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = spark.createDataFrame([(float(x),) for x in data], ['value'])
        >>>
        >>> # Simple usage
        >>> fitter = DistributionFitter()
        >>> results = fitter.fit(df, column='value')
        >>> best = results.best(n=1)[0]
        >>> print(f"Best: {best.distribution} with SSE={best.sse}")
        >>>
        >>> # With custom config
        >>> config = FitConfig(bins=100, support_at_zero=True)
        >>> fitter = DistributionFitter(spark, config=config)
        >>> results = fitter.fit(df, 'value')
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        config: Optional[FitConfig] = None,
        spark_config: Optional[SparkConfig] = None,
        distribution_registry: Optional[DistributionRegistry] = None,
    ):
        """Initialize DistributionFitter.

        Args:
            spark: Spark session. If None, gets or creates one.
            config: Fitting configuration (uses defaults if None)
            spark_config: Spark session configuration (uses defaults if None)
            distribution_registry: Custom distribution registry (uses default if None)
        """
        super().__init__(spark, spark_config)
        self.config: FitConfig = config or FitConfig()
        self.registry: DistributionRegistry = distribution_registry or DistributionRegistry()
        self.histogram_computer: HistogramComputer = HistogramComputer()

    def fit(
        self,
        df: DataFrame,
        column: str,
        config_override: Optional[FitConfig] = None,
        max_distributions: Optional[int] = None,
    ) -> FitResults:
        """Fit distributions to data column.

        This is the main method that orchestrates the entire fitting process:
        1. Validate inputs
        2. Apply sampling if needed
        3. Compute histogram (distributed, no collect!)
        4. Broadcast histogram
        5. Collect small sample for parameter fitting
        6. Apply Pandas UDF for fitting
        7. Return results

        Args:
            df: Spark DataFrame containing data
            column: Name of column to fit distributions to
            config_override: Optional config to override instance config
            max_distributions: Limit number of distributions to fit (for testing).
                              If None, fits all available distributions.

        Returns:
            FitResults object with fitted distributions

        Raises:
            ValueError: If column not found, DataFrame empty, or max_distributions is 0
            TypeError: If column is not numeric

        Example:
            >>> results = fitter.fit(df, column='value')
            >>> best = results.best(n=1)[0]
            >>> print(f"Best fit: {best.distribution}")
            >>>
            >>> # For fast testing with only 5 distributions
            >>> results = fitter.fit(df, 'value', max_distributions=5)
        """
        # Input validation
        self._validate_inputs(df, column, max_distributions)

        config = config_override or self.config

        # Step 1: Get row count
        row_count = df.count()
        if row_count == 0:
            raise ValueError("DataFrame is empty")
        logger.info(f"Row count: {row_count}")

        # Step 2: Sample if needed (for very large datasets)
        df_sample = self._apply_sampling(df, config, row_count)

        # Step 3: Compute histogram (distributed, NO collect of raw data!)
        logger.info("Computing histogram...")
        y_hist, x_hist = self.histogram_computer.compute_histogram(
            df_sample,
            column,
            bins=config.bins,
            use_rice_rule=config.use_rice_rule,
            approx_count=row_count,
        )
        logger.info(f"Histogram computed: {len(x_hist)} bins")

        # Step 4: Broadcast histogram (tiny: ~1KB for 100 bins)
        histogram_bc = self.spark.sparkContext.broadcast((y_hist, x_hist))

        # Step 5: Create small sample for parameter fitting (~10k rows)
        logger.info("Creating data sample for parameter fitting...")
        data_sample = self._create_fitting_sample(df_sample, column, config, row_count)
        data_sample_bc = self.spark.sparkContext.broadcast(data_sample)
        logger.info(f"Data sample size: {len(data_sample)}")

        # Step 6: Get distributions to fit
        distributions = self.registry.get_distributions(
            support_at_zero=config.support_at_zero,
            additional_exclusions=list(config.excluded_distributions),
        )

        # Limit distributions for testing if specified
        if max_distributions is not None and max_distributions > 0:
            distributions = distributions[:max_distributions]

        logger.info(f"Fitting {len(distributions)} distributions...")

        try:
            # Step 7: Create DataFrame of distributions
            dist_df = self.spark.createDataFrame([(dist,) for dist in distributions], ["distribution_name"])

            # Step 8: Determine optimal partitioning
            num_partitions = config.num_partitions or self._calculate_partitions(len(distributions))
            dist_df = dist_df.repartition(num_partitions)

            # Step 9: Apply Pandas UDF for fitting
            fitting_udf = create_fitting_udf(histogram_bc, data_sample_bc)

            results_df = dist_df.select(fitting_udf(F.col("distribution_name")).alias("result")).select("result.*")

            # Step 10: Filter out failed fits and cache
            results_df = results_df.filter(F.col("sse") < float(np.inf))
            results_df = results_df.cache()

            # Trigger computation and show progress
            num_results = results_df.count()
            logger.info(f"Successfully fit {num_results}/{len(distributions)} distributions")

            return FitResults(results_df)

        finally:
            # Clean up broadcast variables to prevent memory leaks
            histogram_bc.unpersist()
            data_sample_bc.unpersist()

    @staticmethod
    def _validate_inputs(df: DataFrame, column: str, max_distributions: Optional[int]) -> None:
        """Validate inputs to fit method.

        Args:
            df: Spark DataFrame
            column: Column name
            max_distributions: Maximum distributions to fit

        Raises:
            ValueError: If column not found or max_distributions is 0
            TypeError: If column is not numeric
        """
        if max_distributions == 0:
            raise ValueError("max_distributions cannot be 0")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {df.columns}")

        # Check column type
        col_type = df.schema[column].dataType
        if not isinstance(col_type, NumericType):
            raise TypeError(f"Column '{column}' must be numeric, got {col_type}")

    @staticmethod
    def _apply_sampling(df: DataFrame, config: FitConfig, row_count: int) -> DataFrame:
        """Apply sampling if enabled and data exceeds threshold.

        Args:
            df: Spark DataFrame
            config: Fitting configuration
            row_count: Row count

        Returns:
            Sampled or full DataFrame
        """
        if not config.enable_sampling or row_count <= config.sample_threshold:
            return df

        # Calculate sample fraction
        if config.sample_fraction is not None:
            fraction = config.sample_fraction
        else:
            # Auto-determine: aim for max_sample_size rows
            fraction = min(config.max_sample_size / row_count, 0.35)

        logger.info(
            "Sampling %.1f%% of data (%d rows)",
            fraction * 100,
            int(row_count * fraction),
        )
        return df.sample(fraction=fraction, seed=config.random_seed)

    @staticmethod
    def _create_fitting_sample(df: DataFrame, column: str, config: FitConfig, row_count: int) -> np.ndarray:
        """Create small sample for scipy distribution fitting.

        Most scipy distributions can be fit well with ~10k samples.
        This avoids passing large datasets to UDFs.

        Args:
            df: Spark DataFrame
            column: Column to sample
            config: Fitting configuration
            row_count: Total row count (avoids extra count)

        Returns:
            Numpy array with sample data
        """
        sample_size = min(FITTING_SAMPLE_SIZE, row_count)
        fraction = min(sample_size / row_count, 1.0)

        # Collect only the small sample using toPandas (more efficient than RDD)
        sample_df = df.select(column).sample(fraction=fraction, seed=config.random_seed)
        return sample_df.toPandas()[column].values

    def _calculate_partitions(self, num_distributions: int) -> int:
        """Calculate optimal number of partitions.

        Aims for 2-3 distributions per core for good parallelism.

        Args:
            num_distributions: Number of distributions to fit

        Returns:
            Optimal partition count
        """
        total_cores = self.spark.sparkContext.defaultParallelism
        return min(num_distributions, total_cores * 2)

    def plot(
        self,
        result: DistributionFitResult,
        df: DataFrame,
        column: str,
        config: Optional[PlotConfig] = None,
        title: str = "",
        xlabel: str = "Value",
        ylabel: str = "Density",
    ):
        """Plot fitted distribution against data histogram.

        Args:
            result: DistributionFitResult to plot
            df: DataFrame with data
            column: Column name
            config: Plot configuration
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label

        Returns:
            Tuple of (figure, axis) from matplotlib

        Example:
            >>> results = fitter.fit(df, 'value')
            >>> best = results.best(n=1)[0]
            >>> fitter.plot(best, df, 'value', title='Best Fit Distribution')
        """
        from spark_dist_fit.plotting import plot_distribution

        # Compute histogram for plotting
        y_hist, x_hist = self.histogram_computer.compute_histogram(
            df,
            column,
            bins=self.config.bins,
            use_rice_rule=self.config.use_rice_rule,
        )

        plot_config = config or PlotConfig()
        return plot_distribution(
            result,
            y_hist,
            x_hist,
            plot_config,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
