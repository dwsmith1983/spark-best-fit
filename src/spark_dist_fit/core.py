"""Core distribution fitting engine for Spark."""

import logging
from typing import Optional, Tuple

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from spark_dist_fit.config import FitConfig, PlotConfig
from spark_dist_fit.distributions import DistributionRegistry
from spark_dist_fit.fitting import create_fitting_udf
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
        spark: SparkSession,
        config: Optional[FitConfig] = None,
        distribution_registry: Optional[DistributionRegistry] = None,
    ):
        """Initialize DistributionFitter.

        Args:
            config: Fitting configuration (uses defaults if None)
            distribution_registry: Custom distribution registry (uses default if None)
        """
        super().__init__(spark)
        self.config = config or FitConfig()
        self.registry = distribution_registry or DistributionRegistry()
        self.histogram_computer = HistogramComputer(spark)

        # Enable Arrow optimization for Pandas UDFs
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

        self.spark.conf.set("spark.sql.adaptive.enabled", "true")
        self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Cache last fit for plotting convenience
        self._last_histogram: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_column: Optional[str] = None
        self._last_data: Optional[DataFrame] = None

    def fit(
        self,
        df: DataFrame,
        column: str,
        config_override: Optional[FitConfig] = None,
        max_distributions: Optional[int] = None,
    ) -> FitResults:
        """Fit distributions to data column.

        This is the main method that orchestrates the entire fitting process:
        1. Determine processing strategy (adaptive)
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

        Example:
            >>> results = fitter.fit(df, column='value')
            >>> best = results.best(n=1)[0]
            >>> print(f"Best fit: {best.distribution}")
            >>>
            >>> # For fast testing with only 5 distributions
            >>> results = fitter.fit(df, 'value', max_distributions=5)
        """
        if max_distributions == 0:
            raise ValueError("max_distributions cannot be 0")

        config = config_override or self.config
        self._last_column = column
        self._last_data = df

        # Step 1: Determine strategy based on data size
        strategy, row_count = self._determine_strategy(df, config)
        logger.info("Strategy: %s, approximate rows: %d", strategy, row_count)

        # Step 2: Sample if needed (avoids full scan for very large data)
        df_sample = self._apply_sampling(df, strategy, config, row_count)

        # Step 3: Compute histogram (distributed, NO collect of raw data!)
        logger.info("Computing histogram...")
        y_hist, x_hist = self.histogram_computer.compute_histogram(
            df_sample,
            column,
            bins=config.bins,
            use_rice_rule=config.use_rice_rule,
            approx_count=row_count,
        )
        self._last_histogram = (y_hist, x_hist)
        logger.info("Histogram computed: %d bins", len(x_hist))

        # Step 4: Broadcast histogram (tiny: ~1KB for 100 bins)
        histogram_bc = self.spark.sparkContext.broadcast((y_hist, x_hist))

        # Step 5: Create small sample for parameter fitting (~10k rows)
        logger.info("Creating data sample for parameter fitting...")
        data_sample = self._create_fitting_sample(df_sample, column, config)
        data_sample_bc = self.spark.sparkContext.broadcast(data_sample)
        logger.info("Data sample size: %d", len(data_sample))

        # Step 6: Get distributions to fit
        distributions = self.registry.get_distributions(
            support_at_zero=config.support_at_zero,
            additional_exclusions=config.excluded_distributions,
        )

        # Limit distributions for testing if specified
        if max_distributions is not None and max_distributions > 0:
            distributions = distributions[:max_distributions]

        logger.info("Fitting %d distributions...", len(distributions))

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
        logger.info("Successfully fit %d/%d distributions", num_results, len(distributions))

        return FitResults(results_df)

    @staticmethod
    def _determine_strategy(df: DataFrame, config: FitConfig) -> Tuple[str, int]:
        """Determine processing strategy based on data size.

        Args:
            df: Spark DataFrame
            config: Fitting configuration

        Returns:
            Tuple of (strategy, row_count)
        """
        if not config.adaptive_strategy:
            # Count is cached/fast for many DataFrames
            return "SPARK_FULL", df.count()

        # Try to get count efficiently
        try:
            count = df.count()
        except (RuntimeError, ValueError):
            # Fallback: sample-based estimation
            sample = df.sample(fraction=0.01, seed=config.random_seed).count()
            count = sample * 100

        # Determine strategy
        if count < config.local_threshold:
            strategy = "LOCAL"  # Could use local processing (future optimization)
        elif count < config.spark_threshold:
            strategy = "SPARK_FULL"
        else:
            strategy = "SPARK_SAMPLED"

        return strategy, count

    @staticmethod
    def _apply_sampling(df: DataFrame, strategy: str, config: FitConfig, row_count: int) -> DataFrame:
        """Apply sampling strategy based on data size.

        Args:
            df: Spark DataFrame
            strategy: Processing strategy
            config: Fitting configuration
            row_count: Approximate row count

        Returns:
            Sampled (or full) DataFrame
        """
        if strategy == "LOCAL" or not config.enable_sampling:
            return df

        if strategy == "SPARK_SAMPLED":
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

        return df

    @staticmethod
    def _create_fitting_sample(df: DataFrame, column: str, config: FitConfig) -> np.ndarray:
        """Create small sample for scipy distribution fitting.

        Most scipy distributions can be fit well with ~10k samples.
        This avoids passing large datasets to UDFs.

        Args:
            df: Spark DataFrame
            column: Column to sample
            config: Fitting configuration

        Returns:
            Numpy array with sample data
        """
        sample_size = min(10_000, df.count())
        fraction = sample_size / df.count()

        # Collect only the small sample
        sample_data = (
            df.select(column)
            .sample(fraction=fraction, seed=config.random_seed)
            .rdd.map(lambda row: float(row[0]))
            .collect()
        )

        return np.array(sample_data)

    def _calculate_partitions(self, num_distributions: int) -> int:
        """Calculate optimal number of partitions.

        Aims for 2-3 distributions per core for good parallelism.

        Args:
            num_distributions: Number of distributions to fit

        Returns:
            Optimal partition count
        """
        try:
            conf = dict(self.spark.sparkContext.getConf().getAll())
            cores = int(conf.get("spark.executor.cores", 4))
            executors = int(conf.get("spark.dynamicAllocation.maxExecutors", 10))
            total_cores = cores * executors

            # Aim for 2-3 distributions per core
            return min(num_distributions, total_cores * 2)
        except (KeyError, ValueError, TypeError):
            # Fallback
            return min(num_distributions, 20)

    def plot(
        self,
        result: DistributionFitResult,
        df: Optional[DataFrame] = None,
        column: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        title: str = "",
        xlabel: str = "Value",
        ylabel: str = "Density",
    ):
        """Plot fitted distribution against data histogram.

        Args:
            result: DistributionFitResult to plot
            df: DataFrame with data (uses last fit data if None)
            column: Column name (uses last fit column if None)
            config: Plot configuration
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label

        Example:
            >>> results = fitter.fit(df, 'value')
            >>> best = results.best(n=1)[0]
            >>> fitter.plot(best, title='Best Fit Distribution')
        """
        from .plotting import plot_distribution

        # Use last fit data if not provided
        if df is None:
            df = self._last_data
        if column is None:
            column = self._last_column
        if df is None or column is None:
            raise ValueError("Must provide df and column, or call fit() first")

        # Get histogram (use cached if same data)
        if self._last_histogram is None or df != self._last_data or column != self._last_column:
            y_hist, x_hist = self.histogram_computer.compute_histogram(
                df,
                column,
                bins=self.config.bins,
                use_rice_rule=self.config.use_rice_rule,
            )
        else:
            y_hist, x_hist = self._last_histogram

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
