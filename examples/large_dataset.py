"""Example demonstrating performance with large datasets."""

import time

import numpy as np
from pyspark.sql import SparkSession

from spark_dist_fit import DistributionFitter, FitConfig

# Create Spark session with more resources
spark = (
    SparkSession.builder.appName("LargeDatasetFitting")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

print("=" * 80)
print("LARGE DATASET PERFORMANCE DEMO")
print("=" * 80)

# ============================================================================
# Test with varying data sizes
# ============================================================================
data_sizes = [100_000, 1_000_000, 10_000_000]

for size in data_sizes:
    print(f"\n{'=' * 80}")
    print(f"Dataset size: {size:,} rows")
    print("=" * 80)

    # Generate data
    print("Generating data...")
    np.random.seed(42)
    data = np.random.gamma(shape=2.0, scale=2.0, size=size)

    # Create DataFrame
    df = spark.createDataFrame([(float(x),) for x in data], ["value"])
    df = df.cache()  # Cache for consistent timing
    df.count()  # Materialize

    # Configure for large data
    # Note: FitConfig uses DEFAULT_EXCLUDED_DISTRIBUTIONS which excludes
    # slow distributions like studentized_range, gausshyper, geninvgauss, etc.
    config = FitConfig(
        bins=50,  # Fewer bins for speed
        enable_sampling=True,  # Enable sampling for large datasets
        sample_fraction=None,  # Auto-determine
        max_sample_size=1_000_000,  # Limit to 1M for fitting
        sample_threshold=1_000_000,  # Sample when exceeding 1M rows
    )

    # Fit distributions
    print(f"\nFitting distributions to {size:,} rows...")
    start_time = time.time()

    fitter = DistributionFitter(spark, config=config)
    results = fitter.fit(df, column="value")

    elapsed = time.time() - start_time

    # Get results
    best = results.best(n=1)[0]
    num_fitted = results.count()

    print(f"\n{'Results':^80}")
    print("-" * 80)
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Distributions fitted: {num_fitted}")
    print(f"Rows per second: {size / elapsed:,.0f}")
    print(f"\nBest distribution: {best.distribution}")
    print(f"SSE: {best.sse:.6f}")
    print(f"Parameters: {[f'{p:.4f}' for p in best.parameters]}")

    # Clean up
    df.unpersist()

print("\n" + "=" * 80)
print("PERFORMANCE DEMO COMPLETED")
print("=" * 80)

spark.stop()
