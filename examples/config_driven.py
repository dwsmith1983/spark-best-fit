"""Config-driven example demonstrating full HOCON config -> dataclass -> code workflow.

This example shows how to:
1. Load SparkConfig, FitConfig, and PlotConfig from a single HOCON file
2. Use SparkConfig to configure the Spark session
3. Use FitConfig to configure distribution fitting
4. Use PlotConfig to configure visualization

Run from the examples/ directory:
    python config_driven.py
"""

from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

from spark_dist_fit import (
    AppConfig,
    DistributionFitter,
    FitConfig,
)


def main():
    # =========================================================================
    # Step 1: Load all configurations from HOCON file
    # =========================================================================
    print("=" * 80)
    print("CONFIG-DRIVEN EXAMPLE: HOCON -> Dataclass -> Code")
    print("=" * 80)

    config_path = Path(__file__).parent.parent / "config" / "example.conf"
    print(f"\nLoading configuration from: {config_path}")

    # Load all configs at once using AppConfig (nested dataclass)
    # The HOCON file has spark{}, fit{}, and plot{} sections
    config = AppConfig.from_file(config_path)

    # Access individual configs through the nested structure
    spark_config = config.spark
    fit_config = config.fit
    plot_config = config.plot

    # =========================================================================
    # Step 2: Display loaded configurations (frozen dataclasses)
    # =========================================================================
    print("\n" + "-" * 80)
    print("LOADED CONFIGURATIONS (immutable dataclasses)")
    print("-" * 80)

    print("\nSparkConfig:")
    print(f"  app_name: {spark_config.app_name}")
    print(f"  arrow_enabled: {spark_config.arrow_enabled}")
    print(f"  adaptive_enabled: {spark_config.adaptive_enabled}")
    print(f"  adaptive_coalesce_enabled: {spark_config.adaptive_coalesce_enabled}")

    print("\nFitConfig:")
    print(f"  bins: {fit_config.bins}")
    print(f"  use_rice_rule: {fit_config.use_rice_rule}")
    print(f"  support_at_zero: {fit_config.support_at_zero}")
    print(f"  enable_sampling: {fit_config.enable_sampling}")
    print(f"  sample_threshold: {fit_config.sample_threshold:,}")
    print(f"  random_seed: {fit_config.random_seed}")
    print(f"  excluded_distributions: {len(fit_config.excluded_distributions)} distributions")

    print("\nPlotConfig:")
    print(f"  figsize: {plot_config.figsize}")
    print(f"  dpi: {plot_config.dpi}")
    print(f"  histogram_alpha: {plot_config.histogram_alpha}")
    print(f"  pdf_linewidth: {plot_config.pdf_linewidth}")
    print(f"  title_fontsize: {plot_config.title_fontsize}")

    # =========================================================================
    # Step 3: Create Spark session with SparkConfig
    # =========================================================================
    print("\n" + "-" * 80)
    print("CREATING SPARK SESSION WITH CONFIG")
    print("-" * 80)

    # SparkConfig.to_spark_config() returns a dict of Spark settings
    spark_settings = spark_config.to_spark_config()
    print("\nApplying Spark settings:")
    for key, value in spark_settings.items():
        print(f"  {key}: {value}")

    # Build Spark session with our config
    builder = SparkSession.builder.appName(spark_config.app_name)
    for key, value in spark_settings.items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()

    print(f"\nSpark session created: {spark.sparkContext.appName}")

    # =========================================================================
    # Step 4: Generate sample data
    # =========================================================================
    print("\n" + "-" * 80)
    print("GENERATING SAMPLE DATA")
    print("-" * 80)

    np.random.seed(fit_config.random_seed)
    data = np.random.normal(loc=50, scale=10, size=100_000)
    df = spark.createDataFrame([(float(x),) for x in data], ["value"])

    print(f"Created DataFrame with {df.count():,} rows")
    print(f"Data statistics: mean={data.mean():.2f}, std={data.std():.2f}")

    # =========================================================================
    # Step 5: Fit distributions using FitConfig
    # =========================================================================
    print("\n" + "-" * 80)
    print("FITTING DISTRIBUTIONS WITH CONFIG")
    print("-" * 80)

    # Pass spark_config to DistributionFitter (it will apply settings)
    fitter = DistributionFitter(
        spark=spark,
        config=fit_config,
        spark_config=spark_config,
    )

    print(f"\nFitting with {fit_config.bins} bins...")
    print(f"Excluding {len(fit_config.excluded_distributions)} slow distributions")

    # Limit to 20 distributions for demo speed
    results = fitter.fit(df, column="value", max_distributions=20)

    print(f"\nSuccessfully fit {results.count()} distributions")

    # =========================================================================
    # Step 6: Display results
    # =========================================================================
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)

    print("\nTop 5 distributions by SSE:")
    for i, result in enumerate(results.best(n=5, metric="sse"), 1):
        print(f"  {i}. {result.distribution:20s} SSE={result.sse:.6f}")

    best = results.best(n=1)[0]
    print(f"\nBest distribution: {best.distribution}")
    print(f"  Parameters: {[f'{p:.4f}' for p in best.parameters]}")
    print(f"  SSE: {best.sse:.6f}")
    print(f"  AIC: {best.aic:.2f}")
    print(f"  BIC: {best.bic:.2f}")

    # =========================================================================
    # Step 7: Plot using PlotConfig
    # =========================================================================
    print("\n" + "-" * 80)
    print("PLOTTING WITH CONFIG")
    print("-" * 80)

    print(f"\nPlotting with figsize={plot_config.figsize}, dpi={plot_config.dpi}")

    fig, ax = fitter.plot(
        best,
        df,
        "value",
        config=plot_config,
        title="Config-Driven Distribution Fitting",
        xlabel="Value",
        ylabel="Density",
    )

    print("Plot generated successfully!")

    # Optionally save the plot
    # fig.savefig("fit_result.png", dpi=plot_config.dpi, format=plot_config.save_format)

    # =========================================================================
    # Step 8: Demonstrate config immutability
    # =========================================================================
    print("\n" + "-" * 80)
    print("CONFIG IMMUTABILITY (frozen dataclasses)")
    print("-" * 80)

    print("\nAttempting to modify frozen FitConfig...")
    try:
        fit_config.bins = 200  # This will raise an error
    except AttributeError as e:
        print(f"  Error (expected): {e}")
        print("  Configs are immutable - create a new instance to change values")

    # To change config, create a new instance
    print("\nCreating new config with different values:")
    from dataclasses import replace

    # Note: replace() works with frozen dataclasses
    new_fit_config = FitConfig(
        bins=200,
        support_at_zero=True,
        enable_sampling=fit_config.enable_sampling,
        random_seed=fit_config.random_seed,
    )
    print(f"  New config bins: {new_fit_config.bins}")
    print(f"  New config support_at_zero: {new_fit_config.support_at_zero}")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n" + "=" * 80)
    print("CONFIG-DRIVEN EXAMPLE COMPLETED")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
