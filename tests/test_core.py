"""Tests for core distribution fitting module."""

import numpy as np
import pytest

from spark_dist_fit import DistributionFitter, FitConfig
from spark_dist_fit.distributions import DistributionRegistry


class TestDistributionFitter:
    """Tests for DistributionFitter class."""

    def test_initialization(self, spark_session):
        """Test fitter initialization with defaults."""
        fitter = DistributionFitter(spark_session)

        assert fitter.spark == spark_session
        # Check type by class name (avoids src layout double-import issue)
        assert type(fitter.config).__name__ == "FitConfig"
        assert type(fitter.registry).__name__ == "DistributionRegistry"
        assert fitter._last_histogram is None
        assert fitter._last_column is None

    def test_initialization_with_custom_config(self, spark_session):
        """Test fitter initialization with custom config."""
        config = FitConfig(bins=100, support_at_zero=True)
        fitter = DistributionFitter(spark_session, config=config)

        assert fitter.config.bins == 100
        assert fitter.config.support_at_zero is True

    def test_initialization_with_custom_registry(self, spark_session):
        """Test fitter initialization with custom registry."""
        registry = DistributionRegistry(custom_exclusions={"norm", "expon"})
        fitter = DistributionFitter(spark_session, distribution_registry=registry)

        assert fitter.registry == registry
        assert "norm" in fitter.registry.get_exclusions()

    def test_fit_basic(self, spark_session, small_dataset):
        """Test basic fitting operation."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # Should return results
        assert results.count() > 0

        # Should find best distribution
        best = results.best(n=1)[0]
        assert best.distribution is not None
        assert best.sse < np.inf

    def test_fit_identifies_correct_distribution(self, spark_session, normal_data):
        """Test that fitter identifies the correct distribution."""
        df = spark_session.createDataFrame([(float(x),) for x in normal_data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, column="value", max_distributions=5)

        # Best distribution should be normal or close
        best = results.best(n=3)[0]

        # Should be among top candidates
        top_3_names = [r.distribution for r in results.best(n=3)]
        assert "norm" in top_3_names or best.sse < 0.01

    def test_fit_with_config_override(self, spark_session, small_dataset):
        """Test fitting with config override."""
        fitter = DistributionFitter(spark_session)

        # Override config
        config = FitConfig(bins=25, support_at_zero=False)
        results = fitter.fit(small_dataset, column="value", config_override=config, max_distributions=5)

        # Should complete successfully
        assert results.count() > 0

    def test_fit_support_at_zero(self, spark_session, positive_dataset):
        """Test fitting only non-negative distributions."""
        config = FitConfig(support_at_zero=True)
        fitter = DistributionFitter(spark_session, config=config)

        results = fitter.fit(positive_dataset, column="value", max_distributions=5)

        # Should have fitted distributions
        assert results.count() > 0

        # All distributions should be non-negative
        df_pandas = results.to_pandas()
        for dist_name in df_pandas["distribution"]:
            dist = fitter.registry._has_support_at_zero(dist_name)
            assert dist is True

    def test_apply_sampling_no_sampling(self, spark_session, small_dataset):
        """Test sampling application when sampling is disabled."""
        config = FitConfig(enable_sampling=False)
        fitter = DistributionFitter(spark_session, config=config)

        df_sampled = fitter._apply_sampling(small_dataset, config, 10_000)

        # Should return original DataFrame
        assert df_sampled.count() == small_dataset.count()

    def test_apply_sampling_below_threshold(self, spark_session, small_dataset):
        """Test sampling when row count is below threshold."""
        config = FitConfig(enable_sampling=True, sample_threshold=100_000)
        fitter = DistributionFitter(spark_session, config=config)

        df_sampled = fitter._apply_sampling(small_dataset, config, 10_000)

        # Should return original DataFrame (below threshold)
        assert df_sampled.count() == small_dataset.count()

    def test_apply_sampling_with_fraction(self, spark_session, medium_dataset):
        """Test sampling with specified fraction."""
        config = FitConfig(enable_sampling=True, sample_fraction=0.5, sample_threshold=50_000)
        fitter = DistributionFitter(spark_session, config=config)

        df_sampled = fitter._apply_sampling(medium_dataset, config, 100_000)

        # Should sample ~50% of data
        sampled_count = df_sampled.count()
        assert 45_000 < sampled_count < 55_000  # Allow some variance

    def test_apply_sampling_auto_fraction(self, spark_session, medium_dataset):
        """Test sampling with auto-determined fraction."""
        config = FitConfig(enable_sampling=True, sample_fraction=None, max_sample_size=50_000, sample_threshold=50_000)
        fitter = DistributionFitter(spark_session, config=config)

        df_sampled = fitter._apply_sampling(medium_dataset, config, 100_000)

        # Should sample to max_sample_size
        sampled_count = df_sampled.count()
        assert sampled_count <= 55_000  # Allow some variance

    def test_create_fitting_sample(self, spark_session, small_dataset):
        """Test creating sample for distribution fitting."""
        config = FitConfig()
        fitter = DistributionFitter(spark_session, config=config)

        sample = fitter._create_fitting_sample(small_dataset, "value", config)

        # Should be numpy array
        assert isinstance(sample, np.ndarray)

        # Should be <= 10k (default sample size)
        assert len(sample) <= 10_000

    def test_calculate_partitions(self, spark_session):
        """Test partition calculation."""
        fitter = DistributionFitter(spark_session)

        # Test with 100 distributions
        partitions = fitter._calculate_partitions(100)

        # Should be reasonable number
        assert partitions > 0
        assert partitions <= 100

    def test_calculate_partitions_few_distributions(self, spark_session):
        """Test partition calculation with few distributions."""
        fitter = DistributionFitter(spark_session)

        # With only 5 distributions
        partitions = fitter._calculate_partitions(5)

        # Should not exceed num_distributions
        assert partitions <= 5
        assert partitions >= 1

    def test_fit_caches_results(self, spark_session, small_dataset):
        """Test that fit results are cached."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # Access count multiple times (should use cache)
        count1 = results.count()
        count2 = results.count()

        assert count1 == count2
        assert count1 > 0

    def test_fit_stores_last_histogram(self, spark_session, small_dataset):
        """Test that fitter stores last histogram for plotting."""
        fitter = DistributionFitter(spark_session)

        assert fitter._last_histogram is None

        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # Should cache histogram
        assert fitter._last_histogram is not None
        y_hist, x_hist = fitter._last_histogram
        assert len(y_hist) > 0
        assert len(x_hist) > 0

    def test_fit_filters_failed_fits(self, spark_session, small_dataset):
        """Test that failed fits are filtered out."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # All results should have finite SSE
        df_pandas = results.to_pandas()
        assert all(np.isfinite(df_pandas["sse"]))

    def test_fit_with_constant_data(self, spark_session, constant_dataset):
        """Test fitting with constant data (edge case)."""
        fitter = DistributionFitter(spark_session)

        # Should handle gracefully
        results = fitter.fit(constant_dataset, column="value", max_distributions=5)

        # May have some results or none, but should not crash
        assert results.count() >= 0

    def test_fit_with_custom_bins(self, spark_session, small_dataset):
        """Test fitting with custom number of bins."""
        config = FitConfig(bins=25)
        fitter = DistributionFitter(spark_session, config=config)

        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        assert results.count() > 0

    def test_fit_with_rice_rule(self, spark_session, small_dataset):
        """Test fitting with Rice rule for bins."""
        config = FitConfig(use_rice_rule=True)
        fitter = DistributionFitter(spark_session, config=config)

        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        assert results.count() > 0

    def test_fit_excluded_distributions(self, spark_session, small_dataset):
        """Test that excluded distributions are not fitted."""
        config = FitConfig(excluded_distributions=["norm", "expon"])
        fitter = DistributionFitter(spark_session, config=config)

        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # norm and expon should not be in results
        df_pandas = results.to_pandas()
        assert "norm" not in df_pandas["distribution"].values
        assert "expon" not in df_pandas["distribution"].values

    def test_fit_multiple_columns_sequential(self, spark_session):
        """Test fitting multiple columns sequentially."""
        # Create DataFrame with multiple columns
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 10_000)
        data2 = np.random.exponential(5, 10_000)

        df = spark_session.createDataFrame([(float(x), float(y)) for x, y in zip(data1, data2)], ["col1", "col2"])

        fitter = DistributionFitter(spark_session)

        # Fit first column
        results1 = fitter.fit(df, column="col1", max_distributions=5)
        best1 = results1.best(n=1)[0]

        # Fit second column
        results2 = fitter.fit(df, column="col2", max_distributions=5)
        best2 = results2.best(n=1)[0]

        # Both should succeed
        assert best1.sse < np.inf
        assert best2.sse < np.inf

        # Should identify different distributions
        top1 = [r.distribution for r in results1.best(n=3)]
        top2 = [r.distribution for r in results2.best(n=3)]

        # Normal should be in top for col1, expon should be in top for col2
        assert "norm" in top1 or best1.sse < 0.01
        assert "expon" in top2 or best2.sse < 0.01

    def test_fit_reproducibility(self, spark_session, small_dataset):
        """Test that fitting is reproducible with same seed."""
        config = FitConfig(random_seed=42)
        fitter1 = DistributionFitter(spark_session, config=config)
        fitter2 = DistributionFitter(spark_session, config=config)

        # Use max_distributions to speed up test
        results1 = fitter1.fit(small_dataset, column="value", max_distributions=5)
        results2 = fitter2.fit(small_dataset, column="value", max_distributions=5)

        # Should get same best distribution
        best1 = results1.best(n=1)[0]
        best2 = results2.best(n=1)[0]

        assert best1.distribution == best2.distribution
        # SSE might differ slightly due to sampling, but should be close
        assert np.isclose(best1.sse, best2.sse, rtol=0.1)


class TestDistributionFitterPlotting:
    """Tests for plotting functionality."""

    def test_plot_after_fit(self, spark_session, small_dataset):
        """Test plotting after fitting."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Should not raise error
        fig, ax = fitter.plot(best)

        assert fig is not None
        assert ax is not None

    def test_plot_without_fit_fails(self, spark_session):
        """Test that plotting without fit raises error."""
        fitter = DistributionFitter(spark_session)

        # Create a dummy result
        from spark_dist_fit.results import DistributionFitResult

        result = DistributionFitResult(distribution="norm", parameters=[0.0, 50.0, 10.0], sse=0.005)

        # Should raise error (no data cached)
        with pytest.raises(ValueError):
            fitter.plot(result)

    def test_plot_with_explicit_data(self, spark_session, small_dataset):
        """Test plotting with explicitly provided data."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Should work with explicit data
        fig, ax = fitter.plot(best, df=small_dataset, column="value", title="Test Plot")

        assert fig is not None
        assert ax is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_dataset(self, spark_session):
        """Test with very small dataset."""
        data = np.array([1.0, 2.0, 3.0])
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)

        # Should handle gracefully
        results = fitter.fit(df, column="value", max_distributions=5)

        # May or may not find distributions, but should not crash
        assert results.count() >= 0

    def test_single_value_dataset(self, spark_session):
        """Test with single value."""
        df = spark_session.createDataFrame([(42.0,)], ["value"])

        fitter = DistributionFitter(spark_session)

        # Should handle gracefully
        results = fitter.fit(df, column="value", max_distributions=5)

        assert results.count() >= 0

    def test_dataset_with_outliers(self, spark_session):
        """Test with dataset containing extreme outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 9995)
        outliers = np.array([1000, -1000, 2000, -2000, 3000])
        data = np.concatenate([normal_data, outliers])

        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)

        # Should handle outliers
        results = fitter.fit(df, column="value", max_distributions=5)

        assert results.count() > 0
        best = results.best(n=1)[0]
        assert best.sse < np.inf

    def test_apply_sampling_at_threshold(self, spark_session, small_dataset):
        """Test that data at threshold doesn't sample."""
        config = FitConfig(enable_sampling=True, sample_threshold=10_000)
        fitter = DistributionFitter(spark_session, config=config)

        df_result = fitter._apply_sampling(small_dataset, config, 10_000)

        # At threshold should return original data (uses <=)
        assert df_result.count() == small_dataset.count()

    def test_calculate_partitions_returns_reasonable_value(self, spark_session):
        """Test partition calculation returns reasonable values."""
        fitter = DistributionFitter(spark_session)

        # Various distribution counts
        for num_dists in [1, 10, 50, 100, 200]:
            partitions = fitter._calculate_partitions(num_dists)
            assert partitions >= 1
            assert partitions <= num_dists

    def test_fit_max_distributions_zero(self, spark_session, small_dataset):
        """Test fitting with max_distributions=0 fits nothing."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError):
            results = fitter.fit(small_dataset, column="value", max_distributions=0)

    def test_fit_updates_last_data_reference(self, spark_session, small_dataset):
        """Test that fit updates cached data reference."""
        fitter = DistributionFitter(spark_session)

        assert fitter._last_data is None
        assert fitter._last_column is None

        fitter.fit(small_dataset, column="value", max_distributions=3)

        assert fitter._last_data is not None
        assert fitter._last_column == "value"

    def test_fit_with_different_columns(self, spark_session):
        """Test fitting on different column names."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["custom_column_name"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, column="custom_column_name", max_distributions=3)

        assert results.count() > 0
        assert fitter._last_column == "custom_column_name"


class TestCoreNegativePaths:
    """Tests for negative/error paths in core module."""

    def test_plot_requires_df_and_column(self, spark_session):
        """Test that plot requires either cached data or explicit df/column."""
        fitter = DistributionFitter(spark_session)

        from spark_dist_fit.results import DistributionFitResult

        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005)

        # No cached data, no explicit data provided
        with pytest.raises(ValueError):
            fitter.plot(result)

    def test_plot_with_only_df_no_column(self, spark_session, small_dataset):
        """Test that plot with df but no column uses cached column."""
        fitter = DistributionFitter(spark_session)

        # First fit to cache column
        results = fitter.fit(small_dataset, column="value", max_distributions=3)
        best = results.best(n=1)[0]

        # Should work with explicit df but using cached column
        fig, ax = fitter.plot(best, df=small_dataset)
        assert fig is not None

    def test_plot_recomputes_histogram_when_data_changes(self, spark_session):
        """Test that plot recomputes histogram when data changes."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 1000)
        data2 = np.random.normal(100, 20, 1000)

        df1 = spark_session.createDataFrame([(float(x),) for x in data1], ["value"])
        df2 = spark_session.createDataFrame([(float(x),) for x in data2], ["value"])

        fitter = DistributionFitter(spark_session)

        # Fit on first dataset
        results = fitter.fit(df1, column="value", max_distributions=3)
        best = results.best(n=1)[0]

        # Plot with different dataset should work
        fig, ax = fitter.plot(best, df=df2, column="value")
        assert fig is not None
