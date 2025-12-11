"""Tests for histogram computation module."""

import numpy as np

from spark_dist_fit.histogram import HistogramComputer


class TestHistogramComputer:
    """Tests for HistogramComputer class."""

    def test_initialization(self, spark_session):
        """Test histogram computer initialization."""
        computer = HistogramComputer(spark_session)

        assert computer.spark == spark_session

    def test_compute_histogram_basic(self, spark_session, small_dataset):
        """Test basic histogram computation."""
        computer = HistogramComputer(spark_session)
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=50)

        # Should return arrays of correct size
        assert len(y_hist) == 50
        assert len(x_hist) == 50

        # Histogram should be normalized (density sums to ~1 when multiplied by bin widths)
        bin_width = x_hist[1] - x_hist[0]
        total_area = np.sum(y_hist * bin_width)
        assert np.isclose(total_area, 1.0, atol=0.01)

        # All values should be non-negative
        assert np.all(y_hist >= 0)

    def test_compute_histogram_custom_bins(self, spark_session, small_dataset):
        """Test histogram with custom number of bins."""
        computer = HistogramComputer(spark_session)

        for n_bins in [10, 25, 100]:
            y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=n_bins)

            assert len(y_hist) == n_bins
            assert len(x_hist) == n_bins

    def test_compute_histogram_rice_rule(self, spark_session, small_dataset):
        """Test histogram with Rice rule for bin calculation."""
        computer = HistogramComputer(spark_session)
        row_count = small_dataset.count()

        y_hist, x_hist = computer.compute_histogram(
            small_dataset, "value", bins=50, use_rice_rule=True, approx_count=row_count
        )

        # Rice rule: bins = 2 * n^(1/3)
        expected_bins = int(np.ceil(row_count ** (1 / 3)) * 2)

        assert len(y_hist) == expected_bins
        assert len(x_hist) == expected_bins

    def test_compute_histogram_constant_values(self, spark_session, constant_dataset):
        """Test histogram with constant values (edge case)."""
        computer = HistogramComputer(spark_session)
        y_hist, x_hist = computer.compute_histogram(constant_dataset, "value", bins=50)

        # Should handle min == max case
        assert len(y_hist) == 1
        assert len(x_hist) == 1

        # Single bin centered at the constant value
        assert np.isclose(x_hist[0], 42.0)
        assert np.isclose(y_hist[0], 1.0)

    def test_compute_histogram_positive_data(self, spark_session, positive_dataset):
        """Test histogram with only positive values."""
        computer = HistogramComputer(spark_session)
        y_hist, x_hist = computer.compute_histogram(positive_dataset, "value", bins=50)

        # All bin centers should be positive
        assert np.all(x_hist >= 0)

        # Should have correct size
        assert len(y_hist) == 50
        assert len(x_hist) == 50

    def test_compute_histogram_bin_edges_array(self, spark_session, small_dataset):
        """Test histogram with custom bin edges as array."""
        computer = HistogramComputer(spark_session)
        custom_bins = np.array([0, 20, 40, 60, 80, 100])

        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=custom_bins)

        # Should have len(bins) - 1 bins
        assert len(y_hist) == len(custom_bins) - 1
        assert len(x_hist) == len(custom_bins) - 1

    def test_get_approx_count(self, spark_session, small_dataset):
        """Test approximate count calculation."""
        computer = HistogramComputer(spark_session)
        approx_count = computer._get_approx_count(small_dataset)

        # Should be close to actual count
        actual_count = small_dataset.count()
        assert approx_count == actual_count  # For small data, should be exact

    def test_compute_histogram_distributed_no_collect(self, spark_session, small_dataset):
        """Test that histogram stays distributed (doesn't collect raw data)."""
        computer = HistogramComputer(spark_session)

        # This should NOT collect raw data, only aggregated histogram
        bin_edges = np.linspace(0, 100, 51)
        histogram_df = computer._compute_histogram_distributed(small_dataset, "value", bin_edges)

        # Result should be a DataFrame with (bin_id, count)
        assert "bin_id" in histogram_df.columns
        assert "count" in histogram_df.columns

        # Should have at most len(bin_edges) - 1 rows (some bins may be empty)
        assert histogram_df.count() <= len(bin_edges) - 1

    def test_compute_statistics(self, spark_session, normal_data, small_dataset):
        """Test computing basic statistics."""
        computer = HistogramComputer(spark_session)
        stats = computer.compute_statistics(small_dataset, "value")

        # Should have all statistics
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "stddev" in stats
        assert "count" in stats

        # Values should be reasonable for normal(50, 10) data
        assert stats["mean"] is not None
        assert 45 < stats["mean"] < 55  # Close to 50

        assert stats["stddev"] is not None
        assert 8 < stats["stddev"] < 12  # Close to 10

        assert stats["count"] == len(normal_data)

    def test_compute_statistics_types(self, spark_session, small_dataset):
        """Test that statistics are returned as floats."""
        computer = HistogramComputer(spark_session)
        stats = computer.compute_statistics(small_dataset, "value")

        # All should be floats or None
        for key, value in stats.items():
            if value is not None:
                assert isinstance(value, float)

    def test_histogram_normalization(self, spark_session, small_dataset):
        """Test that histogram is properly normalized to density."""
        computer = HistogramComputer(spark_session)
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=50)

        # Calculate bin widths
        bin_edges = np.linspace(
            x_hist.min() - (x_hist[1] - x_hist[0]) / 2,
            x_hist.max() + (x_hist[1] - x_hist[0]) / 2,
            51,
        )
        bin_widths = np.diff(bin_edges)

        # Total area under histogram should be ~1
        total_area = np.sum(y_hist * bin_widths)
        assert np.isclose(total_area, 1.0, atol=0.01)

    def test_histogram_no_data_loss(self, spark_session, small_dataset):
        """Test that histogram captures all data (no bins with zero when they shouldn't be)."""
        computer = HistogramComputer(spark_session)
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=50)

        # For normal distribution, most bins should have some data
        non_zero_bins = np.sum(y_hist > 0)
        assert non_zero_bins > 40  # At least 80% of bins should have data

    def test_histogram_with_outliers(self, spark_session):
        """Test histogram computation with outliers."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 9900)
        outliers = np.array([0, 200, -50, 250])  # Extreme outliers
        data = np.concatenate([normal_data, outliers])

        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        computer = HistogramComputer(spark_session)
        y_hist, x_hist = computer.compute_histogram(df, "value", bins=50)

        # Should handle outliers gracefully
        assert len(y_hist) == 50
        assert len(x_hist) == 50

        # Min and max should capture outliers
        assert x_hist.min() < 0
        assert x_hist.max() > 200

    def test_medium_dataset_performance(self, spark_session, medium_dataset):
        """Test histogram computation on medium dataset (100K rows)."""
        computer = HistogramComputer(spark_session)

        # Should complete without errors
        y_hist, x_hist = computer.compute_histogram(medium_dataset, "value", bins=100)

        assert len(y_hist) == 100
        assert len(x_hist) == 100

        # Should still be normalized
        bin_width = x_hist[1] - x_hist[0]
        total_area = np.sum(y_hist * bin_width)
        assert np.isclose(total_area, 1.0, atol=0.01)
