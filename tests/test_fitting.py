"""Tests for fitting module."""

import numpy as np
import scipy.stats as st

from spark_dist_fit.fitting import (
    compute_information_criteria,
    create_sample_data,
    evaluate_pdf,
    fit_single_distribution,
)


class TestFitSingleDistribution:
    """Tests for fitting single distributions."""

    def test_fit_normal_distribution(self, normal_data):
        """Test fitting normal distribution to normal data."""
        # Create histogram
        y_hist, x_edges = np.histogram(normal_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        result = fit_single_distribution("norm", normal_data, x_hist, y_hist)

        # Should succeed
        assert result["distribution"] == "norm"
        assert len(result["parameters"]) > 0
        assert result["sse"] < np.inf
        assert result["aic"] < np.inf
        assert result["bic"] < np.inf

        # Parameters should be close to true values (loc=50, scale=10)
        params = result["parameters"]
        loc = params[-2]
        scale = params[-1]

        assert 45 < loc < 55  # Close to 50
        assert 8 < scale < 12  # Close to 10

    def test_fit_exponential_distribution(self, exponential_data):
        """Test fitting exponential distribution."""
        y_hist, x_edges = np.histogram(exponential_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        result = fit_single_distribution("expon", exponential_data, x_hist, y_hist)

        # Should succeed
        assert result["distribution"] == "expon"
        assert len(result["parameters"]) > 0
        assert result["sse"] < np.inf

        # Scale should be close to 5.0
        scale = result["parameters"][-1]
        assert 4 < scale < 6

    def test_fit_gamma_distribution(self, gamma_data):
        """Test fitting gamma distribution."""
        y_hist, x_edges = np.histogram(gamma_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        result = fit_single_distribution("gamma", gamma_data, x_hist, y_hist)

        # Should succeed
        assert result["distribution"] == "gamma"
        assert result["sse"] < np.inf

    def test_fit_invalid_distribution(self, normal_data):
        """Test fitting with invalid distribution name."""
        y_hist, x_edges = np.histogram(normal_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        result = fit_single_distribution("invalid_dist", normal_data, x_hist, y_hist)

        # Should fail gracefully
        assert result["distribution"] == "invalid_dist"
        assert result["sse"] == np.inf
        assert result["aic"] == np.inf
        assert result["bic"] == np.inf
        assert result["parameters"] == [float(np.nan)]

    def test_fit_with_insufficient_data(self):
        """Test fitting with very little data."""
        data = np.array([1.0, 2.0, 3.0])
        y_hist = np.array([0.5, 0.3, 0.2])
        x_hist = np.array([1.0, 2.0, 3.0])

        result = fit_single_distribution("norm", data, x_hist, y_hist)

        # Should attempt to fit (may succeed or fail)
        assert result["distribution"] == "norm"
        # Either succeeds or returns inf
        assert result["sse"] >= 0 or result["sse"] == np.inf


class TestEvaluatePDF:
    """Tests for PDF evaluation."""

    def test_evaluate_pdf_normal(self):
        """Test evaluating PDF for normal distribution."""
        dist = st.norm
        params = (0, 1)  # Standard normal: loc=0, scale=1
        x = np.array([-2, -1, 0, 1, 2])

        pdf_values = evaluate_pdf(dist, params, x)

        # Should return valid PDF values
        assert len(pdf_values) == len(x)
        assert np.all(pdf_values >= 0)
        assert np.all(np.isfinite(pdf_values))

        # PDF at 0 should be highest for standard normal
        assert pdf_values[2] == np.max(pdf_values)

    def test_evaluate_pdf_with_shape_params(self):
        """Test evaluating PDF with shape parameters."""
        dist = st.gamma
        params = (2.0, 0, 2.0)  # shape=2, loc=0, scale=2
        x = np.linspace(0, 10, 50)

        pdf_values = evaluate_pdf(dist, params, x)

        # Should return valid PDF values
        assert len(pdf_values) == len(x)
        assert np.all(pdf_values >= 0)
        assert np.all(np.isfinite(pdf_values))

    def test_evaluate_pdf_handles_nan(self):
        """Test that PDF evaluation handles NaN gracefully."""
        dist = st.norm
        params = (0, 1)
        x = np.array([np.nan, 0, 1])

        pdf_values = evaluate_pdf(dist, params, x)

        # Should convert NaN to 0
        assert np.isfinite(pdf_values).all()


class TestComputeInformationCriteria:
    """Tests for information criteria calculation."""

    def test_compute_aic_bic_normal(self, normal_data):
        """Test computing AIC and BIC for normal distribution."""
        dist = st.norm
        params = dist.fit(normal_data)

        aic, bic = compute_information_criteria(dist, params, normal_data)

        # Should return finite values
        assert np.isfinite(aic)
        assert np.isfinite(bic)

        # BIC should be higher than AIC (penalizes complexity more)
        assert bic > aic

    def test_compute_aic_bic_gamma(self, gamma_data):
        """Test computing AIC and BIC for gamma distribution."""
        dist = st.gamma
        params = dist.fit(gamma_data)

        aic, bic = compute_information_criteria(dist, params, gamma_data)

        # Should return finite values
        assert np.isfinite(aic)
        assert np.isfinite(bic)

    def test_compute_aic_bic_invalid_data(self):
        """Test information criteria with invalid data."""
        dist = st.norm
        params = (0, 1)
        data = np.array([np.nan, np.inf])

        aic, bic = compute_information_criteria(dist, params, data)

        # Should return inf for invalid data
        assert aic == np.inf
        assert bic == np.inf

    def test_aic_bic_relationship(self, normal_data):
        """Test that AIC and BIC have expected relationship."""
        dist = st.norm
        params = dist.fit(normal_data)

        aic, bic = compute_information_criteria(dist, params, normal_data)

        # For normal distribution (2 params) and large data:
        # BIC = k*ln(n) - 2*ln(L)
        # AIC = 2*k - 2*ln(L)
        # BIC penalty is larger when n is large

        assert bic > aic  # BIC penalizes more for large n


class TestCreateSampleData:
    """Tests for data sampling."""

    def test_create_sample_small_data(self):
        """Test sampling when data is smaller than sample size."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=1000, random_seed=42)

        # Should return all data
        assert len(sample) == len(data)
        assert np.array_equal(sample, data)

    def test_create_sample_large_data(self):
        """Test sampling when data is larger than sample size."""
        data = np.arange(100_000)
        sample = create_sample_data(data, sample_size=10_000, random_seed=42)

        # Should return sampled data
        assert len(sample) == 10_000
        assert len(sample) < len(data)

        # All sampled values should be from original data
        assert np.all(np.isin(sample, data))

    def test_create_sample_reproducible(self):
        """Test that sampling is reproducible with same seed."""
        data = np.arange(100_000)

        sample1 = create_sample_data(data, sample_size=10_000, random_seed=42)
        sample2 = create_sample_data(data, sample_size=10_000, random_seed=42)

        # Should be identical
        assert np.array_equal(sample1, sample2)

    def test_create_sample_different_seeds(self):
        """Test that different seeds produce different samples."""
        data = np.arange(100_000)

        sample1 = create_sample_data(data, sample_size=10_000, random_seed=42)
        sample2 = create_sample_data(data, sample_size=10_000, random_seed=123)

        # Should be different
        assert not np.array_equal(sample1, sample2)

    def test_create_sample_no_replacement(self):
        """Test that sampling is without replacement."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=50, random_seed=42)

        # Should have no duplicates
        assert len(sample) == len(np.unique(sample))


class TestFitSingleDistributionEdgeCases:
    """Edge case tests for fit_single_distribution."""

    def test_fit_with_negative_data_for_positive_dist(self):
        """Test fitting positive-only distribution to negative data."""
        data = np.array([-5.0, -3.0, -1.0, 0.0, 1.0])
        y_hist = np.array([0.2, 0.3, 0.3, 0.2])
        x_hist = np.array([-4.0, -2.0, 0.0, 0.5])

        # expon requires positive data, may fail or produce poor fit
        result = fit_single_distribution("expon", data, x_hist, y_hist)

        assert result["distribution"] == "expon"
        # Either succeeds with a result or fails gracefully
        assert result["sse"] >= 0 or result["sse"] == np.inf

    def test_fit_with_empty_params_distribution(self, normal_data):
        """Test distributions with different parameter structures."""
        y_hist, x_edges = np.histogram(normal_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        # Test uniform distribution (only loc and scale)
        result = fit_single_distribution("uniform", normal_data, x_hist, y_hist)

        assert result["distribution"] == "uniform"
        assert len(result["parameters"]) >= 2  # at least loc, scale

    def test_fit_returns_correct_structure(self, normal_data):
        """Test that fit returns dict with all required keys."""
        y_hist, x_edges = np.histogram(normal_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        result = fit_single_distribution("norm", normal_data, x_hist, y_hist)

        required_keys = {"distribution", "parameters", "sse", "aic", "bic"}
        assert set(result.keys()) == required_keys

    def test_fit_sse_is_float(self, normal_data):
        """Test that SSE is returned as float."""
        y_hist, x_edges = np.histogram(normal_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        result = fit_single_distribution("norm", normal_data, x_hist, y_hist)

        assert isinstance(result["sse"], float)

    def test_fit_parameters_are_floats(self, normal_data):
        """Test that all parameters are returned as floats."""
        y_hist, x_edges = np.histogram(normal_data, bins=50, density=True)
        x_hist = (x_edges[:-1] + x_edges[1:]) / 2

        result = fit_single_distribution("norm", normal_data, x_hist, y_hist)

        for param in result["parameters"]:
            assert isinstance(param, float)


class TestEvaluatePDFEdgeCases:
    """Edge case tests for evaluate_pdf."""

    def test_evaluate_pdf_empty_x(self):
        """Test PDF evaluation with empty x array."""
        dist = st.norm
        params = (0, 1)
        x = np.array([])

        pdf_values = evaluate_pdf(dist, params, x)

        assert len(pdf_values) == 0

    def test_evaluate_pdf_single_point(self):
        """Test PDF evaluation at single point."""
        dist = st.norm
        params = (0, 1)
        x = np.array([0])

        pdf_values = evaluate_pdf(dist, params, x)

        assert len(pdf_values) == 1
        assert pdf_values[0] > 0

    def test_evaluate_pdf_inf_handling(self):
        """Test PDF handles inf values in x."""
        dist = st.norm
        params = (0, 1)
        x = np.array([np.inf, -np.inf, 0])

        pdf_values = evaluate_pdf(dist, params, x)

        # inf should be converted to 0
        assert np.all(np.isfinite(pdf_values))

    def test_evaluate_pdf_extreme_x_values(self):
        """Test PDF with extreme but finite x values."""
        dist = st.norm
        params = (0, 1)
        x = np.array([-1000, -100, 0, 100, 1000])

        pdf_values = evaluate_pdf(dist, params, x)

        # Should all be finite (near 0 for extreme values)
        assert np.all(np.isfinite(pdf_values))
        assert np.all(pdf_values >= 0)


class TestComputeInformationCriteriaEdgeCases:
    """Edge case tests for compute_information_criteria."""

    def test_compute_with_small_sample(self):
        """Test information criteria with very small sample."""
        dist = st.norm
        data = np.array([1.0, 2.0, 3.0])
        params = dist.fit(data)

        aic, bic = compute_information_criteria(dist, params, data)

        # Should return finite values for small sample
        assert np.isfinite(aic) or aic == np.inf
        assert np.isfinite(bic) or bic == np.inf

    def test_compute_with_single_point(self):
        """Test information criteria with single data point."""
        dist = st.norm
        data = np.array([1.0])
        params = (0, 1)  # Use fixed params since can't fit with 1 point

        aic, bic = compute_information_criteria(dist, params, data)

        # May return inf due to numerical issues
        assert isinstance(aic, (int, float))
        assert isinstance(bic, (int, float))

    def test_bic_larger_than_aic_for_large_n(self, normal_data):
        """Test that BIC > AIC for large sample sizes (n > 7)."""
        dist = st.norm
        params = dist.fit(normal_data)

        aic, bic = compute_information_criteria(dist, params, normal_data)

        # For n > e^2 ≈ 7.4, BIC penalty is larger
        if len(normal_data) > 8:
            assert bic > aic

    def test_compute_returns_tuple(self, normal_data):
        """Test that function returns tuple of two values."""
        dist = st.norm
        params = dist.fit(normal_data)

        result = compute_information_criteria(dist, params, normal_data)

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestCreateSampleDataEdgeCases:
    """Edge case tests for create_sample_data."""

    def test_create_sample_exact_size(self):
        """Test sampling when data size equals sample size."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=100, random_seed=42)

        assert len(sample) == 100
        assert np.array_equal(sample, data)

    def test_create_sample_size_one(self):
        """Test sampling to size 1."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=1, random_seed=42)

        assert len(sample) == 1
        assert sample[0] in data

    def test_create_sample_empty_data(self):
        """Test sampling from empty array."""
        data = np.array([])
        sample = create_sample_data(data, sample_size=10, random_seed=42)

        assert len(sample) == 0

    def test_create_sample_preserves_dtype(self):
        """Test that sampling preserves data type."""
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        sample = create_sample_data(data, sample_size=3, random_seed=42)

        assert sample.dtype == data.dtype
