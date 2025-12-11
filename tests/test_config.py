"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from spark_dist_fit.config import FitConfig, PlotConfig


class TestFitConfig:
    """Tests for FitConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = FitConfig()

        assert config.bins == 50
        assert config.use_rice_rule is True
        assert config.support_at_zero is False
        assert config.enable_sampling is True
        assert config.sample_fraction is None
        assert config.max_sample_size == 1_000_000
        assert config.adaptive_strategy is True
        assert config.local_threshold == 100_000
        assert config.spark_threshold == 10_000_000
        assert config.num_partitions is None
        assert config.random_seed == 42

    def test_default_exclusions(self):
        """Test that default exclusions are set."""
        config = FitConfig()

        expected_exclusions = [
            "levy_stable",
            "kappa4",
            "ncx2",
            "ksone",
            "ncf",
            "wald",
            "mielke",
            "exonpow",
        ]

        assert len(config.excluded_distributions) == len(expected_exclusions)
        for dist in expected_exclusions:
            assert dist in config.excluded_distributions

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = FitConfig(
            bins=100,
            use_rice_rule=False,
            support_at_zero=True,
            enable_sampling=False,
            sample_fraction=0.3,
            random_seed=123,
        )

        assert config.bins == 100
        assert config.use_rice_rule is False
        assert config.support_at_zero is True
        assert config.enable_sampling is False
        assert config.sample_fraction == 0.3
        assert config.random_seed == 123

    def test_load_from_hocon_file(self):
        """Test loading config from HOCON file."""
        hocon_content = """
        bins = 75
        use_rice_rule = false
        support_at_zero = true
        enable_sampling = true
        sample_fraction = 0.25
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write(hocon_content)
            temp_path = f.name

        try:
            config = FitConfig.from_file(temp_path)

            assert config.bins == 75
            assert config.use_rice_rule is False
            assert config.support_at_zero is True
            assert config.sample_fraction == 0.25
        finally:
            Path(temp_path).unlink()

    def test_load_from_yaml_file(self):
        """Test loading config from YAML file."""
        yaml_content = """
bins: 60
use_rice_rule: false
support_at_zero: false
enable_sampling: true
sample_fraction: 0.4
random_seed: 999
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = FitConfig.from_file(temp_path)

            assert config.bins == 60
            assert config.use_rice_rule is False
            assert config.sample_fraction == 0.4
            assert config.random_seed == 999
        finally:
            Path(temp_path).unlink()

    def test_load_from_string(self):
        """Test loading config from string."""
        config_str = """
        bins = 80
        support_at_zero = true
        """

        config = FitConfig.from_string(config_str)

        assert config.bins == 80
        assert config.support_at_zero is True

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            FitConfig.from_file("nonexistent_file.conf")

    def test_custom_exclusions(self):
        """Test adding custom exclusions."""
        config = FitConfig(excluded_distributions=["norm", "expon"])

        assert "norm" in config.excluded_distributions
        assert "expon" in config.excluded_distributions
        assert len(config.excluded_distributions) == 2

    def test_bins_as_list(self):
        """Test setting bins as a list."""
        custom_bins = [0, 10, 20, 30, 40, 50]
        config = FitConfig(bins=custom_bins)

        assert config.bins == custom_bins


class TestPlotConfig:
    """Tests for PlotConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = PlotConfig()

        assert config.figsize == (12, 8)
        assert config.dpi == 600
        assert config.show_histogram is True
        assert config.histogram_alpha == 0.5
        assert config.pdf_linewidth == 2
        assert config.save_format == "png"
        assert config.title_fontsize == 14
        assert config.label_fontsize == 12
        assert config.legend_fontsize == 10
        assert config.grid_alpha == 0.3

    def test_custom_values(self):
        """Test creating plot config with custom values."""
        config = PlotConfig(
            figsize=(16, 10),
            dpi=300,
            show_histogram=False,
            histogram_alpha=0.7,
            pdf_linewidth=3,
            save_format="pdf",
        )

        assert config.figsize == (16, 10)
        assert config.dpi == 300
        assert config.show_histogram is False
        assert config.histogram_alpha == 0.7
        assert config.pdf_linewidth == 3
        assert config.save_format == "pdf"

    def test_load_from_file(self):
        """Test loading plot config from file."""
        hocon_content = """
        figsize = [16, 10]
        dpi = 300
        show_histogram = false
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write(hocon_content)
            temp_path = f.name

        try:
            config = PlotConfig.from_file(temp_path)

            assert config.figsize == (16, 10)
            assert config.dpi == 300
            assert config.show_histogram is False
        finally:
            Path(temp_path).unlink()

    def test_invalid_format(self):
        """Test that invalid save format is accepted (no validation)."""
        config = PlotConfig(save_format="invalid")

        assert config.save_format == "invalid"  # No validation in dataclass

    def test_load_from_string(self):
        """Test loading PlotConfig from HOCON string."""
        config_str = """
        dpi = 300
        show_histogram = false
        histogram_alpha = 0.8
        pdf_linewidth = 4
        """

        config = PlotConfig.from_string(config_str)

        assert config.dpi == 300
        assert config.show_histogram is False
        assert config.histogram_alpha == 0.8
        assert config.pdf_linewidth == 4

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing PlotConfig file."""
        with pytest.raises(FileNotFoundError):
            PlotConfig.from_file("nonexistent_plot_config.conf")

    def test_from_string_partial_config(self):
        """Test loading partial config from string (uses defaults for missing)."""
        config_str = """
        dpi = 150
        """

        config = PlotConfig.from_string(config_str)

        assert config.dpi == 150
        # Defaults should be preserved
        assert config.show_histogram is True
        assert config.pdf_linewidth == 2
        assert config.figsize == (12, 8)  # default figsize also preserved


class TestFitConfigEdgeCases:
    """Edge case tests for FitConfig."""

    def test_empty_exclusions(self):
        """Test FitConfig with empty exclusions list."""
        config = FitConfig(excluded_distributions=[])

        assert config.excluded_distributions == []

    def test_num_partitions_explicit(self):
        """Test setting explicit num_partitions."""
        config = FitConfig(num_partitions=50)

        assert config.num_partitions == 50

    def test_all_sampling_disabled(self):
        """Test config with sampling completely disabled."""
        config = FitConfig(enable_sampling=False, adaptive_strategy=False)

        assert config.enable_sampling is False
        assert config.adaptive_strategy is False

    def test_from_string_with_all_fields(self):
        """Test loading FitConfig with all fields from string."""
        config_str = """
        bins = 100
        use_rice_rule = false
        support_at_zero = true
        enable_sampling = false
        sample_fraction = 0.5
        max_sample_size = 500000
        adaptive_strategy = false
        local_threshold = 50000
        spark_threshold = 5000000
        num_partitions = 20
        random_seed = 123
        """

        config = FitConfig.from_string(config_str)

        assert config.bins == 100
        assert config.use_rice_rule is False
        assert config.support_at_zero is True
        assert config.enable_sampling is False
        assert config.sample_fraction == 0.5
        assert config.max_sample_size == 500000
        assert config.adaptive_strategy is False
        assert config.local_threshold == 50000
        assert config.spark_threshold == 5000000
        assert config.num_partitions == 20
        assert config.random_seed == 123
