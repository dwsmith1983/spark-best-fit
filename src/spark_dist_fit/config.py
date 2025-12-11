"""Configuration management using dataclasses with HOCON/YAML/JSON support via dataconf."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from dataconf import load, string


@dataclass
class SparkConfig:
    """Configuration for Spark session settings.

    These settings are applied to the Spark session when using DistributionFitter.

    Example HOCON config:
        ```
        spark {
            app_name = "spark-dist-fit"
            arrow_enabled = true
            adaptive_enabled = true
            adaptive_coalesce_enabled = true
        }
        ```

    Attributes:
        app_name: Spark application name
        arrow_enabled: Enable Arrow optimization for Pandas UDFs
        adaptive_enabled: Enable Adaptive Query Execution
        adaptive_coalesce_enabled: Enable adaptive partition coalescing
        extra_config: Additional Spark config key-value pairs
    """

    app_name: str = "spark-dist-fit"
    arrow_enabled: bool = True
    adaptive_enabled: bool = True
    adaptive_coalesce_enabled: bool = True
    extra_config: Dict[str, str] = field(default_factory=dict)

    def to_spark_config(self) -> Dict[str, str]:
        """Convert to Spark config dictionary.

        Returns:
            Dictionary of Spark config key-value pairs
        """
        config = {
            "spark.sql.execution.arrow.pyspark.enabled": str(self.arrow_enabled).lower(),
            "spark.sql.adaptive.enabled": str(self.adaptive_enabled).lower(),
            "spark.sql.adaptive.coalescePartitions.enabled": str(self.adaptive_coalesce_enabled).lower(),
        }
        config.update(self.extra_config)
        return config

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SparkConfig":
        """Load configuration from HOCON, YAML, or JSON file.

        Args:
            path: Path to configuration file

        Returns:
            SparkConfig instance loaded from file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        return load(str(path), cls)

    @classmethod
    def from_string(cls, content: str) -> "SparkConfig":
        """Load configuration from HOCON/YAML/JSON string.

        Args:
            content: Configuration string

        Returns:
            SparkConfig instance loaded from string
        """
        return string(content, cls)


@dataclass
class FitConfig:
    """Configuration for distribution fitting.

    Can be loaded from HOCON, YAML, or JSON files using FitConfig.from_file(),
    or created programmatically.

    Example HOCON config:
        ```
        fit {
            bins = 100
            support_at_zero = true
            excluded_distributions = ["levy_stable", "kappa4"]
            sampling {
                enable_sampling = true
                sample_fraction = 0.35
                max_sample_size = 1000000
            }
        }
        ```

    Attributes:
        bins: Number of histogram bins or array of bin edges
        use_rice_rule: Use Rice rule to automatically determine bin count
        support_at_zero: Only fit distributions with support at zero (non-negative)
        excluded_distributions: List of scipy distribution names to exclude from fitting
        enable_sampling: Enable sampling for large datasets
        sample_fraction: Fraction of data to sample (None = auto-determine)
        max_sample_size: Maximum number of rows to sample
        sample_threshold: Row count above which sampling is applied
        num_partitions: Number of Spark partitions (None = auto-determine)
        random_seed: Random seed for reproducible sampling
    """

    bins: Union[int, List[float], np.ndarray] = 50
    use_rice_rule: bool = True
    support_at_zero: bool = False
    excluded_distributions: List[str] = field(
        default_factory=lambda: [
            "levy_stable",
            "kappa4",
            "ncx2",
            "ksone",
            "ncf",
            "wald",
            "mielke",
            "exonpow",
        ]
    )
    enable_sampling: bool = True
    sample_fraction: Optional[float] = None
    max_sample_size: int = 1_000_000
    sample_threshold: int = 10_000_000
    num_partitions: Optional[int] = None
    random_seed: int = 42

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "FitConfig":
        """Load configuration from HOCON, YAML, or JSON file.

        Args:
            path: Path to configuration file

        Returns:
            FitConfig instance loaded from file

        Example:
            >>> config = FitConfig.from_file("config.conf")
            >>> config = FitConfig.from_file("config.yaml")
            >>> config = FitConfig.from_file("config.json")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        return load(str(path), cls)

    @classmethod
    def from_string(cls, content: str) -> "FitConfig":
        """Load configuration from HOCON/YAML/JSON string.

        Args:
            content: Configuration string

        Returns:
            FitConfig instance loaded from string
        """
        return string(content, cls)


@dataclass
class PlotConfig:
    """Configuration for plotting fitted distributions.

    Can be loaded from HOCON, YAML, or JSON files using PlotConfig.from_file(),
    or created programmatically.

    Example HOCON config:
        ```
        plot {
            figsize = [12, 8]
            dpi = 600
            show_histogram = true
            histogram_alpha = 0.5
            pdf_linewidth = 2
        }
        ```

    Attributes:
        figsize: Figure size as (width, height) tuple
        dpi: Dots per inch for saved figures
        show_histogram: Show data histogram in plot
        histogram_alpha: Transparency of histogram bars (0-1)
        pdf_linewidth: Line width for PDF curve
        save_format: File format for saving plots (png, pdf, svg)
        title_fontsize: Font size for plot title
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend
        grid_alpha: Transparency of grid lines (0-1)
    """

    figsize: Union[Tuple[int, int], List[int]] = (12, 8)
    dpi: int = 600
    show_histogram: bool = True
    histogram_alpha: float = 0.5
    pdf_linewidth: int = 2
    save_format: str = "png"
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid_alpha: float = 0.3

    def __post_init__(self):
        """Convert figsize from list to tuple if needed (dataconf parses HOCON lists as Python lists)."""
        if isinstance(self.figsize, list):
            self.figsize = tuple(self.figsize)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PlotConfig":
        """Load configuration from HOCON, YAML, or JSON file.

        Args:
            path: Path to configuration file

        Returns:
            PlotConfig instance loaded from file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        return load(str(path), cls)

    @classmethod
    def from_string(cls, content: str) -> "PlotConfig":
        """Load configuration from HOCON/YAML/JSON string.

        Args:
            content: Configuration string

        Returns:
            PlotConfig instance loaded from string
        """
        return string(content, cls)
