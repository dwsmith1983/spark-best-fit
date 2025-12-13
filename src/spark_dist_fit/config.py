"""Configuration management using dataclasses with HOCON/YAML/JSON support via dataconf."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, TypeVar, Union

from dataconf import load, string

from spark_dist_fit.distributions import DistributionRegistry

T = TypeVar("T", bound="ConfigLoadMixin")


class ConfigLoadMixin:
    """Mixin providing file and string loading for config dataclasses."""

    @classmethod
    def from_file(cls: type[T], path: Union[str, Path]) -> T:
        """Load configuration from HOCON, YAML, or JSON file.

        Args:
            path: Path to configuration file

        Returns:
            Config instance loaded from file

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return load(str(path), cls)

    @classmethod
    def from_string(cls: type[T], content: str) -> T:
        """Load configuration from HOCON/YAML/JSON string.

        Args:
            content: Configuration string

        Returns:
            Config instance loaded from string
        """
        return string(content, cls)


@dataclass(frozen=True)
class SparkConfig(ConfigLoadMixin):
    """Configuration for Spark session settings.

    These settings are applied to the Spark session when using DistributionFitter.

    Example HOCON config::

        spark {
            app_name = "spark-dist-fit"
            arrow_enabled = true
            adaptive_enabled = true
            adaptive_coalesce_enabled = true
        }

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
    extra_config: Tuple[Tuple[str, str], ...] = ()  # Immutable tuple of key-value pairs

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
        config.update(dict(self.extra_config))
        return config


# Re-export DEFAULT_EXCLUSIONS as a tuple for backwards compatibility
DEFAULT_EXCLUDED_DISTRIBUTIONS: Tuple[str, ...] = tuple(DistributionRegistry.DEFAULT_EXCLUSIONS)


@dataclass(frozen=True)
class FitConfig(ConfigLoadMixin):
    """Configuration for distribution fitting.

    Can be loaded from HOCON, YAML, or JSON files using ``FitConfig.from_file()``,
    or created programmatically. This is an immutable (frozen) dataclass.

    Example HOCON config::

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

    To customize excluded distributions (e.g., to include a slow distribution)::

        from spark_dist_fit import FitConfig, DEFAULT_EXCLUDED_DISTRIBUTIONS

        # Remove a specific distribution from exclusions
        exclusions = [d for d in DEFAULT_EXCLUDED_DISTRIBUTIONS if d != "studentized_range"]
        config = FitConfig(excluded_distributions=tuple(exclusions))

        # Or use minimal exclusions (warning: some distributions are very slow)
        config = FitConfig(excluded_distributions=("levy_stable", "kappa4"))

        # Or no exclusions (warning: fitting may hang on slow distributions)
        config = FitConfig(excluded_distributions=())

    Attributes:
        bins: Number of histogram bins or array of bin edges
        use_rice_rule: Use Rice rule to automatically determine bin count
        support_at_zero: Only fit distributions with support at zero (non-negative)
        excluded_distributions: Tuple of scipy distribution names to exclude from fitting.
            Defaults to DEFAULT_EXCLUDED_DISTRIBUTIONS which excludes slow distributions.
        enable_sampling: Enable sampling for large datasets
        sample_fraction: Fraction of data to sample (None = auto-determine)
        max_sample_size: Maximum number of rows to sample
        max_sample_fraction: Maximum auto-determined sample fraction (caps auto sampling)
        sample_threshold: Row count above which sampling is applied
        num_partitions: Number of Spark partitions (None = auto-determine)
        random_seed: Random seed for reproducible sampling
    """

    bins: Union[int, Tuple[float, ...]] = 50
    use_rice_rule: bool = True
    support_at_zero: bool = False
    excluded_distributions: Tuple[str, ...] = DEFAULT_EXCLUDED_DISTRIBUTIONS
    enable_sampling: bool = True
    sample_fraction: Optional[float] = None
    max_sample_size: int = 1_000_000
    max_sample_fraction: float = 0.35
    sample_threshold: int = 10_000_000
    num_partitions: Optional[int] = None
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate bins
        if isinstance(self.bins, int):
            if self.bins <= 0:
                raise ValueError(f"bins must be positive, got {self.bins}")
        elif isinstance(self.bins, tuple):
            if len(self.bins) < 2:
                raise ValueError(f"bins array must have at least 2 edges, got {len(self.bins)}")

        # Validate sample_fraction
        if self.sample_fraction is not None:
            if not 0.0 < self.sample_fraction <= 1.0:
                raise ValueError(f"sample_fraction must be in (0, 1], got {self.sample_fraction}")

        # Validate max_sample_size
        if self.max_sample_size <= 0:
            raise ValueError(f"max_sample_size must be positive, got {self.max_sample_size}")

        # Validate sample_threshold
        if self.sample_threshold <= 0:
            raise ValueError(f"sample_threshold must be positive, got {self.sample_threshold}")

        # Validate max_sample_fraction
        if not 0.0 < self.max_sample_fraction <= 1.0:
            raise ValueError(f"max_sample_fraction must be in (0, 1], got {self.max_sample_fraction}")

        # Validate num_partitions
        if self.num_partitions is not None and self.num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {self.num_partitions}")


@dataclass(frozen=True)
class PlotConfig(ConfigLoadMixin):
    """Configuration for plotting fitted distributions.

    Can be loaded from HOCON, YAML, or JSON files using ``PlotConfig.from_file()``,
    or created programmatically. This is an immutable (frozen) dataclass.

    Example HOCON config::

        plot {
            figsize = [12, 8]
            dpi = 600
            show_histogram = true
            histogram_alpha = 0.5
            pdf_linewidth = 2
        }

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

    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 600
    show_histogram: bool = True
    histogram_alpha: float = 0.5
    pdf_linewidth: int = 2
    save_format: str = "png"
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid_alpha: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate figsize
        if len(self.figsize) != 2:
            raise ValueError(f"figsize must be (width, height), got {self.figsize}")
        if self.figsize[0] <= 0 or self.figsize[1] <= 0:
            raise ValueError(f"figsize dimensions must be positive, got {self.figsize}")

        # Validate dpi
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")

        # Validate alpha values (0-1)
        if not 0.0 <= self.histogram_alpha <= 1.0:
            raise ValueError(f"histogram_alpha must be in [0, 1], got {self.histogram_alpha}")
        if not 0.0 <= self.grid_alpha <= 1.0:
            raise ValueError(f"grid_alpha must be in [0, 1], got {self.grid_alpha}")

        # Validate linewidth
        if self.pdf_linewidth <= 0:
            raise ValueError(f"pdf_linewidth must be positive, got {self.pdf_linewidth}")

        # Validate font sizes
        if self.title_fontsize <= 0:
            raise ValueError(f"title_fontsize must be positive, got {self.title_fontsize}")
        if self.label_fontsize <= 0:
            raise ValueError(f"label_fontsize must be positive, got {self.label_fontsize}")
        if self.legend_fontsize <= 0:
            raise ValueError(f"legend_fontsize must be positive, got {self.legend_fontsize}")

        # Validate save_format
        valid_formats = ("png", "pdf", "svg", "jpg", "jpeg")
        if self.save_format.lower() not in valid_formats:
            raise ValueError(f"save_format must be one of {valid_formats}, got {self.save_format}")


@dataclass(frozen=True)
class AppConfig(ConfigLoadMixin):
    """Root configuration containing all config sections.

    This is the recommended way to load configuration from a file that
    contains spark{}, fit{}, and plot{} sections.

    Example HOCON config::

        spark {
            app_name = "my-app"
            arrow_enabled = true
        }
        fit {
            bins = 100
            support_at_zero = true
        }
        plot {
            figsize = [16, 10]
            dpi = 300
        }

    Usage::

        config = AppConfig.from_file("config/app.conf")
        fitter = DistributionFitter(config=config.fit, spark_config=config.spark)
        fitter.plot(result, df, "value", config=config.plot)

    Attributes:
        spark: Spark session configuration
        fit: Distribution fitting configuration
        plot: Plotting configuration
    """

    spark: SparkConfig = field(default_factory=SparkConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
