# spark-dist-fit

**Modern Spark 4 distribution fitting library with efficient parallel processing**

Automatically fit ~100 scipy.stats distributions to your data using Apache Spark's distributed computing power, with
optimized Pandas UDFs.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Spark 4.0+](https://img.shields.io/badge/spark-4.0+-orange.svg)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

**Key Optimizations**
- **No data collection**: Computes histograms distributedly, never collects raw data to driver
- **Pandas UDFs**: Uses vectorized Pandas UDFs with Apache Arrow for 10-100x faster processing
- **Broadcast optimization**: Broadcasts tiny histogram (~1KB) instead of duplicating full dataset
- **Zero cross joins**: Completely eliminates expensive cartesian products
- **Adaptive sampling**: Intelligently samples large datasets (100M+ rows) without accuracy loss

**Capabilities**
- Fits ~100 scipy.stats continuous distributions in parallel
- Handles datasets from 100K to 100M+ rows efficiently
- Returns multiple goodness-of-fit metrics (SSE, AIC, BIC)
- Beautiful matplotlib visualizations
- HOCON/YAML/JSON configuration support
- Type-safe configuration with dataclasses

## Installation

```bash
# From source
git clone <repository-url>
cd spark-best-fit
make install

# For development
make install-dev
```

## Quick Start

```python
from pyspark.sql import SparkSession
from spark_dist_fit import DistributionFitter
import numpy as np

# Create Spark session
spark = SparkSession.builder.appName("DistFit").getOrCreate()

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=100_000)
df = spark.createDataFrame([(float(x),) for x in data], ["value"])

# Fit distributions
fitter = DistributionFitter(spark)
results = fitter.fit(df, column="value")

# Get best distribution
best = results.best(n=1)[0]
print(f"Best: {best.distribution} with SSE={best.sse:.6f}")

# Plot
fitter.plot(best, title="Best Fit Distribution")
```

## Advanced Usage

### Custom Configuration

```python
from spark_dist_fit import DistributionFitter, FitConfig

# Configure fitting parameters
config = FitConfig(
    bins=100,                    # Number of histogram bins
    support_at_zero=True,        # Only fit non-negative distributions
    enable_sampling=True,         # Enable adaptive sampling
    sample_fraction=0.3,         # Sample 30% of data
    excluded_distributions=[     # Exclude specific distributions
        "levy_stable",
        "kappa4",
    ]
)

fitter = DistributionFitter(config=config)
results = fitter.fit(df, column="value")
```

### Load Configuration from HOCON File

The recommended approach for production is to use HOCON configuration files. See `config/example.conf` for a full example.

**config/my_config.conf:**
```hocon
fit {
    bins = 100
    use_rice_rule = false
    support_at_zero = true

    excluded_distributions = [
        "levy_stable"
        "kappa4"
        "ncx2"
    ]

    enable_sampling = true
    sample_fraction = 0.3
    max_sample_size = 1000000

    random_seed = 42
}

plot {
    figsize = [16, 10]
    dpi = 300
    histogram_alpha = 0.6
    pdf_linewidth = 3
}
```

**Load and use:**
```python
from spark_dist_fit import DistributionFitter, FitConfig, PlotConfig

# Load configs from HOCON file
fit_config = FitConfig.from_file("config/my_config.conf")
plot_config = PlotConfig.from_file("config/my_config.conf")

# Use in fitter
fitter = DistributionFitter(config=fit_config)
results = fitter.fit(df, column="value")

# Plot with config
fitter.plot(results.best(n=1)[0], config=plot_config)
```

HOCON supports includes, substitutions, and environment variables:
```hocon
# Include shared config
include "base.conf"

fit {
    # Reference environment variable
    random_seed = ${?RANDOM_SEED}

    # Override from base
    bins = 200
}
```

### Working with Results

```python
# Get top 5 distributions by SSE
top_5 = results.best(n=5, metric="sse")

# Get best by AIC
best_aic = results.best(n=1, metric="aic")[0]

# Filter good fits
good_fits = results.filter(sse_threshold=0.01)

# Convert to pandas for analysis
df_pandas = results.to_pandas()

# Use fitted distribution
samples = best.sample(size=10000)  # Generate samples
pdf_values = best.pdf(x_array)     # Evaluate PDF
cdf_values = best.cdf(x_array)     # Evaluate CDF
```

### Plotting

```python
from spark_dist_fit import PlotConfig

plot_config = PlotConfig(
    figsize=(16, 10),
    dpi=300,
    histogram_alpha=0.6,
    pdf_linewidth=3
)

fitter.plot(
    best,
    df,
    "value",
    config=plot_config,
    title="Distribution Fit",
    xlabel="Value",
    ylabel="Density"
)
```

## Architecture

### Key Components

1. **DistributionFitter**: Main orchestrator
2. **HistogramComputer**: Distributed histogram computation
3. **Pandas UDFs**: Vectorized distribution fitting
4. **Broadcast Variables**: Efficient data sharing
5. **FitResults**: Convenient result handling

## Configuration

### FitConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bins` | int/array | 50 | Number of histogram bins |
| `use_rice_rule` | bool | True | Auto-calculate bins using Rice rule |
| `support_at_zero` | bool | False | Only fit non-negative distributions |
| `excluded_distributions` | list | [...] | Distributions to exclude |
| `enable_sampling` | bool | True | Enable adaptive sampling |
| `sample_fraction` | float | None | Fraction to sample (None = auto) |
| `max_sample_size` | int | 1,000,000 | Max rows to sample |
| `adaptive_strategy` | bool | True | Enable adaptive processing |
| `num_partitions` | int | None | Spark partitions (None = auto) |
| `random_seed` | int | 42 | Random seed |

### PlotConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | tuple | (12, 8) | Figure size (width, height) |
| `dpi` | int | 600 | Dots per inch |
| `show_histogram` | bool | True | Show data histogram |
| `histogram_alpha` | float | 0.5 | Histogram transparency |
| `pdf_linewidth` | int | 2 | PDF line width |
| `save_format` | str | "png" | Save format (png/pdf/svg) |

## Development

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd spark-best-fit

# Setup development environment
make setup

# This will:
# - Install package in editable mode
# - Install dev dependencies
# - Setup pre-commit hooks
```

### Development Commands

```bash
make help              # Show all available commands
make test              # Run tests
make test-cov          # Run tests with coverage
make pre-commit        # Run pre-commit hooks (lint, format, type check)
make check             # Run pre-commit + tests
make clean             # Clean build artifacts
make build             # Build distribution packages
```

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:
- **ruff**: Fast Python linter
- **black**: Code formatting
- **isort**: Import sorting
- **mypy**: Static type checking

Hooks run automatically on commit. To run manually:
```bash
make pre-commit
```

### Code Style

- Line length: 120 characters
- Python: 3.11+
- Type hints: Encouraged
- Docstrings: Google style

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
pytest tests/test_core.py -v
```

## License

MIT License - see LICENSE file for details
