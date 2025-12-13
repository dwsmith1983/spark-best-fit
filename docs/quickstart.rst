Quick Start
===========

Requirements
------------

Compatibility Matrix
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Spark Version
     - Python Versions
     - NumPy
     - Pandas
     - PyArrow
   * - **3.5.x**
     - 3.11, 3.12
     - 1.24+ (< 2.0)
     - 1.5+
     - 12.0 - 16.x
   * - **4.0.x**
     - 3.12, 3.13
     - 2.0+
     - 2.2+
     - 17.0+

.. note::
   Spark 3.5.x does not support NumPy 2.0. If using Spark 3.5 with Python 3.12, ensure ``setuptools`` is installed (provides ``distutils``).

Installation
------------

.. code-block:: bash

   pip install spark-dist-fit

Basic Usage
-----------

.. code-block:: python

   from spark_dist_fit import DistributionFitter
   import numpy as np
   from pyspark.sql import SparkSession

   spark = SparkSession.builder.getOrCreate()

   # Generate sample data
   data = np.random.normal(loc=50, scale=10, size=10_000)

   # Create fitter
   fitter = DistributionFitter()
   df = spark.createDataFrame([(float(x),) for x in data], ["value"])

   # Fit distributions
   results = fitter.fit(df, column="value")

   # Get best fit
   best = results.best(n=1)[0]
   print(f"Best: {best.distribution} with SSE={best.sse:.6f}")

   # Plot
   fitter.plot(best, df, "value", title="Best Fit Distribution")

Custom Configuration
--------------------

.. code-block:: python

   from spark_dist_fit import DistributionFitter, FitConfig

   config = FitConfig(
       bins=100,
       support_at_zero=True,
       excluded_distributions=["levy_stable", "kappa4"],
   )

   fitter = DistributionFitter(config=config)
   results = fitter.fit(df, column="value")

Configuration from HOCON File
-----------------------------

The recommended approach for production is to use HOCON configuration files.
HOCON supports includes, substitutions, and environment variables.

Create a config file (e.g., ``config/app.conf``):

.. code-block:: text

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

Load and use the configuration:

.. code-block:: python

   from spark_dist_fit import AppConfig, DistributionFitter

   # Load configs from HOCON file (nested structure requires AppConfig)
   config = AppConfig.from_file("config/app.conf")

   # Use in fitter
   fitter = DistributionFitter(config=config.fit)
   results = fitter.fit(df, column="value")

   # Plot with config
   best = results.best(n=1)[0]
   fitter.plot(best, df, "value", config=config.plot)

HOCON also supports environment variable substitution:

.. code-block:: text

   fit {
       random_seed = ${?RANDOM_SEED}  # Uses env var if set
       bins = ${?HIST_BINS}
   }

And file includes for sharing common configuration:

.. code-block:: text

   include "base.conf"

   fit {
       # Override specific values from base
       bins = 200
   }

See ``config/example.conf`` in the repository for a complete example.
