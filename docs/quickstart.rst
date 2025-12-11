Quick Start
===========

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
   fitter.plot(best, title="Best Fit Distribution")

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
