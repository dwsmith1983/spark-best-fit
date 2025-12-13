API Reference
=============

Core
----

.. automodule:: spark_dist_fit.core
   :members:
   :exclude-members: plot_config

Configuration
-------------

.. autoclass:: spark_dist_fit.config.FitConfig
   :show-inheritance:
   :no-index:

.. autoclass:: spark_dist_fit.config.PlotConfig
   :show-inheritance:
   :no-index:

.. autoclass:: spark_dist_fit.config.SparkConfig
   :members: to_spark_config
   :show-inheritance:
   :no-index:

.. autoclass:: spark_dist_fit.config.AppConfig
   :show-inheritance:
   :no-index:

.. autoclass:: spark_dist_fit.config.ConfigLoadMixin
   :members:
   :show-inheritance:

.. autodata:: spark_dist_fit.config.DEFAULT_EXCLUDED_DISTRIBUTIONS

Results
-------

.. autoclass:: spark_dist_fit.results.DistributionFitResult
   :members: sample, pdf, cdf
   :show-inheritance:
   :no-index:

.. autoclass:: spark_dist_fit.results.FitResults
   :members:
   :show-inheritance:

Distributions
-------------

.. automodule:: spark_dist_fit.distributions
   :members:

Histogram
---------

.. automodule:: spark_dist_fit.histogram
   :members:

Plotting
--------

.. automodule:: spark_dist_fit.plotting
   :members:
