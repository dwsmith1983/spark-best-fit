"""Utility classes and functions for spark-dist-fit."""

from abc import ABC
from typing import Optional

from pyspark.sql import SparkSession

from spark_dist_fit.config import SparkConfig


class SparkSessionWrapper(ABC):
    """Base class that provides Spark session management.

    Classes that inherit from this will have access to self.spark
    without needing to pass it as a parameter everywhere. If no session
    is provided, automatically gets or creates one with the provided config.

    Example:
        >>> class MySparkClass(SparkSessionWrapper):
        ...     def __init__(self, spark: SparkSession = None, spark_config: SparkConfig = None):
        ...         super().__init__(spark, spark_config)
        ...
        ...     def my_method(self):
        ...         # Use self.spark directly
        ...         df = self.spark.createDataFrame(...)
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        spark_config: Optional[SparkConfig] = None,
    ):
        """Initialize with Spark session.

        Args:
            spark: Spark session. If None, gets or creates one.
            spark_config: Spark configuration. Applied if creating a new session
                or to an existing session's runtime config.
        """
        self._spark_config = spark_config or SparkConfig()

        if spark is not None:
            self._spark = spark
        else:
            # Build a new session with config
            builder = SparkSession.builder.appName(self._spark_config.app_name)
            for key, value in self._spark_config.to_spark_config().items():
                builder = builder.config(key, value)
            self._spark = builder.getOrCreate()

        # Apply config to existing session (runtime settings)
        self._apply_spark_config()

    def _apply_spark_config(self) -> None:
        """Apply Spark configuration to the session."""
        for key, value in self._spark_config.to_spark_config().items():
            self._spark.conf.set(key, value)

    @property
    def spark(self) -> SparkSession:
        """Get the Spark session.

        Returns:
            SparkSession instance
        """
        return self._spark
