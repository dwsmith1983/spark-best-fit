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

    SparkConfig is only applied when creating a NEW session. If an existing
    session is provided or already active, it is used as-is.

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
            spark: Spark session. If None, gets active session or creates one.
            spark_config: Spark configuration. Only applied when creating a NEW
                session. If a session already exists (passed or active), config
                is stored but not applied (Spark configs are set at creation).
        """
        self._spark_config = spark_config or SparkConfig()

        if spark is not None:
            # Use provided session
            self._spark = spark
        else:
            # Check if a session already exists
            existing = SparkSession.getActiveSession()
            if existing is not None:
                # Reuse existing session (config already set at creation)
                self._spark = existing
            else:
                # Create new session with full config
                builder = SparkSession.builder.appName(self._spark_config.app_name)
                for key, value in self._spark_config.to_spark_config().items():
                    builder = builder.config(key, value)
                self._spark = builder.getOrCreate()

    @property
    def spark(self) -> SparkSession:
        """Get the Spark session.

        Returns:
            SparkSession instance
        """
        return self._spark
