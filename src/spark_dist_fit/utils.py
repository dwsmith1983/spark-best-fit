"""Utility classes and functions for spark-dist-fit."""

from abc import ABC
from typing import Optional

from pyspark.sql import SparkSession


class SparkSessionWrapper(ABC):
    """Base class that provides Spark session management.

    Classes that inherit from this will have access to self.spark
    without needing to pass it as a parameter everywhere. If no session
    is provided, automatically gets or creates one.

    Example:
        >>> class MySparkClass(SparkSessionWrapper):
        ...     def __init__(self, spark: SparkSession = None):
        ...         super().__init__(spark)
        ...
        ...     def my_method(self):
        ...         # Use self.spark directly
        ...         df = self.spark.createDataFrame(...)
    """

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with Spark session.

        Args:
            spark: Spark session. If None, gets or creates one.
        """
        self._spark = spark if spark is not None else SparkSession.builder.getOrCreate()

    @property
    def spark(self) -> SparkSession:
        """Get the Spark session.

        Returns:
            SparkSession instance
        """
        return self._spark
