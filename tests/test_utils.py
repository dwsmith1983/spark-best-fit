"""Tests for utils module."""

from pyspark.sql import SparkSession

from spark_dist_fit.utils import SparkSessionWrapper


class ConcreteSparkWrapper(SparkSessionWrapper):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, spark: SparkSession):
        super().__init__(spark)


class TestSparkSessionWrapper:
    """Tests for SparkSessionWrapper base class."""

    def test_initialization(self, spark_session):
        """Test that wrapper initializes with spark session."""
        wrapper = ConcreteSparkWrapper(spark_session)

        assert wrapper._spark == spark_session

    def test_spark_property(self, spark_session):
        """Test spark property returns the session."""
        wrapper = ConcreteSparkWrapper(spark_session)

        assert wrapper.spark == spark_session
        assert wrapper.spark is spark_session

    def test_spark_property_returns_same_instance(self, spark_session):
        """Test that spark property always returns same instance."""
        wrapper = ConcreteSparkWrapper(spark_session)

        spark1 = wrapper.spark
        spark2 = wrapper.spark

        assert spark1 is spark2

    def test_multiple_wrappers_same_session(self, spark_session):
        """Test multiple wrappers can share same spark session."""
        wrapper1 = ConcreteSparkWrapper(spark_session)
        wrapper2 = ConcreteSparkWrapper(spark_session)

        assert wrapper1.spark is wrapper2.spark

    def test_wrapper_can_use_spark(self, spark_session):
        """Test that wrapper can actually use the spark session."""
        wrapper = ConcreteSparkWrapper(spark_session)

        # Should be able to create a DataFrame
        df = wrapper.spark.createDataFrame([(1,), (2,), (3,)], ["value"])

        assert df.count() == 3

    def test_inheritance_pattern(self, spark_session):
        """Test that inheritance works as expected."""

        class MyClass(SparkSessionWrapper):
            def __init__(self, spark: SparkSession):
                super().__init__(spark)

            def get_data(self):
                return self.spark.createDataFrame([(42,)], ["answer"])

        my_instance = MyClass(spark_session)
        df = my_instance.get_data()

        assert df.collect()[0]["answer"] == 42
