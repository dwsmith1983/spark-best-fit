"""Tests for utils module."""

from pyspark.sql import SparkSession

from spark_dist_fit.config import SparkConfig
from spark_dist_fit.utils import SparkSessionWrapper


class ConcreteSparkWrapper(SparkSessionWrapper):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, spark: SparkSession = None, spark_config: SparkConfig = None):
        super().__init__(spark, spark_config)


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


class TestSparkConfigApplication:
    """Tests for SparkConfig application to Spark session.

    SparkConfig is only applied when creating a NEW session.
    When an existing session is provided or active, it is used as-is
    (Spark configs are immutable after session creation).
    """

    def test_spark_config_stored(self, spark_session):
        """Test that spark_config is stored on the wrapper."""
        config = SparkConfig(app_name="test-app")
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        assert wrapper._spark_config == config
        assert wrapper._spark_config.app_name == "test-app"

    def test_config_stored_but_not_applied(self, spark_session):
        """Test that config is stored but NOT applied to an existing session.

        SparkConfig settings can only be set at session creation time.
        When using an existing session, the config is stored for reference
        but not applied to the session.
        """
        config = SparkConfig(
            app_name="custom-app",
            arrow_enabled=False,
            adaptive_enabled=False,
        )
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        # Config should be stored
        assert wrapper._spark_config == config
        assert wrapper._spark_config.app_name == "custom-app"
        assert wrapper._spark_config.arrow_enabled is False

        # Session should be the same instance (not modified)
        assert wrapper.spark is spark_session

    def test_wrapper_uses_provided_session_directly(self, spark_session):
        """Test that wrapper uses provided session without creating new one."""
        config = SparkConfig(app_name="different-app")
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        # Should use exact same session instance
        assert wrapper.spark is spark_session
        assert wrapper._spark is spark_session

    def test_to_spark_config_returns_correct_values(self):
        """Test that to_spark_config returns correct dictionary values."""
        config = SparkConfig(
            arrow_enabled=False,
            adaptive_enabled=False,
            adaptive_coalesce_enabled=False,
        )
        spark_dict = config.to_spark_config()

        assert spark_dict["spark.sql.execution.arrow.pyspark.enabled"] == "false"
        assert spark_dict["spark.sql.adaptive.enabled"] == "false"
        assert spark_dict["spark.sql.adaptive.coalescePartitions.enabled"] == "false"

    def test_default_config_values(self):
        """Test that default SparkConfig has expected values."""
        config = SparkConfig()
        spark_dict = config.to_spark_config()

        # Default config has all settings enabled
        assert spark_dict["spark.sql.execution.arrow.pyspark.enabled"] == "true"
        assert spark_dict["spark.sql.adaptive.enabled"] == "true"
        assert spark_dict["spark.sql.adaptive.coalescePartitions.enabled"] == "true"


class TestSparkSessionCreation:
    """Tests for automatic Spark session creation."""

    def test_creates_session_when_none_provided(self):
        """Test that session is created when none is provided."""
        config = SparkConfig(app_name="auto-created-app")
        wrapper = ConcreteSparkWrapper(spark=None, spark_config=config)

        assert wrapper.spark is not None
        assert isinstance(wrapper.spark, SparkSession)

    def test_uses_provided_session(self, spark_session):
        """Test that provided session is used instead of creating new."""
        config = SparkConfig(app_name="different-name")
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        # Should use provided session, not create new
        assert wrapper.spark is spark_session


class TestSparkSessionWrapperEdgeCases:
    """Edge case tests for SparkSessionWrapper."""

    def test_wrapper_with_none_spark_and_none_config(self):
        """Test wrapper with both spark and config as None."""
        wrapper = ConcreteSparkWrapper(spark=None, spark_config=None)

        # Should create session with default config
        assert wrapper.spark is not None
        assert wrapper._spark_config.app_name == "spark-dist-fit"

    def test_to_spark_config_returns_dict(self, spark_session):
        """Test that to_spark_config returns a dictionary."""
        config = SparkConfig()
        spark_dict = config.to_spark_config()

        assert isinstance(spark_dict, dict)
        assert len(spark_dict) >= 3  # At least arrow, adaptive, coalesce

    def test_wrapper_preserves_session_after_config_apply(self, spark_session):
        """Test that session is still usable after config application."""
        config = SparkConfig()
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        # Should still be able to use session
        df = wrapper.spark.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
        assert df.count() == 2
