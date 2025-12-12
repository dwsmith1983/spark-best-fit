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
    """Tests for SparkConfig application to Spark session."""

    def test_apply_spark_config_arrow_enabled(self, spark_session):
        """Test that arrow config is applied to session."""
        config = SparkConfig(arrow_enabled=True)
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        # Verify config was applied
        arrow_setting = wrapper.spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")
        assert arrow_setting == "true"

    def test_apply_spark_config_arrow_disabled(self, spark_session):
        """Test that arrow can be disabled."""
        config = SparkConfig(arrow_enabled=False)
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        arrow_setting = wrapper.spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")
        assert arrow_setting == "false"

    def test_apply_spark_config_adaptive_enabled(self, spark_session):
        """Test that adaptive query execution config is applied."""
        config = SparkConfig(adaptive_enabled=True)
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        adaptive_setting = wrapper.spark.conf.get("spark.sql.adaptive.enabled")
        assert adaptive_setting == "true"

    def test_apply_spark_config_adaptive_disabled(self, spark_session):
        """Test that adaptive query execution can be disabled."""
        config = SparkConfig(adaptive_enabled=False)
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        adaptive_setting = wrapper.spark.conf.get("spark.sql.adaptive.enabled")
        assert adaptive_setting == "false"

    def test_apply_spark_config_coalesce_enabled(self, spark_session):
        """Test that coalesce partitions config is applied."""
        config = SparkConfig(adaptive_coalesce_enabled=True)
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        coalesce_setting = wrapper.spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled")
        assert coalesce_setting == "true"

    def test_apply_spark_config_coalesce_disabled(self, spark_session):
        """Test that coalesce partitions can be disabled."""
        config = SparkConfig(adaptive_coalesce_enabled=False)
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        coalesce_setting = wrapper.spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled")
        assert coalesce_setting == "false"

    def test_apply_spark_config_all_settings(self, spark_session):
        """Test applying all config settings at once."""
        config = SparkConfig(
            arrow_enabled=False,
            adaptive_enabled=False,
            adaptive_coalesce_enabled=False,
        )
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        assert wrapper.spark.conf.get("spark.sql.execution.arrow.pyspark.enabled") == "false"
        assert wrapper.spark.conf.get("spark.sql.adaptive.enabled") == "false"
        assert wrapper.spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "false"

    def test_default_config_applied(self, spark_session):
        """Test that default SparkConfig is applied when none provided."""
        wrapper = ConcreteSparkWrapper(spark_session)

        # Default config has all settings enabled
        assert wrapper.spark.conf.get("spark.sql.execution.arrow.pyspark.enabled") == "true"
        assert wrapper.spark.conf.get("spark.sql.adaptive.enabled") == "true"
        assert wrapper.spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "true"

    def test_spark_config_stored(self, spark_session):
        """Test that spark_config is stored on the wrapper."""
        config = SparkConfig(app_name="test-app")
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        assert wrapper._spark_config == config
        assert wrapper._spark_config.app_name == "test-app"

    def test_config_applied_to_existing_session(self, spark_session):
        """Test that config is applied to an existing session."""
        # First set a different value
        spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

        # Now create wrapper with config that should override
        config = SparkConfig(arrow_enabled=True)
        wrapper = ConcreteSparkWrapper(spark_session, spark_config=config)

        # Should be overridden to true
        assert wrapper.spark.conf.get("spark.sql.execution.arrow.pyspark.enabled") == "true"


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
