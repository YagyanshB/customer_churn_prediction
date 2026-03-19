"""Feature engineering for the churn prediction pipeline."""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from src.utils.config import FeatureConfig, PipelineConfig # pyright: ignore[reportMissingImports]
from src.utils.logging import LoggerMixin # pyright: ignore[reportMissingImports]


class CallAggregator(LoggerMixin):
    """Aggregates call center data per customer."""

    def __init__(self, call_types: Optional[List[str]] = None):
        """Initialize the aggregator.

        Args:
            call_types: List of call types to track. Defaults to standard types.
        """
        self.call_types = call_types or ["Tech", "Loyalty", "Customer Finance"]

    def aggregate(self, calls_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate call data per customer.

        Args:
            calls_df: DataFrame with call records.

        Returns:
            DataFrame with one row per customer and aggregated call features.
        """
        self.logger.info("Aggregating call data...")

        # Basic aggregations
        agg_dict = {
            "event_date": "count",
            "talk_time_seconds": ["sum", "mean", "max"],
            "hold_time_seconds": ["sum", "mean", "max"],
            "call_type": lambda x: x.value_counts().to_dict(),
        }

        call_features = (
            calls_df.groupby("unique_customer_identifier").agg(agg_dict).reset_index()
        )

        # Flatten column names
        call_features.columns = [
            "unique_customer_identifier",
            "total_calls",
            "total_talk_time",
            "avg_talk_time",
            "max_talk_time",
            "total_hold_time",
            "avg_hold_time",
            "max_hold_time",
            "call_types_dict",
        ]

        # Extract call type counts
        for ct in self.call_types:
            col_name = f"calls_{ct.lower().replace(' ', '_')}"
            call_features[col_name] = call_features["call_types_dict"].apply(
                lambda x: x.get(ct, 0) if isinstance(x, dict) else 0
            )

        call_features = call_features.drop("call_types_dict", axis=1)

        self.logger.info(f"Created {len(call_features.columns) - 1} call features")
        return call_features


class FeatureEngineer(LoggerMixin):
    """Main feature engineering class that creates all features for modeling."""

    # Feature columns used for modeling
    FEATURE_COLUMNS: List[str] = [
        "tenure_days",
        "ooc_days",
        "speed",
        "line_speed",
        "total_calls",
        "total_talk_time",
        "avg_talk_time",
        "max_talk_time",
        "total_hold_time",
        "avg_hold_time",
        "max_hold_time",
        "calls_tech",
        "calls_loyalty",
        "calls_customer_finance",
        "is_out_of_contract",
        "is_early_contract",
        "technology_encoded",
        "hold_ratio",
    ]

    def __init__(self, config: Optional[FeatureConfig | PipelineConfig] = None):
        """Initialize the feature engineer.

        Args:
            config: FeatureConfig or PipelineConfig. Uses defaults if not provided.
        """
        if config is None:
            self.config = FeatureConfig()
        elif isinstance(config, PipelineConfig):
            self.config = config.features
        else:
            self.config = config

        self.call_aggregator = CallAggregator(call_types=self.config.call_types)
        self._churned_customers: Set[str] = set()

    def create_churn_labels(
        self, customers_df: pd.DataFrame, cease_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add churn labels based on cease records.

        Args:
            customers_df: Customer data.
            cease_df: Cease/churn data.

        Returns:
            Customer DataFrame with 'churned' column added.
        """
        self.logger.info("Creating churn labels...")

        self._churned_customers = set(cease_df["unique_customer_identifier"].unique())

        df = customers_df.copy()
        df["churned"] = df["unique_customer_identifier"].apply(
            lambda x: 1 if x in self._churned_customers else 0
        )

        churn_rate = df["churned"].mean() * 100
        self.logger.info(
            f"Churn rate: {churn_rate:.2f}% ({df['churned'].sum()} churned)"
        )

        return df

    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on contract status.

        Args:
            df: DataFrame with contract_status column.

        Returns:
            DataFrame with new contract features.
        """
        self.logger.info("Creating contract features...")

        df = df.copy()
        df["is_out_of_contract"] = df["contract_status"].apply(
            lambda x: 1 if "OOC" in str(x) else 0
        )
        df["is_early_contract"] = df["contract_status"].apply(
            lambda x: 1 if "Early" in str(x) else 0
        )

        return df

    def create_technology_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on technology type.

        Args:
            df: DataFrame with technology column.

        Returns:
            DataFrame with encoded technology feature.
        """
        self.logger.info("Creating technology features...")

        df = df.copy()
        df["technology_encoded"] = (
            df["technology"].map(self.config.technology_encoding).fillna(-1)
        )

        return df

    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure-based features.

        Args:
            df: DataFrame with tenure_days column.

        Returns:
            DataFrame with tenure category features.
        """
        self.logger.info("Creating tenure features...")

        df = df.copy()
        df["tenure_months"] = df["tenure_days"] / 30
        df["tenure_category"] = pd.cut(
            df["tenure_months"],
            bins=self.config.tenure_bins,
            labels=self.config.tenure_labels,
        )

        return df

    def create_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create speed tier features.

        Args:
            df: DataFrame with speed column.

        Returns:
            DataFrame with speed tier category.
        """
        self.logger.info("Creating speed features...")

        df = df.copy()
        if "speed" in df.columns:
            df["speed_tier"] = pd.cut(
                df["speed"],
                bins=self.config.speed_bins,
                labels=self.config.speed_labels,
            )

        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived/calculated features.

        Args:
            df: DataFrame with call metrics.

        Returns:
            DataFrame with derived features.
        """
        self.logger.info("Creating derived features...")

        df = df.copy()

        # Hold time ratio (frustration indicator)
        if "total_hold_time" in df.columns and "total_talk_time" in df.columns:
            df["hold_ratio"] = df["total_hold_time"] / (df["total_talk_time"] + 1)

        # Ensure ooc_days is filled
        if "ooc_days" in df.columns:
            df["ooc_days"] = df["ooc_days"].fillna(0)

        return df

    def merge_call_features(
        self, customers_df: pd.DataFrame, calls_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge aggregated call features with customer data.

        Args:
            customers_df: Customer data.
            calls_df: Call records data.

        Returns:
            Merged DataFrame with call features.
        """
        self.logger.info("Merging call features...")

        # Aggregate call data
        call_features = self.call_aggregator.aggregate(calls_df)

        # Merge with customer data
        df = customers_df.merge(
            call_features, on="unique_customer_identifier", how="left"
        )

        # Fill missing call features with 0
        call_cols = [
            "total_calls",
            "total_talk_time",
            "avg_talk_time",
            "max_talk_time",
            "total_hold_time",
            "avg_hold_time",
            "max_hold_time",
            "calls_tech",
            "calls_loyalty",
            "calls_customer_finance",
        ]

        for col in call_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def fit_transform(
        self,
        customers_df: pd.DataFrame,
        calls_df: pd.DataFrame,
        cease_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run the complete feature engineering pipeline.

        Args:
            customers_df: Customer data.
            calls_df: Call records data.
            cease_df: Cease/churn data.

        Returns:
            DataFrame with all engineered features.
        """
        self.logger.info("Starting feature engineering pipeline...")

        # Create churn labels
        df = self.create_churn_labels(customers_df, cease_df)

        # Merge call features
        df = self.merge_call_features(df, calls_df)

        # Create all feature groups
        df = self.create_contract_features(df)
        df = self.create_technology_features(df)
        df = self.create_tenure_features(df)
        df = self.create_speed_features(df)
        df = self.create_derived_features(df)

        self.logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns available in the DataFrame.

        Args:
            df: DataFrame to check.

        Returns:
            List of available feature column names.
        """
        return [col for col in self.FEATURE_COLUMNS if col in df.columns]