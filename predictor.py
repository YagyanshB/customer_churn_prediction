"""Prediction and scoring for the churn prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingImports]

from trainer import ChurnModelTrainer # pyright: ignore[reportMissingImports]
from config import PipelineConfig, RiskConfig # pyright: ignore[reportMissingImports]
from logging import LoggerMixin # pyright: ignore[reportMissingImports]


class ChurnPredictor(LoggerMixin):
    """Generates churn predictions and priority scores."""

    def __init__(
        self,
        trainer: ChurnModelTrainer,
        risk_config: Optional[RiskConfig] = None,
    ):
        """Initialize the predictor.

        Args:
            trainer: Trained ChurnModelTrainer instance.
            risk_config: Risk categorization configuration.
        """
        self.trainer = trainer
        self.risk_config = risk_config or RiskConfig()

        if trainer.model is None:
            raise ValueError("Trainer must have a fitted model")

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        risk_config: Optional[RiskConfig] = None,
    ) -> "ChurnPredictor":
        """Load a predictor from a saved model file.

        Args:
            model_path: Path to the saved model.
            risk_config: Risk categorization configuration.

        Returns:
            ChurnPredictor instance.
        """
        trainer = ChurnModelTrainer.load(model_path)
        return cls(trainer, risk_config)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities.

        Args:
            df: DataFrame with feature columns.

        Returns:
            Array of churn probabilities.
        """
        # Get features
        X = df[self.trainer.feature_columns].copy()

        # Prepare features using the trainer's preprocessors
        X_scaled = self.trainer.prepare_features(X, fit=False)

        # Predict
        probabilities = self.trainer.model.predict_proba(X_scaled)[:, 1]

        return probabilities

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict churn (binary).

        Args:
            df: DataFrame with feature columns.
            threshold: Probability threshold for positive class.

        Returns:
            Array of binary predictions.
        """
        probabilities = self.predict_proba(df)
        return (probabilities >= threshold).astype(int)

    def categorize_risk(self, probabilities: np.ndarray) -> pd.Series:
        """Categorize customers into risk levels.

        Args:
            probabilities: Array of churn probabilities.

        Returns:
            Series with risk category labels.
        """
        thresholds = self.risk_config.thresholds

        conditions = [
            probabilities <= thresholds["low"],
            probabilities <= thresholds["medium"],
            probabilities <= thresholds["high"],
            probabilities > thresholds["high"],
        ]
        choices = ["Low Risk", "Medium Risk", "High Risk", "Critical"]

        return pd.Series(np.select(conditions, choices))

    def calculate_value_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate customer value score.

        Args:
            df: DataFrame with customer data.

        Returns:
            Array of value scores (0-1).
        """
        value_score = np.zeros(len(df))

        # Tenure component
        if "tenure_days" in df.columns:
            tenure_norm = df["tenure_days"] / df["tenure_days"].max()
            value_score += tenure_norm.fillna(0).values * 0.5

        # Speed component (proxy for service tier/revenue)
        if "speed" in df.columns:
            speed_norm = df["speed"] / df["speed"].max()
            value_score += speed_norm.fillna(0).values * 0.5

        return value_score

    def calculate_retention_priority(
        self,
        churn_probability: np.ndarray,
        value_score: np.ndarray,
    ) -> np.ndarray:
        """Calculate retention priority score.

        Args:
            churn_probability: Array of churn probabilities.
            value_score: Array of customer value scores.

        Returns:
            Array of priority scores.
        """
        weights = self.risk_config.priority_weights
        priority = (
            churn_probability * weights["churn_probability"]
            + value_score * weights["customer_value"]
        )
        return priority

    def predict_priority(
        self,
        df: pd.DataFrame,
        include_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate full priority report for customers.

        Args:
            df: DataFrame with customer data and features.
            include_features: Additional columns to include in output.

        Returns:
            DataFrame with predictions, risk categories, and priority scores.
        """
        self.logger.info(f"Generating predictions for {len(df)} customers...")

        # Calculate scores
        churn_probability = self.predict_proba(df)
        risk_category = self.categorize_risk(churn_probability)
        value_score = self.calculate_value_score(df)
        retention_priority = self.calculate_retention_priority(
            churn_probability, value_score
        )

        # Build output DataFrame
        output_columns = ["unique_customer_identifier"]

        # Add optional columns if they exist
        optional_cols = ["contract_status", "technology", "tenure_days"]
        for col in optional_cols:
            if col in df.columns:
                output_columns.append(col)

        # Add any user-specified columns
        if include_features:
            for col in include_features:
                if col in df.columns and col not in output_columns:
                    output_columns.append(col)

        result = df[output_columns].copy()
        result["churn_probability"] = churn_probability
        result["risk_category"] = risk_category.values
        result["value_score"] = value_score
        result["retention_priority"] = retention_priority

        # Sort by retention priority (descending)
        result = result.sort_values("retention_priority", ascending=False)

        self.logger.info(f"Priority report generated with {len(result)} customers")
        self.logger.info(f"Risk distribution:\n{result['risk_category'].value_counts()}")

        return result.reset_index(drop=True)

    def get_top_priority_customers(
        self,
        df: pd.DataFrame,
        n: int = 100,
        min_risk_level: str = "Medium Risk",
    ) -> pd.DataFrame:
        """Get top N priority customers for retention outreach.

        Args:
            df: DataFrame with customer data.
            n: Number of customers to return.
            min_risk_level: Minimum risk level to include.

        Returns:
            DataFrame with top priority customers.
        """
        priority_df = self.predict_priority(df)

        # Filter by minimum risk level
        risk_order = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2, "Critical": 3}
        min_risk_value = risk_order.get(min_risk_level, 1)

        priority_df["risk_order"] = priority_df["risk_category"].map(risk_order)
        filtered = priority_df[priority_df["risk_order"] >= min_risk_value]
        filtered = filtered.drop("risk_order", axis=1)

        return filtered.head(n)