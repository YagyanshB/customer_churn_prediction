"""Model training for the churn prediction pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib  # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingImports]
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # pyright: ignore[reportMissingImports]
from sklearn.impute import SimpleImputer # pyright: ignore[reportMissingImports]
from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingImports]
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score # pyright: ignore[reportMissingImports]
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import StandardScaler # pyright: ignore[reportMissingImports]

from src.utils.config import ModelConfig, PipelineConfig # pyright: ignore[reportMissingImports]
from src.utils.logging import LoggerMixin # pyright: ignore[reportMissingImports]


class ChurnModelTrainer(LoggerMixin):
    """Trains and evaluates churn prediction models."""

    ALGORITHM_MAP = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }

    def __init__(
        self,
        config: Optional[ModelConfig | PipelineConfig] = None,
        algorithm: Optional[str] = None,
    ):
        """Initialize the trainer.

        Args:
            config: ModelConfig or PipelineConfig instance.
            algorithm: Algorithm name (overrides config if provided).
        """
        if config is None:
            self.config = ModelConfig()
        elif isinstance(config, PipelineConfig):
            self.config = config.model
        else:
            self.config = config

        self.algorithm = algorithm or self.config.algorithm
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}

    def _get_model_instance(self) -> Any:
        """Get a new model instance based on configuration."""
        if self.algorithm not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Choose from: {list(self.ALGORITHM_MAP.keys())}"
            )

        model_class = self.ALGORITHM_MAP[self.algorithm]
        params = self.config.get_algorithm_params()
        params["random_state"] = self.config.random_state

        return model_class(**params)

    def prepare_features(
        self, X: pd.DataFrame, fit: bool = True
    ) -> np.ndarray:
        """Prepare features for modeling.

        Args:
            X: Feature DataFrame.
            fit: Whether to fit the preprocessors (True for training).

        Returns:
            Scaled and imputed feature array.
        """
        X_copy = X.copy()

        # Handle missing values
        if fit:
            self.imputer = SimpleImputer(strategy="median")
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X_copy),
                columns=X_copy.columns,
            )
        else:
            if self.imputer is None:
                raise ValueError("Imputer not fitted. Call fit() first.")
            X_imputed = pd.DataFrame(
                self.imputer.transform(X_copy),
                columns=X_copy.columns,
            )

        # Handle infinite values
        X_imputed = X_imputed.replace([np.inf, -np.inf], 0)

        # Scale features
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit() first.")
            X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = "churned",
    ) -> "ChurnModelTrainer":
        """Train the churn prediction model.

        Args:
            df: DataFrame with features and target.
            feature_columns: List of feature column names. Auto-detected if None.
            target_column: Name of the target column.

        Returns:
            Self for method chaining.
        """
        self.logger.info(f"Training {self.algorithm} model...")

        # Determine feature columns
        if feature_columns is None:
            from engineer import FeatureEngineer # pyright: ignore[reportMissingImports]

            engineer = FeatureEngineer()
            feature_columns = engineer.get_feature_columns(df)

        self.feature_columns = feature_columns
        self.logger.info(f"Using {len(self.feature_columns)} features")

        # Prepare data
        X = df[self.feature_columns].copy()
        y = df[target_column].copy()

        # Prepare features
        X_scaled = self.prepare_features(X, fit=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")

        # Train model
        self.model = self._get_model_instance()
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        self.logger.info(
            f"Metrics - Accuracy: {self.metrics['accuracy']:.3f}, "
            f"F1: {self.metrics['f1_score']:.3f}, "
            f"ROC-AUC: {self.metrics['roc_auc']:.3f}"
        )

        return self

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = "churned",
    ) -> Dict[str, float]:
        """Perform cross-validation.

        Args:
            df: DataFrame with features and target.
            feature_columns: List of feature column names.
            target_column: Name of the target column.

        Returns:
            Dictionary with CV scores.
        """
        self.logger.info(f"Running {self.config.cv_folds}-fold cross-validation...")

        if feature_columns is None:
            from engineer import FeatureEngineer

            engineer = FeatureEngineer()
            feature_columns = engineer.get_feature_columns(df)

        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Prepare features
        X_scaled = self.prepare_features(X, fit=True)

        # Cross-validation
        model = self._get_model_instance()
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

        cv_results = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

        self.logger.info(
            f"CV ROC-AUC: {cv_results['cv_mean']:.3f} (+/- {cv_results['cv_std']:.3f})"
        )

        return cv_results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model.

        Returns:
            DataFrame with feature names and importance scores.

        Raises:
            ValueError: If model hasn't been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError(
                f"Model type {type(self.model)} doesn't support feature importance"
            )

        importance_df = pd.DataFrame(
            {"feature": self.feature_columns, "importance": importance}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save(self, path: str | Path) -> None:
        """Save the trained model and preprocessors.

        Args:
            path: Path to save the model.

        Raises:
            ValueError: If model hasn't been trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "feature_columns": self.feature_columns,
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "config": self.config.model_dump(),
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ChurnModelTrainer":
        """Load a trained model.

        Args:
            path: Path to the saved model.

        Returns:
            ChurnModelTrainer instance with loaded model.
        """
        path = Path(path)
        model_data = joblib.load(path)

        trainer = cls(algorithm=model_data["algorithm"])
        trainer.model = model_data["model"]
        trainer.scaler = model_data["scaler"]
        trainer.imputer = model_data["imputer"]
        trainer.feature_columns = model_data["feature_columns"]
        trainer.metrics = model_data["metrics"]

        return trainer


def train_multiple_models(
    df: pd.DataFrame,
    algorithms: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    target_column: str = "churned",
) -> Dict[str, ChurnModelTrainer]:
    """Train multiple models and return the best one.

    Args:
        df: DataFrame with features and target.
        algorithms: List of algorithms to try.
        feature_columns: Feature column names.
        target_column: Target column name.

    Returns:
        Dictionary mapping algorithm names to trained trainers.
    """
    if algorithms is None:
        algorithms = ["logistic_regression", "random_forest", "gradient_boosting"]

    trainers = {}
    for algo in algorithms:
        trainer = ChurnModelTrainer(algorithm=algo)
        trainer.fit(df, feature_columns, target_column)
        trainers[algo] = trainer

    return trainers


def get_best_model(trainers: Dict[str, ChurnModelTrainer]) -> ChurnModelTrainer:
    """Get the best model based on ROC-AUC score.

    Args:
        trainers: Dictionary of trained models.

    Returns:
        Best performing ChurnModelTrainer.
    """
    return max(trainers.values(), key=lambda t: t.metrics.get("roc_auc", 0))