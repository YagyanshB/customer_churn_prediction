"""Data loading utilities for the churn prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd   # pyright: ignore[reportMissingImports]

from src.utils.config import DataConfig, PipelineConfig # pyright: ignore[reportMissingImports]
from src.utils.logging import LoggerMixin # pyright: ignore[reportMissingImports]


class DataLoader(LoggerMixin):
    """Handles loading and initial validation of data files.

    Attributes:
        config: DataConfig instance with file paths.
    """

    def __init__(self, config: DataConfig | PipelineConfig | str | Path):
        """Initialize the DataLoader.

        Args:
            config: Either a DataConfig, PipelineConfig, or path to config file.
        """
        if isinstance(config, (str, Path)):
            from src.utils.config import load_config # pyright: ignore[reportMissingImports]

            pipeline_config = load_config(config)
            self.config = pipeline_config.data
        elif isinstance(config, PipelineConfig):
            self.config = config.data
        else:
            self.config = config

    def load_calls(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load call center data.

        Args:
            path: Optional override path. Uses config path if not provided.

        Returns:
            DataFrame containing call records.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        file_path = path or self.config.calls_path
        self.logger.info(f"Loading call data from {file_path}")

        df = pd.read_csv(file_path)
        df["event_date"] = pd.to_datetime(df["event_date"])

        self.logger.info(f"Loaded {len(df)} call records")
        return df

    def load_cease(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load cease/churn data.

        Args:
            path: Optional override path. Uses config path if not provided.

        Returns:
            DataFrame containing cease records.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        file_path = path or self.config.cease_path
        self.logger.info(f"Loading cease data from {file_path}")

        df = pd.read_csv(file_path)
        df["cease_placed_date"] = pd.to_datetime(df["cease_placed_date"])
        df["cease_completed_date"] = pd.to_datetime(
            df["cease_completed_date"], errors="coerce"
        )

        self.logger.info(f"Loaded {len(df)} cease records")
        return df

    def load_customers(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load customer data.

        Args:
            path: Optional override path. Uses config path if not provided.

        Returns:
            DataFrame containing customer records.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        file_path = path or self.config.customers_path
        self.logger.info(f"Loading customer data from {file_path}")

        df = pd.read_csv(file_path)

        # Handle date column (might be 'date' or 'datevalue')
        date_col = "datevalue" if "datevalue" in df.columns else "date"
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

        self.logger.info(f"Loaded {len(df)} customer records")
        return df

    def load_all(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three datasets.

        Returns:
            Tuple of (calls_df, cease_df, customers_df).
        """
        self.logger.info("Loading all datasets...")

        calls_df = self.load_calls()
        cease_df = self.load_cease()
        customers_df = self.load_customers()

        return calls_df, cease_df, customers_df


class DataFrameLoader:
    """Alternative loader for when data is already in memory (e.g., from notebooks)."""

    def __init__(
        self,
        calls_df: Optional[pd.DataFrame] = None,
        cease_df: Optional[pd.DataFrame] = None,
        customers_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize with existing DataFrames.

        Args:
            calls_df: Call center data.
            cease_df: Cease/churn data.
            customers_df: Customer data.
        """
        self._calls = calls_df
        self._cease = cease_df
        self._customers = customers_df

    def load_calls(self) -> pd.DataFrame:
        """Return calls DataFrame."""
        if self._calls is None:
            raise ValueError("Calls data not provided")
        return self._calls.copy()

    def load_cease(self) -> pd.DataFrame:
        """Return cease DataFrame."""
        if self._cease is None:
            raise ValueError("Cease data not provided")
        return self._cease.copy()

    def load_customers(self) -> pd.DataFrame:
        """Return customers DataFrame."""
        if self._customers is None:
            raise ValueError("Customers data not provided")
        return self._customers.copy()

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return all DataFrames."""
        return self.load_calls(), self.load_cease(), self.load_customers()