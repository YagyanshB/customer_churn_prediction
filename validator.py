"""Data validation utilities for the churn prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import pandas as pd # pyright: ignore[reportMissingImports]

from src.utils.logging import LoggerMixin  # pyright: ignore[reportMissingImports]


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __bool__(self) -> bool:
        return self.is_valid

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            raise ValueError(f"Validation failed: {'; '.join(self.errors)}")


class DataValidator(LoggerMixin):
    """Validates data quality and schema compliance."""

    # Expected columns for each dataset
    CALLS_REQUIRED_COLUMNS: Set[str] = {
        "unique_customer_identifier",
        "event_date",
        "call_type",
        "talk_time_seconds",
        "hold_time_seconds",
    }

    CEASE_REQUIRED_COLUMNS: Set[str] = {
        "unique_customer_identifier",
        "cease_placed_date",
        "reason_description_insight",
    }

    CUSTOMERS_REQUIRED_COLUMNS: Set[str] = {
        "unique_customer_identifier",
        "contract_status",
        "technology",
        "tenure_days",
    }

    def validate_calls(self, df: pd.DataFrame) -> ValidationResult:
        """Validate call center data.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationResult with status and any issues found.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required columns
        missing_cols = self.CALLS_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")

        # Check for duplicates
        if df.duplicated().sum() > 0:
            warnings.append(f"Found {df.duplicated().sum()} duplicate rows")

        # Check for negative values in numeric columns
        if "talk_time_seconds" in df.columns:
            neg_talk = (df["talk_time_seconds"] < 0).sum()
            if neg_talk > 0:
                warnings.append(f"Found {neg_talk} negative talk_time values")

        if "hold_time_seconds" in df.columns:
            neg_hold = (df["hold_time_seconds"] < 0).sum()
            if neg_hold > 0:
                warnings.append(f"Found {neg_hold} negative hold_time values")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_cease(self, df: pd.DataFrame) -> ValidationResult:
        """Validate cease/churn data.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationResult with status and any issues found.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required columns
        missing_cols = self.CEASE_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")

        # Check for missing dates
        if "cease_placed_date" in df.columns:
            null_dates = df["cease_placed_date"].isna().sum()
            if null_dates > 0:
                errors.append(f"Found {null_dates} null cease_placed_date values")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_customers(self, df: pd.DataFrame) -> ValidationResult:
        """Validate customer data.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationResult with status and any issues found.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required columns
        missing_cols = self.CUSTOMERS_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")

        # Check for duplicate customer IDs
        if "unique_customer_identifier" in df.columns:
            dup_count = df["unique_customer_identifier"].duplicated().sum()
            if dup_count > 0:
                warnings.append(f"Found {dup_count} duplicate customer IDs")

        # Check tenure values
        if "tenure_days" in df.columns:
            neg_tenure = (df["tenure_days"] < 0).sum()
            if neg_tenure > 0:
                warnings.append(f"Found {neg_tenure} negative tenure values")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_all(
        self,
        calls_df: pd.DataFrame,
        cease_df: pd.DataFrame,
        customers_df: pd.DataFrame,
    ) -> ValidationResult:
        """Validate all datasets.

        Args:
            calls_df: Call center data.
            cease_df: Cease/churn data.
            customers_df: Customer data.

        Returns:
            Combined ValidationResult.
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []

        # Validate each dataset
        for name, df, validator in [
            ("calls", calls_df, self.validate_calls),
            ("cease", cease_df, self.validate_cease),
            ("customers", customers_df, self.validate_customers),
        ]:
            result = validator(df)
            all_errors.extend([f"[{name}] {e}" for e in result.errors])
            all_warnings.extend([f"[{name}] {w}" for w in result.warnings])

        # Cross-dataset validation
        customer_ids = set(customers_df["unique_customer_identifier"])
        cease_ids = set(cease_df["unique_customer_identifier"])

        # Check if all cease customers exist in customer data
        orphan_cease = cease_ids - customer_ids
        if orphan_cease:
            all_warnings.append(
                f"Found {len(orphan_cease)} cease records without matching customers"
            )

        is_valid = len(all_errors) == 0

        if all_warnings:
            for warning in all_warnings:
                self.logger.warning(warning)

        return ValidationResult(
            is_valid=is_valid, errors=all_errors, warnings=all_warnings
        )