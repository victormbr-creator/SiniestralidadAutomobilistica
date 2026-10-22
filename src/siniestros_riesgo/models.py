"""Comparable count and binary model definitions."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class TrailingMeanBaseline:
    """Twelve-month trailing mean baseline already calculated in the panel."""
    column: str = "mean_lag_12"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TrailingMeanBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].fillna(0).clip(lower=0).to_numpy(dtype=float)


def preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    """Build train-only preprocessing for comparable feature sets."""
    numeric = [name for name in features if pd.api.types.is_numeric_dtype(frame[name])]
    categorical = [name for name in features if name not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def count_models(seed: int) -> dict[str, object]:
    """Return count estimators; import XGBoost lazily for portable validation."""
    from xgboost import XGBRegressor
    return {
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed, criterion="poisson"),
        "xgboost_poisson": XGBRegressor(objective="count:poisson", n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, n_jobs=-1, random_state=seed),
    }


def binary_models(seed: int) -> dict[str, object]:
    """Return optional binary estimators."""
    from xgboost import XGBClassifier
    return {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest_binary": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=seed),
        "xgboost_binary": XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, n_jobs=-1, random_state=seed),
    }


def negative_binomial_formula(features: list[str]) -> str:
    """Create a statsmodels formula with month and region effects."""
    fixed = [name for name in features if name not in {"month", "region"}]
    rhs = " + ".join(fixed + ["C(month)", "C(region)"])
    return f"victim_crashes ~ {rhs}"
