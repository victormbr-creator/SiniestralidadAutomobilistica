"""Leakage-aware temporal training for three classification baselines."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import IDENTIFIERS, LEAKAGE_COLUMNS, MODELS, SEED, TARGET, TEST_YEARS, TRAIN_YEARS
from .utils import get_logger, write_json

LOG = get_logger(__name__)


def temporal_split(frame: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select fixed train/test years and reject missing periods or labels."""
    if "year" not in frame or target not in frame:
        raise ValueError("Dataset needs year and target columns")
    train = frame.loc[frame["year"].isin(TRAIN_YEARS)].copy()
    test = frame.loc[frame["year"].isin(TEST_YEARS)].copy()
    if train.empty or test.empty:
        raise ValueError("Both 2018–2022 training and 2023–2024 test rows are required")
    if train[target].isna().any() or test[target].isna().any():
        raise ValueError("Target contains missing values")
    if train[target].nunique() < 2:
        raise ValueError("Training target needs both classes")
    return train, test


def select_predictors(frame: pd.DataFrame, target: str = TARGET) -> list[str]:
    """Exclude outcome-derived fields, identifiers, date/time source fields and year."""
    excluded = set(IDENTIFIERS) | set(LEAKAGE_COLUMNS) | {target, "event_date", "event_time"}
    columns = [name for name in frame if name not in excluded and (pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]) or pd.api.types.is_string_dtype(frame[name]) or isinstance(frame[name].dtype, pd.CategoricalDtype))]
    if not columns:
        raise ValueError("No eligible predictors; confirm source mapping and feature definitions")
    return columns


def make_preprocessor(frame: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    """Fit imputers and one-hot categories on training data only via a pipeline."""
    numeric = [name for name in columns if pd.api.types.is_numeric_dtype(frame[name]) and not pd.api.types.is_bool_dtype(frame[name])]
    categorical = [name for name in columns if name not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers)


def model_specs(class_weight: str | None = "balanced") -> dict[str, object]:
    """Return reproducible estimators; XGBoost balancing uses sample weights."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError("XGBoost requires its native runtime; on macOS install libomp before training") from exc
    return {
        "logit": LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=class_weight, random_state=SEED, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }


def train_models(frame: pd.DataFrame, target: str = TARGET, output_dir: Path = MODELS, class_weight: str | None = "balanced") -> dict[str, Pipeline]:
    """Fit models on past years and persist models plus holdout data."""
    train, test = temporal_split(frame, target)
    columns = select_predictors(train, target)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "training_manifest.json", {"target": target, "predictors": columns, "train_years": TRAIN_YEARS, "test_years": TEST_YEARS, "seed": SEED, "class_weight": class_weight})
    test[[*columns, target, "year"]].to_parquet(output_dir / "test_holdout.parquet", index=False)
    y_train = train[target].astype(int)
    sample_weight = None
    if class_weight == "balanced":
        counts = y_train.value_counts()
        sample_weight = y_train.map({label: len(y_train) / (2 * count) for label, count in counts.items()}).to_numpy()
    fitted = {}
    for name, estimator in model_specs(class_weight).items():
        pipeline = Pipeline([("preprocess", make_preprocessor(train, columns)), ("model", estimator)])
        options = {"model__sample_weight": sample_weight} if name == "xgboost" and sample_weight is not None else {}
        pipeline.fit(train[columns], y_train, **options)
        joblib.dump(pipeline, output_dir / f"{name}.joblib")
        fitted[name] = pipeline
        LOG.info("Trained %s", name)
    return fitted
