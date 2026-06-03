from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudsim.models.base import ModelArtifact


class DataFrameOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.numeric_fill_: dict[str, float] = {}
        self.category_maps_: dict[str, dict[Any, int]] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "DataFrameOrdinalEncoder":
        self.columns_ = x.columns.tolist()
        self.categorical_columns_ = x.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_columns_ = [col for col in self.columns_ if col not in self.categorical_columns_]

        for col in self.numeric_columns_:
            values = pd.to_numeric(x[col], errors="coerce")
            median = values.median()
            self.numeric_fill_[col] = 0.0 if pd.isna(median) else float(median)

        for col in self.categorical_columns_:
            values = x[col].astype("object").where(pd.notna(x[col]), "__missing__")
            categories = pd.Series(values.unique()).sort_values(kind="mergesort").tolist()
            self.category_maps_[col] = {value: idx for idx, value in enumerate(categories)}
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        frame = x.copy()
        for col in self.columns_:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[self.columns_]

        arrays: list[np.ndarray] = []
        if self.numeric_columns_:
            numeric = pd.DataFrame(index=frame.index)
            for col in self.numeric_columns_:
                numeric[col] = pd.to_numeric(frame[col], errors="coerce").fillna(self.numeric_fill_[col])
            arrays.append(numeric.to_numpy(dtype=np.float32))

        if self.categorical_columns_:
            categorical = pd.DataFrame(index=frame.index)
            for col in self.categorical_columns_:
                mapping = self.category_maps_[col]
                values = frame[col].astype("object").where(pd.notna(frame[col]), "__missing__")
                categorical[col] = values.map(mapping).fillna(-1)
            arrays.append(categorical.to_numpy(dtype=np.float32))

        if not arrays:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.hstack(arrays)


class SklearnHGBAdapter:
    name = "sklearn_hgb"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        model = Pipeline([
            ("encode", DataFrameOrdinalEncoder()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=350,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=25,
                class_weight="balanced",
                random_state=20260511,
            )),
        ])
        model.fit(x_train, y_train)
        return ModelArtifact(model_name=self.name, model_version=_version(), model=model)

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        return pd.Series(artifact.model.predict_proba(x)[:, 1], index=x.index)


class SklearnExtraTreesAdapter:
    name = "sklearn_extra_trees"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        model = Pipeline([
            ("encode", DataFrameOrdinalEncoder()),
            ("clf", ExtraTreesClassifier(
                n_estimators=220,
                max_depth=28,
                min_samples_leaf=4,
                max_features="sqrt",
                bootstrap=True,
                max_samples=0.45,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=20260511,
            )),
        ])
        model.fit(x_train, y_train)
        return ModelArtifact(model_name=self.name, model_version=_version(), model=model)

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        return pd.Series(artifact.model.predict_proba(x)[:, 1], index=x.index)


class SklearnLogisticAdapter:
    name = "sklearn_logistic"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        model = Pipeline([
            ("encode", DataFrameOrdinalEncoder()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.8,
                class_weight="balanced",
                max_iter=300,
                n_jobs=-1,
                random_state=20260511,
            )),
        ])
        model.fit(x_train, y_train)
        return ModelArtifact(model_name=self.name, model_version=_version(), model=model)

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        return pd.Series(artifact.model.predict_proba(x)[:, 1], index=x.index)


class IsolationForestRiskModel:
    def __init__(self) -> None:
        self.encoder = DataFrameOrdinalEncoder()
        self.model = IsolationForest(
            n_estimators=240,
            max_samples=100000,
            contamination="auto",
            n_jobs=-1,
            random_state=20260511,
        )
        self.low_: float = 0.0
        self.high_: float = 1.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "IsolationForestRiskModel":
        normal = x[y == 0]
        if normal.empty:
            normal = x
        encoded_normal = self.encoder.fit_transform(normal)
        self.model.fit(encoded_normal)
        train_risk = -self.model.score_samples(self.encoder.transform(x))
        self.low_ = float(np.quantile(train_risk, 0.01))
        self.high_ = float(np.quantile(train_risk, 0.99))
        if self.high_ <= self.low_:
            self.high_ = self.low_ + 1.0
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        risk = -self.model.score_samples(self.encoder.transform(x))
        scaled = (risk - self.low_) / (self.high_ - self.low_)
        return np.clip(scaled, 0.0, 1.0)


class SklearnIsolationForestAdapter:
    name = "sklearn_isolation_forest"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        model = IsolationForestRiskModel().fit(x_train, y_train)
        return ModelArtifact(model_name=self.name, model_version=_version(), model=model)

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        return pd.Series(artifact.model.predict_proba(x), index=x.index)


def _version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
