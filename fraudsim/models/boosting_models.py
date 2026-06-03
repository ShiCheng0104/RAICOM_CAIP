from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from fraudsim.models.base import ModelArtifact
from fraudsim.models.sklearn_models import DataFrameOrdinalEncoder, _version


class XGBoostAdapter:
    name = "xgboost"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed. Run: pip install xgboost") from exc

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = (neg / pos) if pos else 1.0

        model = Pipeline([
            ("encode", DataFrameOrdinalEncoder()),
            ("clf", XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                n_estimators=900,
                learning_rate=0.045,
                max_depth=7,
                min_child_weight=2.0,
                subsample=0.82,
                colsample_bytree=0.82,
                reg_lambda=1.2,
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                n_jobs=-1,
                random_state=20260511,
            )),
        ])
        model.fit(x_train, y_train)
        return ModelArtifact(model_name=self.name, model_version=_version(), model=model)

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        return pd.Series(artifact.model.predict_proba(x)[:, 1], index=x.index)


class CatBoostAdapter:
    name = "catboost"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("catboost is not installed. Run: pip install catboost") from exc

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        class_weights = [1.0, (neg / pos) if pos else 1.0]

        model = Pipeline([
            ("encode", DataFrameOrdinalEncoder()),
            ("clf", CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="PRAUC",
                iterations=900,
                learning_rate=0.045,
                depth=7,
                l2_leaf_reg=4.0,
                random_seed=20260511,
                class_weights=class_weights,
                verbose=100,
                allow_writing_files=False,
                thread_count=-1,
            )),
        ])
        model.fit(x_train, y_train)
        return ModelArtifact(model_name=self.name, model_version=_version(), model=model)

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        return pd.Series(artifact.model.predict_proba(x)[:, 1], index=x.index)
