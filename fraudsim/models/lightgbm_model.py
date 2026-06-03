from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from fraudsim.models.base import ModelArtifact


class LightGBMAdapter:
    name = "lightgbm"

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed. Run: pip install -r requirements-fraudsim.txt") from exc

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = (neg / pos) if pos else 1.0

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=20260511,
            n_jobs=-1,
        )
        callbacks = [lgb.early_stopping(100), lgb.log_evaluation(100)]
        categorical = x_train.select_dtypes(include=["category"]).columns.tolist()
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="average_precision",
            categorical_feature=categorical,
            callbacks=callbacks,
        )
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return ModelArtifact(model_name=self.name, model_version=version, model=model)

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        scores = artifact.model.predict_proba(x)[:, 1]
        return pd.Series(scores, index=x.index)
