from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class ModelArtifact:
    model_name: str
    model_version: str
    model: object


class ModelAdapter(Protocol):
    name: str

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> ModelArtifact:
        ...

    def predict_proba(self, artifact: ModelArtifact, x: pd.DataFrame) -> pd.Series:
        ...
