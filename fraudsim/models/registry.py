from __future__ import annotations

from fraudsim.models.boosting_models import CatBoostAdapter, XGBoostAdapter
from fraudsim.models.lightgbm_model import LightGBMAdapter
from fraudsim.models.sklearn_models import (
    SklearnExtraTreesAdapter,
    SklearnHGBAdapter,
    SklearnIsolationForestAdapter,
    SklearnLogisticAdapter,
)


_REGISTRY = {
    CatBoostAdapter.name: CatBoostAdapter,
    LightGBMAdapter.name: LightGBMAdapter,
    SklearnHGBAdapter.name: SklearnHGBAdapter,
    SklearnExtraTreesAdapter.name: SklearnExtraTreesAdapter,
    SklearnLogisticAdapter.name: SklearnLogisticAdapter,
    SklearnIsolationForestAdapter.name: SklearnIsolationForestAdapter,
    XGBoostAdapter.name: XGBoostAdapter,
}


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def get_model_adapter(name: str):
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc
