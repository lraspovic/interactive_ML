"""
Fit a scikit-learn classifier from a labelled pixel feature matrix and
persist the artifact to disk.

Model type and hyperparameters are driven by project.model_config so the
wizard choices are honoured without changing this file.
"""
from __future__ import annotations

import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ARTIFACTS_DIR = Path(os.getenv("MODEL_ARTIFACTS_DIR", "ml/artifacts"))


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    model_config: dict[str, Any] | None = None,
    version: str | None = None,
) -> tuple[Any, dict[str, Any], Path]:
    """
    Fit a classifier on (X_train, y_train) and evaluate on (X_test, y_test).

    Parameters
    ----------
    X_train:      (n_train, n_features) float32 feature matrix
    y_train:      (n_train,) integer class-id labels
    X_test:       (n_test,  n_features) held-out feature matrix; may be None
    y_test:       (n_test,)  held-out labels; may be None
    model_config: project.model_config dict (model_type + hyperparameters)
    version:      short identifier for the artifact filename; auto-generated if None

    Returns
    -------
    (fitted_model, metrics_dict, artifact_path)
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    cfg = model_config or {}
    model_type = cfg.get("model_type", "random_forest")
    hp = cfg.get("hyperparameters", {})

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 100)),
            max_depth=hp.get("max_depth") or None,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("xgboost not installed, falling back to random_forest")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = XGBClassifier(
                n_estimators=int(hp.get("n_estimators", 100)),
                learning_rate=float(hp.get("learning_rate", 0.1)),
                max_depth=int(hp.get("max_depth", 6)),
                eval_metric="mlogloss",
                random_state=42,
            )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=int(hp.get("n_estimators", 100)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            max_depth=int(hp.get("max_depth", 3)),
            random_state=42,
        )
    elif model_type == "svm":
        model = SVC(
            kernel=hp.get("kernel", "rbf"),
            C=float(hp.get("C", 1.0)),
            probability=True,
        )
    else:
        logger.warning("Unknown model_type '%s', using random_forest", model_type)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    has_test = X_test is not None and len(X_test) > 0
    test_acc = float(accuracy_score(y_test, model.predict(X_test))) if has_test else None

    metrics: dict[str, Any] = {
        # primary headline metric: test accuracy when available, else train
        "accuracy": round(test_acc if has_test else train_acc, 4),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4) if has_test else None,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)) if has_test else 0,
        # keep legacy key so nothing else breaks
        "n_samples": int(len(y_train)) + (int(len(y_test)) if has_test else 0),
        "n_features": int(X_train.shape[1]),
        "classes": sorted(int(c) for c in np.unique(y_train)),
        "model_type": model_type,
    }

    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ver = version or uuid.uuid4().hex[:8]
    artifact_path = MODEL_ARTIFACTS_DIR / f"model_{ver}.pkl"
    with open(artifact_path, "wb") as fh:
        pickle.dump({"model": model, "metrics": metrics}, fh)

    logger.info(
        "Saved model to %s — type=%s train_acc=%.3f test_acc=%s n_train=%d n_test=%d",
        artifact_path, model_type,
        train_acc,
        f"{test_acc:.3f}" if has_test else "n/a",
        len(y_train),
        len(y_test) if has_test else 0,
    )
    return model, metrics, artifact_path
