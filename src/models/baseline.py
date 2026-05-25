import logging

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import SEED

logger = logging.getLogger(__name__)

BASELINES = {
    "dummy": DummyClassifier(strategy="most_frequent", random_state=SEED),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
}

SCORING = ["roc_auc", "average_precision", "f1", "accuracy"]


def train_baselines(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for name, clf in BASELINES.items():
        logger.info("Training baseline: %s", name)
        scores = cross_validate(
            clf, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=False
        )
        results[name] = {
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
            "pr_auc": float(np.mean(scores["test_average_precision"])),
            "f1": float(np.mean(scores["test_f1"])),
            "accuracy": float(np.mean(scores["test_accuracy"])),
        }
        clf.fit(X_train, y_train)
        logger.info(
            "%s → AUC=%.4f | F1=%.4f",
            name,
            results[name]["roc_auc"],
            results[name]["f1"],
        )

    return results
