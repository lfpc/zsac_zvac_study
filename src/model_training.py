
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from skopt import BayesSearchCV


@dataclass
class TrainingResult:
    """Container holding the trained estimator and evaluation metadata."""

    best_estimator: BaseEstimator
    best_params: Dict[str, Any]
    test_metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_scores: np.ndarray | None
    search: BayesSearchCV | None


def _score_predictions(
    estimator: BaseEstimator,
    X_test,
    y_test,
    *,
    y_pred=None,
    y_scores=None,
) -> Dict[str, float]:
    """Compute standard metrics for the fitted estimator on held-out data."""

    if y_pred is None:
        y_pred = estimator.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test, y_pred)}

    if y_scores is None:
        if hasattr(estimator, "predict_proba"):
            y_scores = estimator.predict_proba(X_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            y_scores = estimator.decision_function(X_test)

    if y_scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_scores)
        metrics["aupr"] = average_precision_score(y_test, y_scores)
    else:
        metrics["roc_auc"] = np.nan
        metrics["aupr"] = np.nan

    return metrics


def _to_numpy(matrix):
    if hasattr(matrix, "values"):
        return matrix.values
    return np.asarray(matrix)


def _is_rank_deficient(X) -> bool:
    try:
        mat = _to_numpy(X)
        if mat.ndim != 2:
            mat = np.atleast_2d(mat)
        rank = np.linalg.matrix_rank(mat)
        return rank < mat.shape[1]
    except Exception:
        return False


def train_model(
    estimator: BaseEstimator,
    X,
    y,
    *,
    search_spaces: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    scoring: str = "roc_auc",
    cv_splits: int = 5,
    test_size: float = 0.2,
    n_iter: int = 32,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
) -> TrainingResult:
    """Optimize hyperparameters for ``estimator`` and return the trained model.

    Parameters
    ----------
    estimator:
        Unfitted sklearn-compatible estimator.
    X, y:
        Feature matrix and target vector.
    search_spaces:
        Space definition understood by ``BayesSearchCV`` (dict or list of dicts).
    scoring:
        Metric to maximize during optimization.
    cv_splits:
        Number of folds for StratifiedKFold.
    test_size:
        Fraction of the data reserved for hold-out evaluation.
    n_iter:
        Number of Bayesian optimization iterations.
    random_state:
        Seed applied to the split, CV, and optimizer.
    n_jobs:
        Parallelism for the search; defaults to all cores.
    verbose:
        Verbosity forwarded to ``BayesSearchCV``.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    base_estimator = clone(estimator)
    best_estimator = base_estimator
    best_params: Dict[str, Any] = {}
    search: BayesSearchCV | None = None

    skip_bayes = False
    if isinstance(base_estimator, LinearDiscriminantAnalysis) and _is_rank_deficient(X_train):
        skip_bayes = True

    if not skip_bayes:
        search = BayesSearchCV(
            estimator=base_estimator,
            search_spaces=search_spaces,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
        try:
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
        except Exception as exc:
            if verbose:
                print(f"Warning: BayesSearchCV failed with error: {exc}. Falling back to base estimator.")
            skip_bayes = True

    if skip_bayes:
        best_estimator = clone(estimator)
        best_estimator.fit(X_train, y_train)
        best_params = {}
        search = None

    y_pred = best_estimator.predict(X_test)
    if hasattr(best_estimator, "predict_proba"):
        y_scores = best_estimator.predict_proba(X_test)[:, 1]
    elif hasattr(best_estimator, "decision_function"):
        y_scores = best_estimator.decision_function(X_test)
    else:
        y_scores = None

    metrics = _score_predictions(best_estimator, X_test, y_test, y_pred=y_pred, y_scores=y_scores)

    return TrainingResult(
        best_estimator=best_estimator,
        best_params=best_params,
        test_metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        y_scores=y_scores,
        search=search,
    )


if __name__ == "__main__":
    # Example usage with synthetic data.
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from skopt.space import Integer, Categorical

    X_example, y_example = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    rf = RandomForestClassifier(random_state=42)
    param_space = {
        "n_estimators": Integer(100, 600),
        "max_depth": Integer(3, 30),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", None]),
    }

    result = train_model(rf, X_example, y_example, search_spaces=param_space, n_iter=20)
    print("Best parameters:", result.best_params)
    print("Test metrics:", result.test_metrics)
