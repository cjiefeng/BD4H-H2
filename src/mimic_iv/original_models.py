from typing import Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import reports


def fit_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> Tuple[Pipeline, pd.Series, pd.Series]:
    estimators = [
        ("Scaler", StandardScaler()),
        ("Log_Reg", LogisticRegression(random_state=seed, solver="liblinear")),
    ]

    pipe = Pipeline(estimators)
    clf = pipe.fit(X_train, y_train)
    return clf


def fit_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> Tuple[GradientBoostingClassifier, pd.Series, pd.Series]:
    clf = GradientBoostingClassifier(n_estimators=160, max_depth=3, random_state=seed)
    clf = clf.fit(X_train, y_train)
    return clf


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> Tuple[RandomForestClassifier, pd.Series, pd.Series]:
    clf = RandomForestClassifier(n_estimators=140, max_depth=5, random_state=seed)
    clf = clf.fit(X_train, y_train)
    return clf


def fit_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> Tuple[Pipeline, pd.Series, pd.Series]:
    estimators = [
        ("Scaler", StandardScaler()),
        (
            "Neural_Network",
            MLPClassifier(
                hidden_layer_sizes=(100),
                random_state=seed,
                activation="logistic",
                solver="sgd",
                max_iter=1000,
            ),
        ),
    ]
    pipe = Pipeline(estimators)
    clf = pipe.fit(X_train, y_train)
    return clf


def run_model(
    clf: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    print("Training set")
    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_train_pred = clf.predict(X_train)
    reports.print_metrics(y_train, y_train_prob, y_train_pred)
    print("")

    print("Testing set")
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    y_test_pred = clf.predict(X_test)
    reports.print_metrics(y_test, y_test_prob, y_test_pred)
    print("")

    return clf


def run(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
):
    models = [
        ("Logistic Regression", fit_logistic_regression(X_train, y_train, seed)),
        ("Gradient Boosting", fit_gradient_boosting(X_train, y_train, seed)),
        ("Random Forest", fit_random_forest(X_train, y_train, seed)),
        ("MLP", fit_neural_network(X_train, y_train, seed)),
    ]

    for name, clf in models:
        print(f"##### {name} #####")
        run_model(clf, X_train, X_test, y_train, y_test)
