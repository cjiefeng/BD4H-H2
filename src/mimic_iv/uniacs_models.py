import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import medical_explainer
from utils import reports


def explain_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
):
    print("##### Logistic Regression #####")
    estimators = [
        ("Scaler", StandardScaler()),
        ("Log_Reg", LogisticRegression(random_state=seed, solver="liblinear")),
    ]
    clf = Pipeline(estimators)
    clf.fit(X_train, y_train)

    explainer = medical_explainer.explainer(clf, X_train, y_train, X_test, y_test)
    explainer.fit(10, shap_method="linear", method="novel")
    scores, probs, predictions = explainer.predict_calculator(X_test)
    print("Uniacs testing set")
    reports.print_metrics(y_test, probs, predictions)
    print("")
    explainer.print_calculator()
    explainer.plot_calculator_features(save_image=True)
    print([x + np.abs(np.min(x)) for x in explainer.score_array_list])
    print("")


def explain_gradient_boosting(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
):
    print("##### Gradient Boosting #####")
    clf = GradientBoostingClassifier(n_estimators=160, max_depth=3, random_state=seed)
    clf.fit(X_train, y_train)

    explainer = medical_explainer.explainer(clf, X_train, y_train, X_test, y_test)
    explainer.fit(10, shap_method="tree", method="novel")
    scores, probs, predictions = explainer.predict_calculator(X_test)
    print("Uniacs testing set")
    reports.print_metrics(y_test, probs, predictions)
    print("")
    explainer.print_calculator()
    explainer.plot_calculator_features(save_image=True)
    print([x + np.abs(np.min(x)) for x in explainer.score_array_list])
    print("")


def explain_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
):
    print("##### Random Forest #####")
    clf = RandomForestClassifier(n_estimators=140, max_depth=5, random_state=seed)
    clf.fit(X_train, y_train)

    explainer = medical_explainer.explainer(clf, X_train, y_train, X_test, y_test)
    explainer.fit(10, shap_method="tree", method="novel", calculator_threshold=0.0001)
    scores, probs, predictions = explainer.predict_calculator(X_test)
    print("Uniacs testing set")
    reports.print_metrics(y_test, probs, predictions)
    print("")
    explainer.print_calculator()
    explainer.plot_calculator_features(save_image=True)
    print([x + np.abs(np.min(x)) for x in explainer.score_array_list])
    print("")


def explain_neural_network(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
):
    print("##### MLP #####")
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
    clf = Pipeline(estimators)
    clf.fit(X_train, y_train)

    explainer = medical_explainer.explainer(clf, X_train, y_train, X_test, y_test)
    explainer.fit(10, shap_method="kernel", method="novel", n_splits=2)
    scores, probs, predictions = explainer.predict_calculator(X_test)
    print("Uniacs testing set")
    reports.print_metrics(y_test, probs, predictions)
    print("")
    explainer.print_calculator()
    explainer.plot_calculator_features(save_image=True)
    print([x + np.abs(np.min(x)) for x in explainer.score_array_list])
    print("")


def run(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
):
    explain_logistic_regression(X_train, X_test, y_train, y_test, seed)
    explain_gradient_boosting(X_train, X_test, y_train, y_test, seed)
    explain_random_forest(X_train, X_test, y_train, y_test, seed)
    explain_neural_network(X_train, X_test, y_train, y_test, seed)
