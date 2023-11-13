import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score


def print_metrics(
    y_test: pd.Series,
    y_prob: pd.Series,
    y_pred: pd.Series,
):
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC: " + str(roc_auc))

    average_precision = average_precision_score(y_test, y_prob)
    print("Average Precision: " + str(average_precision))

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(accuracy))
