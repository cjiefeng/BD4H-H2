from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(dir: str, drop_cols: List[str]) -> pd.DataFrame:
    data = pd.read_csv(dir).drop(drop_cols, axis=1)
    return data


def one_hot_encode(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
        data = data.drop([col], axis=1)
    return data


def clean_train_test_split(
    data: pd.DataFrame,
    drop_cols: List[str],
    label_col: str,
    seed,
    test_size: float = 0.9,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(drop_cols, axis=1)
    y = data[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    return X_train, X_test, y_train, y_test


def run(
    dir: str,
    extra_cols: List[str],
    ohe_cols: List[str],
    drop_cols: List[str],
    label_col: str,
    seed: int,
):
    data = read_csv(
        dir,
        extra_cols,
    )

    data = one_hot_encode(
        data,
        ohe_cols,
    )

    X_train, X_test, y_train, y_test = clean_train_test_split(
        data,
        drop_cols,
        label_col,
        seed,
    )

    return X_train, X_test, y_train, y_test
