import pandas as pd
from sklearn.model_selection import train_test_split
from mimic_iv import original_models, uniacs_models
def main():
    seed = 9973

    # Load the dataset and drop unnecessary columns
    data = pd.read_csv('../data/mimiciii_sepsis.csv')
    data.drop(['Unnamed: 0', 'SUBJECT_ID', 'HADM_ID'], axis=1, inplace=True)

    # Convert gender to numeric values: 1 for Male and 0 for Female
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)

    X = data.drop(['28 Day Death', 'In Hospital Death'], axis=1)
    y = data['28 Day Death']

    # Convert all columns to numeric and fill missing values with column mean
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors='coerce').fillna(X[column].mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    print("original models")
    original_models.run(X_train, X_test, y_train, y_test, seed)
    print("")
    print("------------------------------------------------------------------")
    print("uniacs models")
    uniacs_models.run(X_train, X_test, y_train, y_test, seed)
    print("")


if __name__ == "__main__":
    main()