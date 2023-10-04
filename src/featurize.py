import pathlib
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'
    RANDOM_SEED = 42

    clean_data_path = DATA_DIR / 'processed' / 'ames_clean.pkl'

    with open(clean_data_path, 'rb') as file:
        data = pickle.load(file)

    model_data = data.copy()

    categorical_columns = []
    ordinal_columns = []
    for col in model_data.select_dtypes('category').columns:
        if model_data[col].cat.ordered:
            ordinal_columns.append(col)
        else:
            categorical_columns.append(col)

    for col in ordinal_columns:
        codes, _ = pd.factorize(data[col], sort=True)
        model_data[col] = codes

    model_data = pd.get_dummies(model_data, drop_first=True)

    X = model_data.drop(columns=['SalePrice'])
    y = model_data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
    )

    with open(DATA_DIR / 'processed' / 'ames_features_train.pkl',
              'wb') as file:
        pickle.dump((X_train, y_train), file)

    with open(DATA_DIR / 'processed' / 'ames_features_test.pkl', 'wb') as file:
        pickle.dump((X_test, y_test), file)


if __name__ == '__main__':
    main()
