import pickle
import pathlib

import pandas as pd


def main():
    DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'

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

    X = model_data.drop(columns=['SalePrice']).copy()
    y = model_data['SalePrice'].copy()

    with open(DATA_DIR / 'processed' / 'ames_features.pkl', 'wb') as file:
        pickle.dump((X, y), file)


if __name__ == '__main__':
    main()
