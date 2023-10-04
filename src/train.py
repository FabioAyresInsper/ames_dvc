import pathlib
import pickle

from sklearn.linear_model import LinearRegression


def main():
    DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'

    with open(DATA_DIR / 'processed' / 'ames_features_train.pkl',
              'rb') as file:
        X_train, y_train = pickle.load(file)

    model = LinearRegression()
    model.fit(X_train, y_train)

    model_dir = DATA_DIR / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(DATA_DIR / 'models' / 'linear_regression.pkl', 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    main()
