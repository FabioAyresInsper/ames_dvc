import pathlib
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error

from dvclive import Live


def main():
    DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'

    with open(DATA_DIR / 'processed' / 'ames_features_test.pkl', 'rb') as file:
        Xtest, ytest = pickle.load(file)

    with open(DATA_DIR / 'models' / 'linear_regression.pkl', 'rb') as file:
        model = pickle.load(file)

    ypred = model.predict(Xtest)

    RMSE = np.sqrt(mean_squared_error(ytest, ypred))
    error_percent = 100 * (10**RMSE - 1)

    with Live("metrics") as metrics:
        metrics.log_metric("RMSE", RMSE)
        metrics.log_metric("error_percent", error_percent)


if __name__ == '__main__':
    main()
