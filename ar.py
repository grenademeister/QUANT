"""AR model implementation"""

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series
import matplotlib.pyplot as plt

from helper import TimeSeriesModel
from helper import acf
from helper import levinson_durbin
from helper import optimize_gd


# AR(p) model
class ARModel(TimeSeriesModel):
    def __init__(self, p: int):
        self.p = p
        self.coef = None

    def fit(self, series: Series):
        print("Fitting AR model...", end="")
        r = acf(series.to_numpy(dtype=float), self.p)
        self.coef = levinson_durbin(r, self.p)
        print("Done!")

    def predict(self, series: Series, steps: int = 1) -> np.ndarray:
        hist = series.to_numpy(dtype=float).tolist()
        out = []
        for _ in range(steps):
            y_hat = np.dot(self.coef, hist[-self.p :][::-1])
            out.append(y_hat)
            hist.append(y_hat)
        return np.array(out)


if __name__ == "__main__":
    np.random.seed(0)

    # Simulate ARIMA(2, 1, 2) series
    n = 300
    ar_coef = np.array([0.6, -0.3])
    ma_coef = np.array([0.5, 0.4])
    noise = np.random.randn(n + 100)
    x = np.zeros(n + 100)

    for t in range(2, len(x)):
        ar_part = np.dot(ar_coef, x[t - 2 : t][::-1])
        ma_part = np.dot(ma_coef, noise[t - 2 : t][::-1])
        x[t] = ar_part + ma_part + noise[t]

    x = np.cumsum(x[100:])  # integration (d = 1)
    ts = pd.Series(x)

    # Fit model
    ar_model = ARModel(2)
    ar_model.fit(ts)

    # Forecast next five points
    print("AR: ", ar_model.predict(ts, 5))
