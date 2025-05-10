"""ARIMA model implementations"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper import TimeSeriesModel
from helper import optimize_gd
from arma import ARMAModel


class ARIMAModel(TimeSeriesModel):
    def __init__(self, p: int, d: int, q: int, lr: float = 1e-4, epochs: int = 10000):
        self.p = p  # AR order
        self.d = d  # Differencing order
        self.q = q  # MA order
        self.lr = lr
        self.epochs = epochs
        self.arma_model = ARMAModel(p, q, lr, epochs)
        self.original_series = None

    def fit(self, series: pd.Series):
        print(f"Fitting ARIMA({self.p},{self.d},{self.q}) model...", end="")
        self.original_series = series.copy()

        # Apply d-order differencing
        differenced_series = series.copy()
        for _ in range(self.d):
            differenced_series = differenced_series.diff().dropna()

        # Fit ARMA model on differenced series
        self.arma_model.fit(differenced_series)
        print("ARIMA model fitted!")

    def predict(self, series: pd.Series, steps: int = 1) -> np.ndarray:
        if self.arma_model.ar_coef is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        latest_series = series.copy()
        differenced_series = latest_series.copy()
        for _ in range(self.d):
            differenced_series = differenced_series.diff().dropna()

        diff_forecasts = self.arma_model.predict(differenced_series, steps)

        # Undo differencing (integration)
        if self.d == 0:
            return diff_forecasts
        last_values = [latest_series.iloc[-i - 1] for i in range(self.d)]

        forecasts = np.zeros(steps)
        for i in range(steps):
            value = diff_forecasts[i]
            for j in range(self.d):
                value += last_values[j]
            forecasts[i] = value
            for j in range(self.d - 1, 0, -1):
                last_values[j] = last_values[j - 1]
            last_values[0] = forecasts[i]

        return forecasts


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

    # Fit models
    arma_model = ARMAModel(2, 2, lr=1e-2, epochs=5000)
    arma_model.fit(ts.diff().dropna())  # Manually difference for ARMA

    arima_model = ARIMAModel(2, 1, 2, lr=1e-2, epochs=5000)
    arima_model.fit(ts)

    # Forecast next five points
    print("ARMA coeffs: ", arma_model.ar_coef, arma_model.ma_coef)
    print("ARIMA coeffs: ", arma_model.ar_coef, arma_model.ma_coef)
    print("ARMA predictions: ", arma_model.predict(ts.diff().dropna(), 5))
    print("ARIMA predictions: ", arima_model.predict(ts, 5))
