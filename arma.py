"""ARMA model implementations"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper import TimeSeriesModel
from helper import optimize_gd


class ARMAModel(TimeSeriesModel):
    def __init__(self, p: int, q: int, lr: float = 1e-4, epochs: int = 10000):
        self.p, self.q, self.lr, self.epochs = p, q, lr, epochs
        self.ar_coef = None
        self.ma_coef = None
        self.mu = None
        self.std = None

    def fit(self, series: pd.Series):
        print("Fitting ARMA model...", end="")
        x_original = series.to_numpy(dtype=float)
        self.mu = x_original.mean()
        self.std = x_original.std()
        x = (x_original - self.mu) / self.std
        n, p, q = len(x), self.p, self.q

        def loss_grad(params):
            eps = np.zeros(n)
            ar_params = params[:p]
            ma_params = params[p:]

            # Compute residuals
            for t in range(n):
                eps[t] = x[t]
                for j in range(min(t, p)):
                    eps[t] -= ar_params[j] * x[t - j - 1]
                for j in range(min(t, q)):
                    eps[t] -= ma_params[j] * eps[t - j - 1]

            # Compute loss
            loss = 0.5 * np.mean(eps**2)

            # Compute gradients - finite difference
            grad = np.zeros(p + q)
            # ar params
            for i in range(p):
                h = 1e-6
                params_plus = params.copy()
                params_plus[i] += h
                ar_plus = params_plus[:p]
                ma_plus = params_plus[p:]

                eps_plus = np.zeros(n)
                for t in range(n):
                    eps_plus[t] = x[t]
                    for j in range(min(t, p)):
                        eps_plus[t] -= ar_plus[j] * x[t - j - 1]
                    for j in range(min(t, q)):
                        eps_plus[t] -= ma_plus[j] * eps_plus[t - j - 1]

                loss_plus = 0.5 * np.mean(eps_plus**2)
                grad[i] = (loss_plus - loss) / h

            # ma params
            for i in range(q):
                h = 1e-6
                params_plus = params.copy()
                params_plus[p + i] += h
                ar_plus = params_plus[:p]
                ma_plus = params_plus[p:]

                eps_plus = np.zeros(n)
                for t in range(n):
                    eps_plus[t] = x[t]
                    for j in range(min(t, p)):
                        eps_plus[t] -= ar_plus[j] * x[t - j - 1]
                    for j in range(min(t, q)):
                        eps_plus[t] -= ma_plus[j] * eps_plus[t - j - 1]

                loss_plus = 0.5 * np.mean(eps_plus**2)
                grad[p + i] = (loss_plus - loss) / h
            return loss, grad

        params = optimize_gd(
            np.zeros(p + q),
            loss_grad,
            lr=self.lr,
            epochs=self.epochs,
            grad_clip=1.0,
        )
        self.ar_coef = params[:p]
        self.ma_coef = params[p:]
        print("Done!")

    def predict(self, series: pd.Series, steps: int = 1) -> np.ndarray:
        if self.ar_coef is None or self.ma_coef is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        x_original = series.to_numpy(dtype=float)
        x = (x_original - self.mu) / self.std
        n = len(x)
        eps = np.zeros(n)

        # Compute residuals
        for t in range(n):
            eps[t] = x[t]
            for j in range(min(t, self.p)):
                eps[t] -= self.ar_coef[j] * x[t - j - 1]
            for j in range(min(t, self.q)):
                eps[t] -= self.ma_coef[j] * eps[t - j - 1]

        preds = []
        forecast_x = np.append(x, np.zeros(steps))
        forecast_eps = np.append(eps, np.zeros(steps))

        for t in range(n, n + steps):
            forecast = self.mu
            for j in range(self.p):
                if t - j - 1 >= 0:
                    forecast += self.ar_coef[j] * self.std * forecast_x[t - j - 1]
            for j in range(self.q):
                if t - j - 1 >= 0:
                    forecast += self.ma_coef[j] * self.std * forecast_eps[t - j - 1]

            preds.append(forecast)
            forecast_x[t] = (forecast - self.mu) / self.std
        return np.array(preds)


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

    # Forecast next five points
    print("ARMA coeffs: ", arma_model.ar_coef, arma_model.ma_coef)
    print("ARMA predictions: ", arma_model.predict(ts.diff().dropna(), 5))
