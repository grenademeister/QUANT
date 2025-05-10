"""MA model implementation"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper import TimeSeriesModel
from helper import acf
from helper import levinson_durbin
from helper import optimize_gd


# MA(q) model
class MAModel(TimeSeriesModel):
    def __init__(self, q: int, lr: float = 1e-4, epochs: int = 10000):
        self.q, self.lr, self.epochs = q, lr, epochs
        self.mu = None
        self.std = None
        self.theta = None

    # the fit method is written by Clade 3.7 Sonnet.
    # since I am stupid, I failed to write working code.
    def fit(self, series: pd.Series):
        print("Fitting MA model...", end="")
        x_original = series.to_numpy(dtype=float)
        self.mu = x_original.mean()
        self.std = x_original.std()
        x = (x_original - self.mu) / self.std
        n, q = len(x), self.q

        def loss_grad(theta):
            eps = np.zeros(n)

            # Forward pass - compute residuals
            for t in range(n):
                eps[t] = x[t]
                for j in range(min(t, q)):
                    eps[t] -= theta[j] * eps[t - j - 1]

            # Compute loss
            loss = 0.5 * np.mean(eps**2)

            # Compute gradient more stably by using finite differences
            # for the most sensitive components
            grad = np.zeros(q)
            for i in range(q):
                h = 1e-6  # Small perturbation
                theta_plus = theta.copy()
                theta_plus[i] += h

                # Compute perturbed residuals
                eps_plus = np.zeros(n)
                for t in range(n):
                    eps_plus[t] = x[t]
                    for j in range(min(t, q)):
                        eps_plus[t] -= theta_plus[j] * eps_plus[t - j - 1]

                # Compute numerical gradient
                loss_plus = 0.5 * np.mean(eps_plus**2)
                grad[i] = (loss_plus - loss) / h

            return loss, grad

        # Use a smaller learning rate and tighter gradient clipping
        self.theta = optimize_gd(
            np.zeros(q),  # Start from zeros instead of random
            loss_grad,
            lr=self.lr,
            epochs=self.epochs,
            grad_clip=1.0,  # Much tighter gradient clipping
        )
        print("Done!")

    def predict(self, series: pd.Series, steps: int = 1) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        x_original = series.to_numpy(dtype=float)
        x = (x_original - self.mu) / self.std
        eps_hist = np.zeros(len(x))

        # Compute residuals on training data (used as history)
        for t in range(len(x)):
            eps_hist[t] = x[t]
            for j in range(min(t, self.q)):
                eps_hist[t] -= self.theta[j] * eps_hist[t - j - 1]

        preds = []
        for _ in range(steps):
            pred = self.mu
            for j in range(min(self.q, len(eps_hist))):
                pred += self.theta[j] * self.std * eps_hist[-(j + 1)]
            preds.append(pred)
            eps_hist = np.append(eps_hist, 0.0)

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

    # Fit model
    ma_model = MAModel(2, lr=1e-3, epochs=10000)
    ma_model.fit(ts)

    # Forecast next five points
    print("MA    →", ma_model.predict(ts, 5))
