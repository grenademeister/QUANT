# Helper functions
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series
import matplotlib.pyplot as plt


def acf(series: ndarray, max_lag: int) -> np.ndarray:
    """Return biased autocorrelation sequence up to max_lag."""
    n = len(series)
    mean = series.mean()
    var = ((series - mean) ** 2).sum() / n
    res = np.empty(max_lag + 1)

    for k in range(max_lag + 1):
        num = np.dot(series[: n - k] - mean, series[k:] - mean) / n
        res[k] = num / var
    return res


def levinson_durbin(r: ndarray, p: int) -> np.ndarray:
    """Solve the Yule-Walker equations via Levinson-Durbin recursion."""
    phi = np.zeros(p)
    sig = r[0]

    for k in range(1, p + 1):
        acc = r[k] - sum(phi[j - 1] * r[k - j] for j in range(1, k))
        gamma = acc / sig

        phi_prev = phi.copy()
        phi[k - 1] = gamma
        for j in range(1, k):
            phi[j - 1] = phi_prev[j - 1] - gamma * phi_prev[k - j - 1]

        sig *= 1.0 - gamma**2
    return phi


def difference(x: ndarray, d: int = 1) -> np.ndarray:
    """Apply d-order differencing."""
    for _ in range(d):
        x = np.diff(x)
    return x


def optimize_gd(
    init: ndarray,
    loss_grad_fn,
    lr: float = 1e-3,
    epochs: int = 5000,
    grad_clip: float = 1000.0,
    tol: float = 1e-8,
):
    """
    Generic first-order optimizer.
    Uses gradient descent with clipping

    Parameters
    ----------
    init        : Initial parameter vector (copied internally).
    loss_grad_fn: Callable(params) -> (loss, grad).
    lr          : Learning rate.
    epochs      : Maximum iterations.
    grad_clip   : Gradient absolute-value clipping threshold.
    tol         : Convergence threshold on parameter update norm.
    """
    params = init.copy()

    for _ in range(epochs):
        loss, grad = loss_grad_fn(params)
        grad = np.clip(grad, -grad_clip, grad_clip)
        params -= lr * grad
        step_norm = np.sqrt(np.sum((lr * grad) ** 2))
        if _ % 1000 == 0:
            # print(loss, grad, params)
            continue
        if step_norm < tol and 0:
            # print(f"loss = {loss}, converged")
            break
    return params


# Base model interface
class TimeSeriesModel:
    def fit(self, series: Series):
        raise NotImplementedError

    def predict(self, series: pd.Series, steps: int = 1) -> np.ndarray:
        raise NotImplementedError
