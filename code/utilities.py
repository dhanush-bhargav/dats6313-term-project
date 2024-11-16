import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.api import OLS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels import tsa
import matplotlib.pyplot as plt
from scipy import signal

def calculate_rolling_mean_var(data: pd.Series) -> pd.DataFrame:
    total_samples = data.shape[0]
    rolling_mean = []
    rolling_var = []
    samples = []
    for i in range(total_samples+1):
        samples.append(i)
        rolling_mean.append(data.head(i).mean())
        rolling_var.append(data.head(i).var())

    result_df = pd.DataFrame({'Samples': samples, 'Rolling_Mean': rolling_mean, 'Rolling_Var': rolling_var})
    return result_df

def calculate_adf(x : np.ndarray) -> None:
    result = adfuller(x)

    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: ")
    for key, value in result[4].items():
        print(f"\t{key}: {value :.2f}")

def calculate_kpss(x: np.ndarray) -> None:
    result = kpss(x, regression='c', nlags='auto')
    kpss_output = pd.Series(result[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in result[3].items():
        kpss_output[f"Critical Value ({key})"] = value
    print(kpss_output)

def calculate_non_seasonal_differential(x: np.ndarray, order=1) -> np.ndarray:
    result = x.copy()
    while (order > 0):
        result = result[1:] - result[:-1]
        order -= 1
    return result

def calculate_log_transform(x: np.ndarray) -> np.ndarray:
    result = x.copy()
    return np.log(result)

def calculate_acf(x: np.ndarray, num_lags: int) -> np.ndarray:
    mean = np.mean(x)
    ssq = np.sum((x - mean) ** 2)
    acf = [1]

    for i in range(1, num_lags):
        sum = 0
        for j in range(i+1, x.shape[0] + 1):
            sum += (x[j - 1] - mean) * (x[(j-i) - 1] - mean)
        acf.append(sum / ssq)

    return np.array(acf)

def forecast_average_method(y_train: np.ndarray, y_test: np.ndarray) -> (np.ndarray, np.ndarray):
    y_preds = []
    y_forecasts = [np.mean(y_train)] * y_test.shape[0]
    for i in range(1, y_train.shape[0]):
        y_preds.append(np.mean(y_train[:i]))
    return np.array(y_preds), np.array(y_forecasts)

def prediction_errors(y_train: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if (y_train.shape[0] == y_pred.shape[0]):
        return y_train - y_pred
    else:
        return y_train[(y_train.shape[0] - y_pred.shape[0]):] - y_pred

def forecast_errors(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_test - y_pred

def calc_q_value(residuals: np.ndarray, num_lags: int) -> np.float64:
    acf = calculate_acf(residuals, num_lags+1)
    return (residuals.shape[0]) * np.sum(np.square(acf[1:]))

def forecast_naive_method(y_train: np.ndarray, y_test: np.ndarray) -> (np.ndarray, np.ndarray):
    y_preds = y_train[:-1]
    y_forecasts = [y_train[-1]] * len(y_test)
    return np.array(y_preds), np.array(y_forecasts)

def forecast_drift_method(y_train: np.ndarray, y_test: np.ndarray) -> (np.ndarray, np.ndarray):
    y_forecasts = []
    for i in range(y_test.shape[0]):
        y_forecasts.append(y_train[-1] + (i+1) * ((y_train[-1] - y_train[0]) / y_train.shape[0]))
    y_preds = []
    for i in range(2, y_train.shape[0]):
        y_preds.append(y_train[i-1] + (y_train[i-1] - y_train[0]) / (i-1))
    return np.array(y_preds), np.array(y_forecasts)

def forecast_ses_method(y_train: np.ndarray, y_test: np.ndarray, alpha: float = 0.5, initial_value=None) -> (np.ndarray, np.ndarray):
    y_preds = []

    initial_value = initial_value if initial_value is not None else y_train[0]
    y_preds.append(initial_value)

    for i in range(1, y_train.shape[0]):
        y_preds.append(alpha * y_train[i-1] + (1 - alpha) * y_preds[i-1])

    y_forecasts = [alpha * y_train[-1] + (1-alpha) * y_preds[-1]] * y_test.shape[0]

    return np.array(y_preds[1:]), np.array(y_forecasts)

def calc_moving_average(x: np.ndarray, inner_window: int, outer_window: int=None) -> np.ndarray:
    if inner_window % 2 == 0:
        outer_window = outer_window or 2
        return calc_ma_even(x, inner_window, outer_window)
    else:
        return calc_ma_odd(x, inner_window)

def calc_ma_odd(x: np.ndarray, window: int) -> np.ndarray:
    result = []
    k = (window - 1) // 2
    for i in range(k, x.shape[0] - k):
        result.append(
            (1 / float(window)) * np.sum(x[(i-k):(i+k)])
        )
    return np.array(result)

def calc_ma_even(x: np.ndarray, inner_window: int, outer_window: int) -> np.ndarray:
    inner_result = []
    outer_result = []
    k1 = inner_window // 2
    k2 = outer_window // 2

    for i in range(k1, x.shape[0] - k1):
        inner_result.append(
            (1 / float(inner_window)) * np.sum(x[(i-k1):(i+k1-1)])
        )

    inner_result = np.array(inner_result)

    for i in range(k2, inner_result.shape[0] - k2):
        outer_result.append(
            (1 / float(outer_window)) * np.sum(inner_result[(i-k2):(i+k2-1)])
        )

    return np.array(outer_result)

def additive_decomposition(trend: np.ndarray, seasonality: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    if trend.shape[0] == seasonality.shape[0] and seasonality.shape[0] == residuals.shape[0]:
        return trend + seasonality + residuals
    else:
        raise ValueError('Trend, Seasonality and Residuals must have the same shape')

def multiplicative_decomposition(trend: np.ndarray, seasonality: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    if trend.shape[0] == seasonality.shape[0] and seasonality.shape[0] == residuals.shape[0]:
        return trend * seasonality * residuals
    else:
        raise ValueError('Trend, Seasonality and Residuals must have the same shape')

def moving_average_interactive(x: np.ndarray) -> np.ndarray:
    while(True):
        inner_window = input("Enter the moving average order (positive integer except 1 and 2): ")
        if inner_window.isnumeric() and int(inner_window) in range(3, x.shape[0] + 1):
            if int(inner_window) % 2 == 0:
                outer_window = input("Enter the folding order (positive even integer): ")
                if outer_window.isnumeric() and int(outer_window) % 2 == 0:
                    return calc_moving_average(x, int(inner_window), int(outer_window))
                else:
                    print("Folding order must be an even integer. Try again.")
            else:
                return calc_moving_average(x, int(inner_window))
        else:
            print("Invalid input. Try again.")


def simulate_ar_process():
    while(True):
        num_samples = input("Enter the number of samples: ")
        if num_samples.isnumeric():
            num_samples = int(num_samples)
        else:
            print("Num samples must be an integer. Try again.")
            continue
        order = input("Enter the order # of the AR process: ")
        if order.isnumeric():
            order = int(order)
        else:
            print("Order must be an integer. Try again.")
            continue
        coefficients = input("Enter the coefficients of the AR process (comma seperated): ")
        coefficients = [-1.0 * float(c.strip()) for c in coefficients.split(',')]
        if len(coefficients) != order:
            print("Invalid number of coefficients. Try again.")
            continue

        e_t = np.random.normal(0, 1, num_samples)
        y_t = np.zeros((num_samples, ))
        coefficients = coefficients[::-1]
        for t in range(num_samples):
            if t == 0:
                y_t[t] = e_t[t]
            elif 0 < t < order:
                y_t[t] = e_t[t] + np.dot(coefficients[:t], y_t[:t])
            else:
                y_t[t] = e_t[t] + np.dot(coefficients, y_t[t-order:t])
        data_dict = {
            "target": y_t[order:]
        }
        for o in range(1, order + 1):
            data_dict[f"feature{o}"] = y_t[order - o: -o]

        df = pd.DataFrame(data_dict)

        X = df.drop(columns=["target"]).to_numpy()
        y = df["target"].to_numpy()
        hats = np.linalg.inv(X.T @ X) @ X.T @ y

        return hats

def simulate_arma_process():
    while(True):
        num_samples = input("Enter the number of data samples: ")
        if num_samples.isnumeric():
            num_samples = int(num_samples)
        else:
            print("Num samples must be an integer. Try again.")
            continue
        mean = input("Enter the mean of the white noise: ")
        if mean.isnumeric():
            mean = float(mean)
        else:
            print("Mean must be a number. Try again.")
            continue
        variance = input("Enter the variance of the white noise: ")
        if variance.isnumeric():
            variance = float(variance)
        else:
            print("Variance must be a number. Try again.")
            continue
        ar_order = input("Enter the order # of the AR process: ")
        if ar_order.isnumeric():
            ar_order = int(ar_order)
        else:
            print("AR Order must be an integer. Try again.")
            continue
        ma_order = input("Enter the order # of the MA process: ")
        if ma_order.isnumeric():
            ma_order = int(ma_order)
        else:
            print("MA Order must be an integer. Try again.")
            continue

        if ar_order > 0:
            ar_coeffs_string = input("Enter coefficients of AR (lag-polynomial representation, comma seperated): ")
            ar_coeffs = [float(c.strip()) for c in ar_coeffs_string.split(',')]

            if len(ar_coeffs) != ar_order:
                print("Invalid number of coefficients. Try again.")
                continue
            else:
                ar_coeffs.insert(0, 1.0)
        else:
            ar_coeffs = [1.0]

        if ma_order > 0:
            ma_coeffs_string = input("Enter coefficients of MA (lag-polynomial representation, comma seperated): ")
            ma_coeffs = [float(c.strip()) for c in ma_coeffs_string.split(',')]

            if len(ma_coeffs) != ma_order:
                print("Invalid number of coefficients. Try again.")
                continue
            else:
                ma_coeffs.insert(0, 1.0)
        else:
            ma_coeffs = [1.0]

        break

    arma_process = tsa.arima_process.ArmaProcess(ar_coeffs, ma_coeffs, num_samples)

    return arma_process

def generate_gpac_table(acf, j_max=7, k_max=7):
    phi_table = np.zeros((j_max, k_max))
    for j in range(j_max):
        for k in range(1, k_max):
            den_mat = np.zeros((k, k))
            window = np.arange(start=j, stop=j+k)
            for t in range(k):
                window = np.abs(window - t)
                den_mat[:, t] = acf[window]
            num_mat = den_mat.copy()
            num_mat[:, k-1] = acf[np.arange(start=j+1, stop=j+k+1)]
            if np.linalg.det(num_mat) == 0:
                phi_table[j, k] = 0
            else:
                if np.linalg.det(den_mat) != 0:
                    phi_table[j, k] = np.linalg.det(num_mat) / np.linalg.det(den_mat)
                else:
                    phi_table[j, k] = np.nan

    return phi_table

def plot_acf_pacf(y, n_lags):
    acf = tsa.stattools.acf(y, nlags=n_lags)
    pacf = tsa.stattools.pacf(y, nlags=n_lags)
    fig = plt.figure()
    plt.subplot(211)
    plot_acf(y, ax=plt.gca(), lags=n_lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=n_lags)
    fig.tight_layout(pad=3)
    plt.show()

def lm_algorithm(y, ar_order, ma_order, max_iter=10, delta=1e-6, eps=1e-5, mu=1e-2, mu_max=1e3):
    num_params = ar_order + ma_order
    params = np.zeros((num_params, ))
    sigma_error = None
    cov_params = None
    num_iter = 1
    sse_hist = []

    while num_iter <= max_iter:
        ar = [1, *params[:ar_order].flatten()]
        ma = [1, *params[ar_order:].flatten()]
        ar = np.pad(ar, (0, max(0, ma_order - ar_order)), mode='constant')
        ma = np.pad(ma, (0, max(0, ar_order - ma_order)), mode='constant')
        _, residuals = signal.dlsim((ar, ma, 1.0), y)
        residuals = residuals.flatten()
        sse = residuals.T @ residuals
        if num_iter == 1:
            sse_hist.append(sse)
        X = []
        for i in range(num_params):
            params_i = params.copy()
            params_i[i] = params[i] + delta
            ar_i = [1, *params_i[:ar_order].flatten()]
            ma_i = [1, *params_i[ar_order:].flatten()]
            ar_i = np.pad(ar_i, (0, max(0, ma_order - ar_order)), mode='constant')
            ma_i = np.pad(ma_i, (0, max(0, ar_order - ma_order)), mode='constant')
            _, residuals_i = signal.dlsim((ar_i, ma_i, 1.0), y)
            residuals_i = residuals_i.flatten()
            x_i = (residuals - residuals_i) / delta
            X.append(x_i)
        X = np.array(X)
        X = X.T
        A = X.T @ X
        g = X.T @ residuals
        del_param = np.linalg.inv((A + mu * np.identity(num_params))) @ g
        params_new = params + del_param
        ar_new = [1, *params_new[:ar_order].flatten()]
        ma_new = [1, *params_new[ar_order:].flatten()]
        ar_new = np.pad(ar_new, (0, max(0, ma_order - ar_order)), mode='constant')
        ma_new = np.pad(ma_new, (0, max(0, ar_order - ma_order)), mode='constant')
        _, residuals_new = signal.dlsim((ar_new, ma_new, 1.0), y)
        residuals_new = residuals_new.flatten()
        sse_new = residuals_new.T @ residuals_new

        if sse_new < sse:
            if np.linalg.norm(del_param, 2) < eps:
                params = params_new
                sigma_error = sse_new / (y.shape[0] - num_params)
                cov_params = sigma_error * np.linalg.inv(A)
                sse_hist.append(sse_new)
                break
            else:
                params = params_new
                mu = mu / 10
        while sse_new >= sse:
            mu *= 10
            if mu > mu_max:
                break
            else:
                del_param = np.linalg.inv((A + mu * np.identity(num_params))) @ g
                params_new = params + del_param
                ar_new = [1, *params_new[:ar_order].flatten()]
                ma_new = [1, *params_new[ar_order:].flatten()]
                ar_new = np.pad(ar_new, (0, max(0, ma_order - ar_order)), mode='constant')
                ma_new = np.pad(ma_new, (0, max(0, ar_order - ma_order)), mode='constant')
                _, residuals_new = signal.dlsim((ar_new, ma_new, 1.0), y)
                residuals_new = residuals_new.flatten()
                sse_new = residuals_new.T @ residuals_new

        if mu > mu_max:
            sse_hist.append(sse_new)
            break

        num_iter += 1
        sse_hist.append(sse_new)
        params = params_new

    if num_iter > max_iter:
        print("Error: Model did not converge")
        return params, sigma_error, cov_params, sse_hist
    else:
        if mu > mu_max:
            print("Error: Mu exceeded max value")
            return params, sigma_error, cov_params, sse_hist
        else:
            return params, sigma_error, cov_params, sse_hist