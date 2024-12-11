#%%
import os
from utilities import *
from CONSTANTS import *
import seaborn as sns

#%%
train_df = pd.read_csv('../data/currency_exchange_rates_train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df = pd.read_csv('../data/currency_exchange_rates_test.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])

#%%
y_train = train_df[DEPENDENT_FEATURES].to_numpy().flatten()
y_test = test_df[DEPENDENT_FEATURES].to_numpy().flatten()
u_train = train_df['Pakistani Rupee'].to_numpy()
u_test = test_df['Pakistani Rupee'].to_numpy()


#%%
R_u_t = acf_exog_box_jenkins(u_train, max_k=100)
R_uy_t = acf_cross_box_jenkins(u_train, y_train, max_k=100)
impulse_response = np.linalg.inv(R_u_t) @ R_uy_t

#%%
g_gpac = generate_gpac_table(impulse_response, 7,7)
g_gpac = np.delete(g_gpac, 0, axis=1)
plt.figure()
sns.heatmap(g_gpac, annot=True, xticklabels=[1,2,3,4,5,6])
plt.title("G-GPAC table")
plt.show()

#%%
v_t = calc_vt_box_jenkins(y_train, impulse_response.flatten(), u_train)
v_t_acf = calculate_acf(v_t, 100)

#%%
h_gpac = generate_gpac_table(v_t_acf, 7, 7)
h_gpac = np.delete(h_gpac, 0, 1)
plt.figure()
sns.heatmap(h_gpac, annot=True, xticklabels=[1,2,3,4,5,6])
plt.title("H-GPAC table for estimating n_c and n_d")
plt.show()

#%%
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y_train, u_train, order=(1, 0, 0)).fit()
print(model.summary())

#%%
y_pred = model.predict()
pred_errors = y_train - y_pred
q_value = calc_q_value(pred_errors[1:], 20)
print(f"Q-value: {q_value:.4f}")
#%%
cross_acf = acf_cross_box_jenkins(calculate_non_seasonal_differential(u_train, 1), pred_errors[1:], max_k=50) / (np.var(pred_errors) * np.var(calculate_non_seasonal_differential(u_train, 1)))
s_value = pred_errors.shape[0] * np.sum(np.square(cross_acf))
print(f"S-value = {s_value:.4f}")

#%%
from scipy.stats import chi2
print(f"Q-value critical = {chi2.ppf(0.95, df=19)}")
print(f"Critical S-value = {chi2.ppf(0.95, df=50)}")

#%%
plt.figure()
plt.scatter(calculate_non_seasonal_differential(u_train, 1), pred_errors[1:])
plt.xlabel("Pre-whitened input")
plt.ylabel("Prediction Error")
plt.title("Relationship between residuals and input")
plt.tight_layout()
plt.show()

#%%
plot_acf_pacf(pred_errors, 20)

#%%
print(f"Mean value of residuals = {np.mean(pred_errors) :.4f}")
print(f"Variance of residuals = {np.var(pred_errors) :.4f}")

#%%
plt.figure()
plt.plot(range(y_train.shape[0]), y_train, label='Train')
plt.plot(range(1,y_train.shape[0]), y_pred[1:], label='Prediction')
plt.legend(loc='best')
plt.title("Prediction of Indian Rupee using Box Jenkins")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
T = y_train.shape[0]
y_forecast = []
for i in range(y_test.shape[0]):
    y = 0
    if i == 0:
        y = y_train[T-1] + 0.0375 * (u_test[0] - u_train[T-1])
    else:
        y = y_forecast[i-1] + 0.0375 * (u_test[i] - u_test[i-1])
    y_forecast.append(y)

#%%
y_forecast = np.array(y_forecast)
plt.figure()
plt.plot(range(y_test.shape[0]), y_test, label='Test')
plt.plot(range(y_test.shape[0]), y_forecast, label='Forecast')
plt.legend(loc='best')
plt.title("Forecast of Indian Rupee using Box Jenkins")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
forecast_errors = y_test - y_forecast
print(f"Mean of forecast errors = {np.mean(forecast_errors) :.4f}")
print(f"Variance of forecast errors = {np.var(forecast_errors) :.4f}")
