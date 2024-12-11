#%%
from utilities import *
from CONSTANTS import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns

#%%
train_df = pd.read_csv('../data/currency_exchange_rates_train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df = pd.read_csv('../data/currency_exchange_rates_test.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])

#%%
y_train = train_df[DEPENDENT_FEATURES].to_numpy().flatten()
y_test = test_df[DEPENDENT_FEATURES].to_numpy().flatten()

#%%
y_preds_avg, y_forecast_avg = forecast_average_method(y_train, y_test)
y_preds_naive, y_forecast_naive = forecast_naive_method(y_train, y_test)
y_preds_drift, y_forecast_drift = forecast_drift_method(y_train, y_test)
y_preds_ses, y_forecast_ses = forecast_ses_method(y_train, y_test, alpha=0.3, initial_value=y_train[0])

#%%
plt.figure()
plt.plot(range(y_train.shape[0]), y_train, label='Train')
plt.plot(range(1,y_train.shape[0]), y_preds_avg, label='Prediction')
plt.legend(loc='best')
plt.title("Prediction of Indian Rupee using Average Method")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
plt.figure()
plt.plot(range(y_train.shape[0]), y_train, label='Train')
plt.plot(range(1,y_train.shape[0]), y_preds_naive, label='Prediction')
plt.legend(loc='best')
plt.title("Prediction of Indian Rupee using Naive Method")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
plt.figure()
plt.plot(range(y_train.shape[0]), y_train, label='Train')
plt.plot(range(2,y_train.shape[0]), y_preds_drift, label='Prediction')
plt.legend(loc='best')
plt.title("Prediction of Indian Rupee using Drift Method")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
plt.figure()
plt.plot(range(y_train.shape[0]), y_train, label='Train')
plt.plot(range(1,y_train.shape[0]), y_preds_ses, label='Prediction')
plt.legend(loc='best')
plt.title("Prediction of Indian Rupee using SES(0.3) Method")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
residuals_avg = y_train[2:] - y_preds_avg[1:]
residuals_drift = y_train[2:] - y_preds_drift
residuals_ses = y_train[2:] - y_preds_ses[1:]
residuals_naive = y_train[2:] - y_preds_naive[1:]

#%%
print(f"Mean of prediction error (Average Method): {np.mean(residuals_avg) :.4f}")
print(f"Variance of prediction error (Average Method): {np.var(residuals_avg) :.4f}")
print(f"Mean of prediction error (Drift Method): {np.mean(residuals_drift) :.4f}")
print(f"Variance of prediction error (Drift Method): {np.var(residuals_drift) :.4f}")
print(f"Mean of prediction error (SES Method): {np.mean(residuals_ses):.4f}")
print(f"Variance of prediction error (SES Method): {np.var(residuals_ses):.4f}")
print(f"Mean of prediction error (Naive Method): {np.mean(residuals_naive) :.4f}")
print(f"Variance of prediction error (Naive Method): {np.var(residuals_naive) :.4f}")

#%%
q_value_avg = calc_q_value(residuals_avg, 20)
q_value_drift = calc_q_value(residuals_drift, 20)
q_value_ses = calc_q_value(residuals_ses, 20)
q_value_naive = calc_q_value(residuals_naive, 20)

#%%
print(f"Q-value of Average Method: {q_value_avg:.4f}")
print(f"Q-value of Drift Method: {q_value_drift:.4f}")
print(f"Q-value of SES Method: {q_value_ses:.4f}")
print(f"Q-value of Naive Method: {q_value_naive:.4f}")

#%%
plt.figure()
plt.plot(y_test, label='Test')
plt.plot(y_forecast_drift, label='Forecast')
plt.title("Forecast of Indian Rupee using Drift Method")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
forecast_errors_drift = y_test - y_forecast_drift
print(f"Mean of forecast errors = {np.mean(forecast_errors_drift) :.4f}")
print(f"Variance of forecast errors = {np.var(forecast_errors_drift) :.4f}")

#%%
plot_acf_pacf(y_train, 20)

#%%
acf = calculate_acf(y_train, 20)
gpac_table = generate_gpac_table(acf, 7,7)
gpac_table = np.delete(gpac_table, 0, axis=1)

#%%
plt.figure()
sns.heatmap(gpac_table, annot=True, xticklabels=[1,2,3,4,5,6])
plt.xlabel('AR order')
plt.ylabel('MA order')
plt.title(f'GPAC table of {DEPENDENT_FEATURES[0]}')
plt.show()

#%%
model = SARIMAX(y_train, order=(1,0,0)).fit()
print(model.summary())

#%%
y_preds = model.predict()
pred_errors = y_train - y_preds
q_value = calc_q_value(pred_errors[1:], 20)
print(f"Q-value: {q_value:.4f}")

#%%
plot_acf_pacf(pred_errors, 20)

