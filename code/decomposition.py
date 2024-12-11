#%%
from utilities import *
from CONSTANTS import *
from statsmodels.tsa.seasonal import STL
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#%%
train_df = pd.read_csv('../data/currency_exchange_rates_train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df_final = train_df[ INDEPENDENT_FEATURES + DEPENDENT_FEATURES]
train_df_final.set_index('Date', inplace=True)

#%%
data = train_df_final[DEPENDENT_FEATURES]
stl = STL(data, period=7, seasonal=7)
res = stl.fit()
res.plot()
plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%%
trend_strength = max(0, 1 - (np.var(res.resid) / np.var(res.resid + res.trend)))
print(f'The strength of trend for this dataset is: {trend_strength * 100:.2f}%')

#%%
seasonality_strength = max(0, 1 - (np.var(res.resid) / np.var(res.resid + res.seasonal)))
print(f'The strength of seasonality for this dataset is: {seasonality_strength * 100: .2f}%')

#%%
from statsmodels.tsa.holtwinters import Holt

holt_model = Holt(data).fit()
print(holt_model.summary())

#%%
test_df = pd.read_csv('../data/currency_exchange_rates_test.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df_final = test_df[INDEPENDENT_FEATURES + DEPENDENT_FEATURES]
test_df_final.set_index('Date', inplace=True)

#%%
start = train_df_final.shape[0]
end = train_df_final.shape[0] + test_df_final.shape[0] - 1
predictions = holt_model.predict(start, end)

#%%
test_data = test_df_final[DEPENDENT_FEATURES].copy()
test_data['Predictions'] = predictions.to_numpy()

#%%
plt.figure()
plt.plot(data, label='training')
plt.plot(test_data[DEPENDENT_FEATURES], label='test')
plt.plot(test_data['Predictions'], label='predictions')
plt.title("Holt Winter Method")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid()
plt.legend()
plt.show()

#%%
forecast_errors = test_data[DEPENDENT_FEATURES].to_numpy().flatten() - predictions.to_numpy().flatten()

#%%
acf = calculate_acf(forecast_errors, 21)
acf_sym = np.concatenate([acf[-1:0:-1], acf])
conf_int = 1.96 / np.sqrt(forecast_errors.shape[0])

locs = [i for i in range(-acf.shape[0] + 1, acf.shape[0])]
plt.figure(figsize=(10,10))
plt.stem(locs, acf_sym, linefmt='-', markerfmt='ro', basefmt='-')
plt.fill_between(range(-acf.shape[0], acf.shape[0]+1), -conf_int, conf_int, alpha=0.2)

plt.xlabel("Lags")
plt.xticks(locs)
plt.ylabel("Magnitude")
plt.title("Autocorrelation Function of Forecast Errors")
plt.tight_layout()
plt.show()

#%%
q_value = calc_q_value(forecast_errors, 20)
print(f"Q-value of MLR method: {q_value :.4f}")

#%%
print(f"Variance of Forecast Errors = {np.var(forecast_errors):.4f}")
print(f"Mean of Forecast Errors = {np.mean(forecast_errors):.4f}")

#%%
plot_acf_pacf(forecast_errors, 20)