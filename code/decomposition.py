#%%
import os
os.chdir('code')
from utilities import *
from CONSTANTS import *

#%%
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
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_results = seasonal_decompose(data.to_numpy(), period=7)

residuals = seasonal_results.resid[np.isfinite(seasonal_results.resid)]
trend = seasonal_results.trend[np.isfinite(seasonal_results.trend)]
seasonality = seasonal_results.seasonal[np.isfinite(seasonal_results.seasonal)]
observed = seasonal_results.observed[np.isfinite(seasonal_results.observed)]

#%%
trend_strength = max(0, 1 - (np.var(residuals) / np.var(observed - seasonality)))
print(f'The strength of trend for this dataset is: {trend_strength * 100:.2f}%')

#%%
seasonality_strength = max(0, 1 - (np.var(residuals) / np.var(observed[3:-3] - trend)))
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
plt.legend()
plt.show()
