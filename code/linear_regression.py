#%%
from utilities import *
from CONSTANTS import *
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

#%%
train_df = pd.read_csv('../data/currency_exchange_rates_train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df = pd.read_csv('../data/currency_exchange_rates_test.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])

#%%
X_train = train_df[INDEPENDENT_FEATURES]
y_train = train_df[DEPENDENT_FEATURES]
X_train.drop(columns=['Date'], inplace=True)

X_test = test_df[INDEPENDENT_FEATURES]
y_test = test_df[DEPENDENT_FEATURES]
X_test.drop(columns=['Date'], inplace=True)

#%%
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

#%%
U, S, V = np.linalg.svd(X_train)
print(f"Singular values: {np.round(S, 2)}")

#%%
condition_number = np.linalg.cond(X_train)
print(f"Condition number: {condition_number: .2f}")

#%%
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

#%%
cols = INDEPENDENT_FEATURES.copy()
cols.remove('Date')
cols.insert(0, 'Constant')

#%%
X_train_df = pd.DataFrame(X_train, columns=cols)
X_test_df = pd.DataFrame(X_test, columns=cols)
y_train_df = pd.DataFrame(y_train, columns=DEPENDENT_FEATURES)
y_test_df = pd.DataFrame(y_test, columns=DEPENDENT_FEATURES)

#%%
ols_model = sm.OLS(y_train_df, X_train_df).fit()
print(ols_model.summary())

#%%
X_train_df.drop(columns=['Thai Baht'], inplace=True)
ols_model2 = sm.OLS(y_train_df, X_train_df).fit()
print(ols_model2.summary())

#%%
X_train_df.drop(columns=['Constant'], inplace=True)
ols_model3 = sm.OLS(y_train_df, X_train_df).fit()
print(ols_model3.summary())

#%%
X_train_df = pd.DataFrame(X_train, columns=cols)
X_test_df = pd.DataFrame(X_test, columns=cols)
y_train_df = pd.DataFrame(y_train, columns=DEPENDENT_FEATURES)
y_test_df = pd.DataFrame(y_test, columns=DEPENDENT_FEATURES)

#%%
vifs = {}
for col in X_train_df.columns:
    ols = sm.OLS(X_train_df[col], X_train_df.drop(columns=[col])).fit()
    vif = 1 / (1 - ols.rsquared)
    vifs[col] = float(vif)

print(vifs)

#%%
X_train_df.drop(columns=['Singapore Dollar'], inplace=True)
ols_model4 = sm.OLS(y_train_df, X_train_df).fit()
print(ols_model4.summary())

#%%
vifs = {}
for col in X_train_df.columns:
    ols = sm.OLS(X_train_df[col], X_train_df.drop(columns=[col])).fit()
    vif = 1 / (1 - ols.rsquared)
    vifs[col] = float(vif)

print(vifs)

#%%
X_train_df.drop(columns=['Pakistani Rupee'], inplace=True)
ols_model5 = sm.OLS(y_train_df, X_train_df).fit()
print(ols_model5.summary())

#%%
y_preds = ols_model3.predict(X_test_df.drop(columns=['Thai Baht', 'Constant']))
y_pred_df = pd.DataFrame(y_preds, columns=DEPENDENT_FEATURES)

#%%
plt.figure()
plt.plot(range(y_train_df.shape[0]), scaler_y.inverse_transform(y_train_df[DEPENDENT_FEATURES]), label='Train')
plt.plot(range(y_train_df.shape[0], y_train_df.shape[0] + y_test_df.shape[0]), scaler_y.inverse_transform(y_test_df[DEPENDENT_FEATURES]), label='Test')
plt.plot(range(y_train_df.shape[0], y_train_df.shape[0] + y_test_df.shape[0]),scaler_y.inverse_transform(y_pred_df[DEPENDENT_FEATURES]), label='Prediction')
plt.legend(loc='best')
plt.title("Prediction of Indian Rupee using Multiple Linear Regression")
plt.xlabel("Index")
plt.ylabel("INR per 1US$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
forecast_errors = scaler_y.inverse_transform(y_test_df[DEPENDENT_FEATURES]) - scaler_y.inverse_transform(y_pred_df[DEPENDENT_FEATURES])

#%%
acf = calculate_acf(forecast_errors.flatten(), 21)
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
q_value = calc_q_value(forecast_errors.flatten(), 20)
print(f"Q-value of MLR method: {q_value :.4f}")

#%%
print(f"Variance of Forecast Errors = {np.var(forecast_errors):.4f}")
print(f"Mean of Forecast Errors = {np.mean(forecast_errors):.4f}")

#%%
plot_acf_pacf(forecast_errors, 20)