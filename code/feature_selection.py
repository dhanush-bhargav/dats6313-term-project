#%%
import os
os.chdir('code')
from utilities import *
from CONSTANTS import *
import seaborn as sns

#%%
train_df = pd.read_csv('../data/currency_exchange_rates_train.csv')
test_df = pd.read_csv('../data/currency_exchange_rates_test.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

#%%
train_df_final = train_df[ INDEPENDENT_FEATURES + DEPENDENT_FEATURES]
test_df_final = test_df[INDEPENDENT_FEATURES + DEPENDENT_FEATURES]

#%%
plt.figure(figsize=(10, 6))
plt.plot(train_df_final['Date'], train_df_final[DEPENDENT_FEATURES], label=DEPENDENT_FEATURES)
plt.title(DEPENDENT_FEATURES[0])
plt.xlabel("Index")
plt.ylabel(f"{DEPENDENT_FEATURES[0]} per one USD ($1)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%
correlation_mat = train_df_final.drop(columns=['Date']).corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_mat, annot=True, vmin = -1, vmax = 1, center=True)
plt.tight_layout()
plt.show()

#%%
rolling_mean_var = calculate_rolling_mean_var(train_df_final[DEPENDENT_FEATURES])
fig, ax = plt.subplots(2, 1)
ax[0].plot(rolling_mean_var['Samples'], rolling_mean_var['Rolling_Mean'])
ax[0].set_title(f'Rolling Mean - {DEPENDENT_FEATURES[0]}')
ax[0].set_ylabel('Magnitude')
ax[0].set_xlabel('Samples')
ax[1].plot(rolling_mean_var['Samples'], rolling_mean_var['Rolling_Var'])
ax[1].set_title(f'Rolling Variance - {DEPENDENT_FEATURES[0]}')
ax[1].set_ylabel('Magnitude')
ax[1].set_xlabel('Samples')
plt.tight_layout()
plt.show()

#%%
print("ADF Test Results:")
calculate_adf(train_df_final[DEPENDENT_FEATURES].to_numpy())
print("KPSS Test Results:")
calculate_kpss(train_df_final[DEPENDENT_FEATURES].to_numpy())

#%%
first_order_diff = calculate_non_seasonal_differential(train_df_final[DEPENDENT_FEATURES].to_numpy(), 1)
first_order_diff = pd.DataFrame({"Indian Rupee" : first_order_diff.flatten()})
plt.figure(figsize=(10, 6))
plt.plot(first_order_diff[DEPENDENT_FEATURES], label=DEPENDENT_FEATURES)
plt.title(DEPENDENT_FEATURES[0])
plt.xlabel("Index")
plt.ylabel(f"{DEPENDENT_FEATURES[0]} per one USD ($1)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%
rolling_mean_var = calculate_rolling_mean_var(first_order_diff[DEPENDENT_FEATURES])
fig, ax = plt.subplots(2, 1)
ax[0].plot(rolling_mean_var['Samples'], rolling_mean_var['Rolling_Mean'])
ax[0].set_title(f'Rolling Mean - {DEPENDENT_FEATURES[0]}')
ax[0].set_ylabel('Magnitude')
ax[0].set_xlabel('Samples')
ax[1].plot(rolling_mean_var['Samples'], rolling_mean_var['Rolling_Var'])
ax[1].set_title(f'Rolling Variance - {DEPENDENT_FEATURES[0]}')
ax[1].set_ylabel('Magnitude')
ax[1].set_xlabel('Samples')
plt.tight_layout()
plt.show()

#%%
print("ADF Test Results on first order difference:")
calculate_adf(first_order_diff[DEPENDENT_FEATURES].to_numpy())
print("KPSS Test Results on first order difference:")
calculate_kpss(first_order_diff[DEPENDENT_FEATURES].to_numpy())