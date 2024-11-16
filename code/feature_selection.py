import os

import pandas as pd
from statsmodels.compat.pandas import to_numpy

os.chdir('code')
from utilities import *
from CONSTANTS import *
import seaborn as sns

#%%
train_df = pd.read_csv('../data/currency_exchange_rates_train.csv')
test_df = pd.read_csv('../data/currency_exchange_rates_test.csv')

#%%
train_df_final = train_df[INDEPENDENT_FEATURES + DEPENDENT_FEATURES]
test_df_final = test_df[INDEPENDENT_FEATURES + DEPENDENT_FEATURES]

#%%
correlation_mat = train_df_final.corr()
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
calculate_adf(train_df_final[DEPENDENT_FEATURES])

#%%
# log_transform = calculate_log_transform(train_df_final[DEPENDENT_FEATURES].to_numpy())
first_order_diff = calculate_non_seasonal_differential(train_df_final[DEPENDENT_FEATURES].to_numpy(), 1)
plt.figure(figsize=(10, 10))
plt.plot(first_order_diff)
plt.show()

#%%
first_order_diff = pd.DataFrame({"Indian Rupee" : first_order_diff.flatten()})
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
calculate_adf(first_order_diff[DEPENDENT_FEATURES].to_numpy())

