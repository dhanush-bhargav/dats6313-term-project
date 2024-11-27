#%%
import os
os.chdir('code')
from utilities import *

#%%
raw_data = pd.read_csv("../data/currency_exchange_rates.csv")

#%%
raw_data.isna().sum()

#%%
## Since Many columns have less than 5000 non-null values, we need to drop the invalid columns.
## First step is to keep only the columns which have more than 5000 non-null values. And then take the tail to have latest data.

cols = raw_data.columns
valid_cols = []
min_size = raw_data.shape[0]
for col in cols:
    if raw_data.shape[0] - raw_data[col].isna().sum() > 5000:
        min_size = min(min_size, raw_data.shape[0] - raw_data[col].isna().sum())
        valid_cols.append(col)

cleaned_data = raw_data[valid_cols].tail(min_size)

#%%
cleaned_data.isna().sum()

#%%
## Since there are still some missing values, these are filled using forward fill since it is time series data.

cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
cleaned_data = cleaned_data.ffill()

#%%
cleaned_data.isna().sum()

## Now there are no missing values in any of the columns.

#%%
cleaned_data.to_csv("../data/currency_exchange_rates_cleaned.csv")

#%%
for col in ["Indian Rupee"]:
    plt.figure(figsize=(10, 6))
    plt.plot(cleaned_data['Date'],cleaned_data[col], label=col)
    plt.title(col)
    plt.xlabel("Date")
    plt.ylabel(f"{col} per one USD ($1)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

#%%
plot_acf_pacf(cleaned_data['Indian Rupee'], 50)

#%%
train_df = cleaned_data.head(int(0.8 * cleaned_data.shape[0]))
test_df = cleaned_data.tail(int(0.2 * cleaned_data.shape[0]))
train_df.to_csv("../data/currency_exchange_rates_train.csv", index=False)
test_df.to_csv("../data/currency_exchange_rates_test.csv", index=False)
