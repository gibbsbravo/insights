# Transaction Modelling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dateutil.relativedelta

np.random.seed(42)

df = pd.read_excel('data/OnlineRetailData.xlsx') #head=None
df.dropna(subset=['CustomerID'], axis=0, inplace=True)

# Drop duplicates
duplicate_rows = df.loc[df.duplicated(keep=False)]
df.drop_duplicates(inplace=True)
df.info()

#%%
def sample_feature(input_df, feature_name, proportion=0.20):
    """Sample based on feature"""
    unique_features = input_df[feature_name].unique()
    sample_features = np.random.choice(unique_features, int(len(unique_features) * proportion),
                                       replace=False)
    sampled_df = input_df.loc[input_df[feature_name].isin(sample_features)].reset_index(drop=True)
    return sampled_df

df = sample_feature(df, 'CustomerID')

#%%
# Extract date
df['Hour'] = df['InvoiceDate'].dt.hour
df['Weekday'] = df['InvoiceDate'].dt.weekday # Monday is 0, Sunday is 6
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year

# Calculate revenue
df['Revenue'] = df['Quantity'] * df['UnitPrice']

#%%
# Revenue by day
weekday_data = df.groupby(by='Weekday').agg(['mean', 'sum'])
plt.bar(weekday_data['Revenue']['sum'].index, weekday_data['Revenue']['sum']);

# Revenue by year
yearly_data = df.groupby(by='Year').agg(['mean', 'sum'])
plt.bar(yearly_data['Revenue']['sum'].index, yearly_data['Revenue']['sum']);

#%%

customer_data = df.groupby(by='CustomerID').agg(['min', 'max'])
tenure_per_customer = (customer_data['InvoiceDate']['max'] - customer_data['InvoiceDate']['min']).dt.days
plt.hist(tenure_per_customer)


#%% 

#end_month = customer_data['InvoiceDate']['max'].max() - dateutil.relativedelta.relativedelta(months=1)

customer_start_date = df.groupby(by='CustomerID')['InvoiceDate'].agg(['min'])

# The idea is to loop through each customer and aggregate features from their first invoice
df[(df['CustomerID'] == 15311.0) & (df['InvoiceDate'] == customer_join.loc[15311.0].item())]

for group_label, data in df[:1000].groupby(by='CustomerID'):
    print(data[data['InvoiceDate'] == customer_start_date.loc[group_label].item()])

one_time_customers = tenure_per_customer[tenure_per_customer == 0].index



