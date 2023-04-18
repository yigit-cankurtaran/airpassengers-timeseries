import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

#  import the flights dataset
flights = sns.load_dataset('flights')

#  convert the dataset to a time series
flights.index = pd.DatetimeIndex(flights['year'].astype(
    str) + '-' + flights['month'].astype(str), freq='MS')
flights = flights['passengers']

#  feature engineering
passengers_per_month = flights.resample('MS').mean()
passengers_per_flight = passengers_per_month.fillna(
    passengers_per_month.mean())
passengers_per_flight = passengers_per_month / \
    flights.groupby(flights.index.year).count()
passengers_per_flight = passengers_per_flight.fillna(
    passengers_per_flight.mean())


#  combine the features into a single DataFrame
exog = pd.concat([passengers_per_month, passengers_per_flight], axis=1)
exog = exog.fillna(exog.mean())
exog = exog.replace([np.inf, -np.inf], exog.mean())


#  split the dataset into train and test
train, test = train_test_split(flights, test_size=0.2, shuffle=False)

#  fit the model
model_fit = SARIMAX(train, exog=exog.loc[train.index], order=(
    1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

#  make predictions
exog_forecast = exog.loc[test.index]
predictions = model_fit.predict(
    start=len(train), end=len(train)+len(test)-1, dynamic=False, exog=exog_forecast)

#  calculate the error
error = mean_squared_error(test, predictions)
print(f'Error: {error}')

#  plot the results
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()

# Graph residuals as a heatmap
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(kind='kde')
plt.show()
