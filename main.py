import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

#  import the flights dataset
flights = sns.load_dataset('flights')

#  convert the dataset to a time series
flights.index = pd.DatetimeIndex(flights['year'].astype(
    str) + '-' + flights['month'].astype(str), freq='MS')
flights = flights['passengers']


#  split the dataset into train and test
train, test = train_test_split(flights, test_size=0.2, shuffle=False)

#  fit the model
model_fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

#  make predictions
predictions = model_fit.predict(
    start=len(train), end=len(train)+len(test)-1, dynamic=False)

#  calculate the error
error = mean_squared_error(test, predictions)
print(f'Error: {error}')

#  plot the results
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()

#  Plot residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
