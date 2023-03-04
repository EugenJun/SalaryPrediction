import os

import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data and get variables where the correlation coefficient is greater than 0.2 in the correlation matrix
data = pd.read_csv('../Data/data.csv')

# split data,  and predict salary
x_train, x_test, y_train, y_test = train_test_split(data.drop(['age', 'experience', "salary"], axis=1), data["salary"],
                                                    test_size=0.3, random_state=100)
# train model
model = LinearRegression()
model.fit(x_train, y_train)

# predict salary and replace 1) negatives with zeros; 2) negatives with median from y train
predicted_salary = pd.DataFrame(model.predict(x_test))
predicted_salary_with_negatives_as_zeros = predicted_salary.where(lambda x: x > 0, 0)
predicted_salary_with_negatives_as_median = predicted_salary.where(lambda x: x > 0, y_train.median())

# find and print best data_results
mape_results = [mape(y_test, predicted_salary_with_negatives_as_zeros),
                mape(y_test, predicted_salary_with_negatives_as_median)]

print(round(min(mape_results), 5))
