import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('Housing.csv')

# Encode all object (categorical) columns
for col in df.select_dtypes(include='object').columns:
    df[col] = pd.factorize(df[col])[0]

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
plt.plot(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
plt.xlabel(X.columns[0])
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
