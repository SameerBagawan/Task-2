from google.colab import files
uploaded = files.upload()

import pandas as pd
dataset =pd.read_csv('Housing.csv')
dataset.head()

dataset.isnull().sum()

dataset.select_dtypes(include="object")

import sklearn.model_selection.train_test_split


from sklearn.model_selection import train_test_split
X = dataset.drop("price", axis=1)
y = dataset["price"]

# Convert categorical columns to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

X_tarin,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_tarin,y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE = mean_absolute_error(y_test, predict)

MSE = mean_squared_error(y_test,predict)
R2 = r2_score(y_test,predict)

print("Mean Absolute Error (MAE):", MAE)
print("Mean Absolute Error (MAE):", MSE)
print("R² Score:", R2)

predict =model.predict(X_test)

# Display coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print(coefficients)

print("\nIntercept:", model.intercept_)

import matplotlib.pyplot as plt
import seaborn as sns



import matplotlib.pyplot as plt


plt.scatter(y_test, predict, alpha=0.6)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")


plt.show()










