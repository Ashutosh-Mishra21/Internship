### `Importing necessary libraries`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
### `Reading data`
data = pd.read_csv("../Dataset/50_Startups.csv")
data
### `Analysing the data`
data.info()
data.describe()
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create a figure and axis objects
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Profit vs R&D Spend
sns.scatterplot(y='R&D_Spend', x='Profit', data=data, ax=axes[0])
axes[0].set_title('Profit vs R&D Spend')

# Plot Profit vs Administration
sns.scatterplot(y='Administration', x='Profit', data=data, ax=axes[1])
axes[1].set_title('Profit vs Administration')

# Plot Profit vs Marketing Spend
sns.scatterplot(y='Marketing_Spend', x='Profit', data=data, ax=axes[2])
axes[2].set_title('Profit vs Marketing Spend')

# Adjust layout
plt.tight_layout()
plt.show()
print("Missing values before imputation:\n", data.isnull().sum())
### `Data Preprocessing and Splitting`
X = data.iloc[ : , 0:3]
Y = data.iloc[ : , -1]
X
Y
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(Y.values.reshape(-1, 1))
X_scaled
y_scaled
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=45)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}
### `Fitting different Models`
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
XGboostreg = ensemble.GradientBoostingRegressor(**params)
XGboostreg.fit(X_train, y_train.ravel())
decision_tree_reg = DecisionTreeRegressor(random_state=42)
decision_tree_reg.fit(X_train, y_train)
linear_reg_pred = linear_reg.predict(X_test)
gradient_reg_pred = XGboostreg.predict(X_test)
decision_tree_pred = decision_tree_reg.predict(X_test)
### `Calculating different Regression Metrics`
# Mean Squared Error (MSE)
linear_reg_mse = mean_squared_error(y_test, linear_reg_pred)
gradient_reg_mse = mean_squared_error(y_test, gradient_reg_pred)
decision_tree_mse = mean_squared_error(y_test, decision_tree_pred)

# Mean Absolute Error (MAE)
linear_reg_mae = mean_absolute_error(y_test, linear_reg_pred)
gradient_reg_mae = mean_absolute_error(y_test, gradient_reg_pred)
decision_tree_mae = mean_absolute_error(y_test, decision_tree_pred)

# R-squared (R2)
linear_reg_r2 = r2_score(y_test, linear_reg_pred)
gradient_reg_r2 = r2_score(y_test, gradient_reg_pred)
decision_tree_r2 = r2_score(y_test, decision_tree_pred)
### `Choosing best model according to Regression Metrics`
models = {'Multiple Linear Regression': {'MSE': linear_reg_mse, 'MAE': linear_reg_mae, 'R2': linear_reg_r2},
          'Gradient Boosting Regression': {'MSE': gradient_reg_mse, 'MAE': gradient_reg_mae, 'R2':gradient_reg_r2},
          'Decision Tree Regression': {'MSE': decision_tree_mse, 'MAE': decision_tree_mae, 'R2':decision_tree_r2}}
for model in models:
     print(f"Metrics of {model} model is: ", models[model])
best_model = min(models, key=lambda x: models[x]['MSE'])
print("Best model:", best_model)
print("Metrics for the best model:")
print(models[best_model])
# For Linear Regression
linear_coefficients = linear_reg.coef_  # Coefficients of the features
linear_intercept = linear_reg.intercept_[0]  # Intercept

print("Equation for the Multiple Linear Regression Model is:")
print(linear_intercept ,"+", linear_coefficients[0][0],"x1 +", linear_coefficients[0][1],"x2 +", linear_coefficients[0][2],"x3")