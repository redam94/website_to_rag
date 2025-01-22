To run a regression model in Python, follow these general steps:

1. **Import Necessary Packages**: Typically, you would use libraries such as `numpy`, `pandas`, `scikit-learn`, or `statsmodels`.
```python
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
```
2. **Prepare Your Data**: Load your dataset, and split it into independent variables (features) and the dependent variable (target).
 
```python
# Example with Pandas DataFrame
df = pd.read_csv('data.csv')  # Load your dataset
X = df[['feature1', 'feature2']]  # Features
y = df['target']  # Target
```

3. **Create and Fit the Model**: Using `scikit-learn`, you create an instance of the model and fit it to your data.
```python
model = LinearRegression()  # Create the model instance
model.fit(X, y)  # Fit the model
```
4. **Evaluate the Model**: You can check the model\'s performance using metrics like the coefficient of determination (R²).
```python
r_sq = model.score(X, y)
print(f"R²: {r_sq}")
```
5. **Make Predictions**: After fitting your model, you can use it to make predictions on new data.
```python
y_pred = model.predict(X_new)  # Predict on new data
```
6. **Advanced Use**: For more detailed statistics, you can use `statsmodels` which provides a comprehensive summary of the regression results.
```python
X_with_const = sm.add_constant(X)  # Add constant for intercept
results = sm.OLS(y, X_with_const).fit()  # Fit the model
 print(results.summary())  # Display model summary
```

These steps apply for both simple and multiple linear regression models in Python.