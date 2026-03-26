import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the cleaned dataset
df = pd.read_csv("cleaned_house_data.csv")
print(df.head())

# input and output variables
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


import matplotlib.pyplot as plt

# Plot Actual vs Predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

# Add reference line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")

plt.tight_layout()
plt.savefig("actual_vs_predicted_prices.png")
plt.close()

print("Regression plot saved successfully!")