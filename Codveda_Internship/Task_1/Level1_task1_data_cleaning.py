import pandas as pd

# Load dataset
df = pd.read_csv("house Prediction Data Set.csv", header=None, sep="\s+")

# Preview data
print(df.head())

# Check missing values and row count
print("\nMissing values:")
print(df.isnull().sum())
print("\nDataset shape:", df.shape)
# Check for duplicates
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
# Check data types
print("\nData types:")
print(df.dtypes)
# Assign column names
df.columns = [
"CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
"DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"
]
print(df.head())
# save as cleaned dataset
df.to_csv("cleaned_house_data.csv", index=False)

print("\nCleaned dataset saved successfully!")