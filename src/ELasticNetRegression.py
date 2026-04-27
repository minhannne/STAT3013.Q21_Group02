import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# 1. Load and Preprocess data
df = pd.read_csv("master_final_cleaned.csv")
df = df.head(500_000)
df.columns = df.columns.str.strip()

# Create derived features: UNIT_PRICE and DISCOUNT_RATE
df['UNIT_PRICE'] = np.where(df['QUANTITY'] > 0, df['SALES_VALUE'] / df['QUANTITY'], 0)
df['DISCOUNT_RATE'] = np.where((df['SALES_VALUE'] + df['RETAIL_DISC']) > 0, 
                               df['RETAIL_DISC'] / (df['SALES_VALUE'] + df['RETAIL_DISC']), 0)

# Define target and drop irrelevant columns
target = "QUANTITY"
drop_cols = ["household_key", "final_sales_value", "SALES_VALUE", "BASKET_ID", "PRODUCT_ID", "TRANS_TIME"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Encode categorical variables using LabelEncoder
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle missing values
df = df.fillna(0)

# Define Features (X) and Target (y)
X = df.drop(target, axis=1)
y = df[target]

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (Mandatory for Elastic Net)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Build and Train Elastic Net Model
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Make Predictions and Calculate Metrics
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate sMAPE
epsilon = 1e-10
smape = np.mean(2 * np.abs(y_test.to_numpy() - y_pred) / (np.abs(y_test.to_numpy()) + np.abs(y_pred) + epsilon)) * 100

print("--- ELASTIC NET REGRESSION RESULTS ---")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
print(f"sMAPE: {smape}%")

# 6. Extract Feature Importance (based on absolute coefficients)
importance = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False)

# 7. Save results to 'results1' folder (Same as previous setup)
os.makedirs("results1", exist_ok=True)

# Save metrics
metrics = {"MSE": mse, "RMSE": rmse, "R2": r2, "SMAPE": smape}
pd.DataFrame([metrics]).to_csv("results1/en_metrics.csv", index=False)

# Save y_test and y_pred comparison
pd.DataFrame({"y_test": y_test.to_numpy(), "y_pred": y_pred}).to_csv("results1/en_predictions.csv", index=False)

# Save top features
importance.head(10).to_csv("results1/en_top_features.csv", header=["importance"], index=True)

