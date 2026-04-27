import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# 1. Load and Preprocess data
df = pd.read_csv("master_final_cleaned.csv")
df = df.head(500_000)
df.columns = df.columns.str.strip()

# Feature Engineering
df['UNIT_PRICE'] = np.where(df['QUANTITY'] > 0, df['SALES_VALUE'] / df['QUANTITY'], 0)
df['DISCOUNT_RATE'] = np.where((df['SALES_VALUE'] + df['RETAIL_DISC']) > 0, 
                               df['RETAIL_DISC'] / (df['SALES_VALUE'] + df['RETAIL_DISC']), 0)

target = "QUANTITY"
drop_cols = ["household_key", "final_sales_value", "SALES_VALUE", "BASKET_ID", "PRODUCT_ID", "TRANS_TIME"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Encode categorical variables
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df = df.fillna(0)

# Define Features and Target
X = df.drop(target, axis=1)
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data (Deep Learning rất cần Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Build Wide and Deep Model
input_layer = Input(shape=(X_train_scaled.shape[1],))

# Wide Path (Linear)
wide_path = Dense(16, activation='relu')(input_layer)

# Deep Path (Neural Network)
deep_path = Dense(64, activation='relu')(input_layer)
deep_path = Dense(32, activation='relu')(deep_path)
deep_path = Dense(16, activation='relu')(deep_path)

# Combine Wide and Deep
combined = Concatenate()([wide_path, deep_path])
output_layer = Dense(1)(combined)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. Train the model
print("Training Wide and Deep model... (This may take a minute)")
model.fit(X_train_scaled, y_train, epochs=20, batch_size=128, verbose=1, validation_split=0.1)

# 4. Make Predictions and Calculate Metrics
y_pred = model.predict(X_test_scaled).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
epsilon = 1e-10
smape = np.mean(2 * np.abs(y_test.to_numpy() - y_pred) / (np.abs(y_test.to_numpy()) + np.abs(y_pred) + epsilon)) * 100

print("\n--- WIDE AND DEEP RESULTS ---")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print(f"sMAPE: {smape}%")

# 5. Save results to 'results3'
os.makedirs("results3", exist_ok=True)

# Save metrics
metrics = {"MSE": mse, "RMSE": rmse, "R2": r2, "SMAPE": smape}
pd.DataFrame([metrics]).to_csv("results3/wd_metrics.csv", index=False)

# Save predictions
pd.DataFrame({"y_test": y_test.to_numpy(), "y_pred": y_pred}).to_csv("results3/wd_predictions.csv", index=False)

# Lưu ý: Wide and Deep (Neural Networks) không có Feature Importance trực tiếp như XGBoost/LightGBM.
# Ta sẽ tạo một file thông báo thay thế.
with open("results3/wd_feature_note.txt", "w") as f:
    f.write("Neural Networks do not provide direct feature importance like tree-based models.")

print("\nResults saved to 'results3' folder.")