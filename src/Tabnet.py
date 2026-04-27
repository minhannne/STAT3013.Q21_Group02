import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import torch

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

# --- KHU VỰC SỬA LỖI QUAN TRỌNG ---
# Ép kiểu toàn bộ X và y sang float32 để tránh lỗi "found object" khi tính Feature Importance
X = df.drop(target, axis=1).astype(np.float32).values
y = df[target].values.reshape(-1, 1).astype(np.float32)
# ----------------------------------

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build and Train TabNet
model = TabNetRegressor(
    n_d=16, n_a=16, n_steps=3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax'
)

print("Training TabNet model... (Epochs reduced to 10 for faster results)")
model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'valid'],
    eval_metric=['rmse'],
    max_epochs=10, # Đã giảm xuống 10 cho nhanh nè bạn
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# 3. Make Predictions and Calculate Metrics
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
epsilon = 1e-10
smape = np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + epsilon)) * 100

print("\n--- TABNET RESULTS ---")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print(f"sMAPE: {smape}%")

# 4. Feature Importance
# Lấy lại tên cột từ dataframe ban đầu
feature_names = df.drop(target, axis=1).columns
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances = feat_importances.sort_values(ascending=False)
print("\nTop 10 Features (TabNet):")
print(feat_importances.head(10))

# 5. Save results to 'results4'
os.makedirs("results4", exist_ok=True)
metrics = {"MSE": mse, "RMSE": rmse, "R2": r2, "SMAPE": smape}
pd.DataFrame([metrics]).to_csv("results4/tabnet_metrics.csv", index=False)
pd.DataFrame({"y_test": y_test.flatten(), "y_pred": y_pred.flatten()}).to_csv("results4/tabnet_predictions.csv", index=False)
feat_importances.to_csv("results4/tabnet_features.csv", header=["importance"], index=True)

print("\nResults saved to 'results4' folder.")