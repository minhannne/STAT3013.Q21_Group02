import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
import torch
import warnings

warnings.filterwarnings('ignore')

# 1. Load và Tiền xử lý (Y chang code trước)
df = pd.read_csv("master_final_cleaned.csv").head(500_000)
df.columns = df.columns.str.strip()
df['UNIT_PRICE'] = np.where(df['QUANTITY'] > 0, df['SALES_VALUE'] / df['QUANTITY'], 0)
df['DISCOUNT_RATE'] = np.where((df['SALES_VALUE'] + df['RETAIL_DISC']) > 0, 
                               df['RETAIL_DISC'] / (df['SALES_VALUE'] + df['RETAIL_DISC']), 0)
target = "QUANTITY"
drop_cols = ["household_key", "final_sales_value", "SALES_VALUE", "BASKET_ID", "PRODUCT_ID", "TRANS_TIME"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df = df.fillna(0)

X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# --- HUẤN LUYỆN 4 MÔ HÌNH ---
print("Đang huấn luyện để lấy sai số (Residuals)...")

# 1. ElasticNet
en = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42).fit(X_train_sc, y_train)
res_en = y_test - en.predict(X_test_sc)

# 2. LightGBM
lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42).fit(X_train, y_train)
res_lgbm = y_test - lgbm.predict(X_test)

# 3. TabNet
X_tab = X_train_sc.astype(np.float32)
y_tab = y_train.values.reshape(-1, 1).astype(np.float32)
tn = TabNetRegressor(n_d=16, n_a=16, n_steps=3, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2))
tn.fit(X_tab, y_tab, max_epochs=5, batch_size=1024, virtual_batch_size=128)
res_tn = y_test.values.flatten() - tn.predict(X_test_sc.astype(np.float32)).flatten()

# 4. Wide & Deep (20 Epochs)
input_l = Input(shape=(X_train_sc.shape[1],))
w = Dense(16, activation='relu')(input_l)
d = Dense(64, activation='relu')(input_l)
d = Dense(32, activation='relu')(d)
d = Dense(16, activation='relu')(d)
c = Concatenate()([w, d])
out = Dense(1)(c)
wd = Model(inputs=input_l, outputs=out)
wd.compile(optimizer='adam', loss='mse')
wd.fit(X_train_sc, y_train, epochs=20, batch_size=128, verbose=0)
res_wd = y_test.values.flatten() - wd.predict(X_test_sc).flatten()

# --- VẼ BOXPLOT GỘP ---
# Tạo DataFrame để chứa sai số của tất cả các mô hình
residual_data = pd.DataFrame({
    'ElasticNet': res_en,
    'LightGBM': res_lgbm,
    'TabNet': res_tn,
    'Wide & Deep': res_wd
})

plt.figure(figsize=(12, 8))
sns.boxplot(data=residual_data, palette="Set2", showfliers=False) # showfliers=False để hình dễ nhìn hơn, không bị rối bởi điểm ngoại lai

plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
plt.title('Comparison of Prediction Residuals (Errors) across Models', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Residual Value (Actual - Predicted)', fontsize=14)
plt.xlabel('Models', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)

plt.show()