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

# Tắt các cảnh báo để màn hình hiển thị sạch sẽ hơn
warnings.filterwarnings('ignore')

# 1. Tải và Tiền xử lý dữ liệu (Giữ nguyên logic từ code gốc của bạn)
df = pd.read_csv("master_final_cleaned.csv").head(500_000)
df.columns = df.columns.str.strip()

# Tạo các đặc trưng UNIT_PRICE và DISCOUNT_RATE
df['UNIT_PRICE'] = np.where(df['QUANTITY'] > 0, df['SALES_VALUE'] / df['QUANTITY'], 0)
df['DISCOUNT_RATE'] = np.where((df['SALES_VALUE'] + df['RETAIL_DISC']) > 0, 
                               df['RETAIL_DISC'] / (df['SALES_VALUE'] + df['RETAIL_DISC']), 0)

target = "QUANTITY"
drop_cols = ["household_key", "final_sales_value", "SALES_VALUE", "BASKET_ID", "PRODUCT_ID", "TRANS_TIME"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Mã hóa các biến phân loại
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df = df.fillna(0)

# Định nghĩa X và y
X = df.drop(target, axis=1)
y = df[target]

# Chia tập dữ liệu (test_size=0.2, random_state=42 để đảm bảo tính nhất quán)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (Cần thiết cho ElasticNet và Neural Networks)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# --- HUẤN LUYỆN CÁC MÔ HÌNH ---
print("Đang huấn luyện các mô hình, vui lòng đợi...")

# 1. ElasticNet
en = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42).fit(X_train_sc, y_train)
y_en = en.predict(X_test_sc)

# 2. LightGBM
lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42).fit(X_train, y_train)
y_lgbm = lgbm.predict(X_test)

# 3. TabNet (Giữ nguyên cấu trúc hiện tại)
X_tab = X_train_sc.astype(np.float32)
y_tab = y_train.values.reshape(-1, 1).astype(np.float32)
tn = TabNetRegressor(n_d=16, n_a=16, n_steps=3, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2))
tn.fit(X_tab, y_tab, max_epochs=5, batch_size=1024, virtual_batch_size=128)
y_tn = tn.predict(X_test_sc.astype(np.float32)).flatten()

# 4. Wide & Deep (Tăng lên 20 epochs như code ban đầu của bạn)
input_l = Input(shape=(X_train_sc.shape[1],))
# Nhánh Wide
wide_path = Dense(16, activation='relu')(input_l)
# Nhánh Deep
deep_path = Dense(64, activation='relu')(input_l)
deep_path = Dense(32, activation='relu')(deep_path)
deep_path = Dense(16, activation='relu')(deep_path)
# Kết hợp
combined = Concatenate()([wide_path, deep_path])
output_layer = Dense(1)(combined)

wd_model = Model(inputs=input_l, outputs=output_layer)
wd_model.compile(optimizer='adam', loss='mse')
wd_model.fit(X_train_sc, y_train, epochs=20, batch_size=128, verbose=0)
y_wd = wd_model.predict(X_test_sc).flatten()

# --- VẼ BIỂU ĐỒ TỔNG HỢP (CHỈNH SỬA KHOẢNG CÁCH) ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Comparison of Model Forecasting Performance', fontsize=22, fontweight='bold')

models = [
    ('ElasticNet', y_en, 'royalblue', axes[0, 0]),
    ('LightGBM', y_lgbm, 'seagreen', axes[0, 1]),
    ('TabNet', y_tn, 'darkorange', axes[1, 0]),
    ('Wide & Deep', y_wd, 'mediumpurple', axes[1, 1])
]

for name, pred, color, ax in models:
    sns.scatterplot(x=y_test, y=pred, alpha=0.4, color=color, ax=ax)
    # Đường tham chiếu Perfect Prediction
    line_val = [y_test.min(), y_test.max()]
    ax.plot(line_val, line_val, 'r--', lw=2.5, label='Perfect Prediction')
    
    ax.set_title(f'Model: {name}', fontsize=16, pad=20)
    ax.set_xlabel('Actual Quantity', fontsize=12)
    ax.set_ylabel('Predicted Quantity', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

# Chỉnh sửa quan trọng: hspace giãn cách hàng, top dành chỗ cho tiêu đề tổng
plt.subplots_adjust(top=0.88, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.25)

plt.show()