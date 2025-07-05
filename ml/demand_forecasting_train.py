import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os

# Path Configs
DATA_PATH = '../data/mock_walmart_pricing_dataset.csv'
MODEL_SAVE_PATH = '../backend/app/models/demand_forecast_model.pkl'

# Load Dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Categorical Features to Encode
cat_features = [
    'category', 'brand', 'price_reason_code',
    'user_segment', 'preferred_category', 'price_test_group'
]

# Encode Categorical Columns
print("Encoding categorical features...")
le = LabelEncoder()
for col in cat_features:
    df[col] = le.fit_transform(df[col])

# Features and Target Definition
features = [
    'category', 'brand', 'base_price', 'current_price',
    'perishable', 'current_stock', 'max_capacity', 'reorder_point',
    'restock_lead_time', 'promotion_applied', 'is_holiday',
    'forecast_confidence', 'average_basket_size', 'preferred_category',
    'price_reason_code', 'user_segment', 'price_test_group'
]

target = 'units_sold'

X = df[features]
y = df[target]

# Train-Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBM Dataset Creation
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# LightGBM Parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1
}

# Model Training
print("Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=100
)

# Model Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.2f}")

# Save Trained Model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
pickle.dump(model, open(MODEL_SAVE_PATH, 'wb'))
print(f"Model saved to {MODEL_SAVE_PATH}")
