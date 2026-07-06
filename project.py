import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# STEP 1: DATA GENERATION WITH REALISTIC E-COMMERCE LOGIC
# =====================================================================
print("=== Step 1: Generating Realistic E-commerce Dataset ===")
np.random.seed(42)
n_samples = 1000

# Generating fundamental e-commerce features
price = np.random.uniform(15.0, 499.0, size=n_samples).round(2)
past_purchases = np.random.randint(1, 30, size=n_samples)
discount = np.random.uniform(0.05, 0.40, size=n_samples).round(2)

# Rating generation logic linked to features to ensure positive R2 scores
rating = 2.0 + (price * 0.002) + (past_purchases * 0.05) + (discount * 2.0) + np.random.normal(0, 0.3, n_samples)
rating = np.clip(rating.round(), 1, 5) 

data = {
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], size=n_samples),
    'Price': price,
    'User_Past_Purchases': past_purchases,
    'Avg_Discount_Offered': discount,
    'User_Rating': rating
}
df = pd.DataFrame(data)
print(df.head())

# =====================================================================
# STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING
# =====================================================================
print("\n=== Step 2: Preprocessing and Splitting Data ===")
# Converting categorical categories into numerical vectors
df_encoded = pd.get_dummies(df, columns=['Product_Category'], drop_first=True)

X = df_encoded.drop(['User_Rating'], axis=1)
y = df_encoded['User_Rating']

# Data Splitting: 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling for uniform data distributions
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# STEP 3: HYPERPARAMETER OPTIMIZATION USING GRIDSEARCHCV
# =====================================================================
print("\n=== Step 3: Optimizing Random Forest Hyperparameters ===")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}
rf_base = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_
print(f"Optimal Hyperparameters: {grid_search.best_params_}")

# =====================================================================
# STEP 4: MULTI-ALGORITHM EVALUATION AND COMPARISON
# =====================================================================
print("\n=== Step 4: Final Multi-Algorithm Evaluation ===")
models = {
    "Ridge Regression (Baseline)": Ridge(),
    "Tuned Random Forest": best_rf_model
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    results[name] = {"RMSE": rmse, "MAE": mae, "R2 Score": r2}

# Compiling data metrics into a structured dataframe
performance_df = pd.DataFrame(results).T
print("\n--- PERFORMANCE COMPARISON REPORT ---")
print(performance_df.round(4))

# =====================================================================
# STEP 5: VISUALIZATION & PERFORMANCE ANALYSIS
# =====================================================================
print("\n=== Step 5: Generating Performance Comparison Plots ===")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: RMSE Matrix Check
sns.barplot(x=performance_df.index, y=performance_df['RMSE'], ax=axes[0], palette="Blues_d")
axes[0].set_title('RMSE (Lower Error is Better)')
axes[0].set_ylabel('Error Value')

# Plot 2: R2 Score Matrix Check
sns.barplot(x=performance_df.index, y=performance_df['R2 Score'], ax=axes[1], palette="Greens_d")
axes[1].set_title('R2 Score (Higher Accuracy is Better)')
axes[1].set_ylabel('R2 Score Value')

plt.tight_layout()

# Saving the comparison visualization graph to project workspaces
plt.savefig('model_performance_comparison.png', dpi=300)
print("Execution Complete. Graph saved as 'model_performance_comparison.png'.")
plt.show()