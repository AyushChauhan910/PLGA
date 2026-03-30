# OpenCode Prompt — STEP 4.1
# File: 04_model_training.py
# STRATEGY: LOO-CV is mandatory (n=25). Train all models on summary targets first.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# --- LOAD DATA ---
X = pd.read_csv('X_features.csv')
Y_summary = pd.read_csv('Y_release_summary.csv')
Y_curve = pd.read_csv('Y_release_curve.csv')

# NOTE: For single-target analysis, focus on Burst Release first (most data-rich signal)
y_burst = Y_summary['Burst Release 24h percent'].values
y_t50 = Y_summary['T50 days'].values

# --- EVALUATION FUNCTION ---
def evaluate_model_loo(model, X, y, name):
    """LOO-CV evaluation -- correct for n=25"""
    loo = LeaveOneOut()
    scaler = StandardScaler()

    y_pred_all = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        y_pred_all[test_idx] = model.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    mae = mean_absolute_error(y, y_pred_all)
    r2 = r2_score(y, y_pred_all)
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred_all}

# --- MODELS ---
# Gaussian Process kernel: Matern (smooth, handles small data well) + White (noise)
gp_kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)

models = {
    'Ridge (a=1.0)':     Ridge(alpha=1.0),
    'Ridge (a=10.0)':    Ridge(alpha=10.0),
    'Lasso':             Lasso(alpha=0.1, max_iter=5000),
    'ElasticNet':        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=4,
                                               min_samples_leaf=3, random_state=42),
    'XGBoost':           xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.7,
                                           reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                                           verbosity=0),
    'Gaussian Process':  GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-6,
                                                   normalize_y=True, n_restarts_optimizer=5),
}

# --- EVALUATE ON BURST RELEASE ---
print("=== TARGET: Burst Release 24h (%) ===")
results_burst = []
for name, model in models.items():
    r = evaluate_model_loo(model, X, y_burst, name)
    results_burst.append({k: v for k, v in r.items() if k != 'y_pred'})
    print(f"  {name:25s} | RMSE: {r['RMSE']:6.2f} | MAE: {r['MAE']:6.2f} | R2: {r['R2']:6.3f}")

# --- EVALUATE ON T50 ---
print("\n=== TARGET: T50 (days) ===")
results_t50 = []
for name, model in models.items():
    r = evaluate_model_loo(model, X, y_t50, name)
    results_t50.append({k: v for k, v in r.items() if k != 'y_pred'})
    print(f"  {name:25s} | RMSE: {r['RMSE']:6.2f} | MAE: {r['MAE']:6.2f} | R2: {r['R2']:6.3f}")

# Save results
pd.DataFrame(results_burst).to_csv('results_burst.csv', index=False)
pd.DataFrame(results_t50).to_csv('results_t50.csv', index=False)
print("\n[OK] Model evaluation complete.")
