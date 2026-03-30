# OpenCode Prompt — STEP 7.1
# File: 07_hyperparameter_tuning.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

X = pd.read_csv('X_features.csv')
Y_summary = pd.read_csv('Y_release_summary.csv')
y_burst = Y_summary['Burst Release 24h percent'].values

rmse_scorer = make_scorer(lambda y, yp: -np.sqrt(mean_squared_error(y, yp)),
                           greater_is_better=True)

# --- RIDGE: ALPHA SEARCH ---
ridge_pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
ridge_grid = GridSearchCV(ridge_pipe,
                           param_grid={'ridge__alpha': [0.01, 0.1, 1, 5, 10, 50, 100, 500]},
                           cv=LeaveOneOut(), scoring=rmse_scorer)
ridge_grid.fit(X, y_burst)
print(f"Best Ridge a: {ridge_grid.best_params_['ridge__alpha']} | LOO RMSE: {-ridge_grid.best_score_:.2f}%")

# --- RANDOM FOREST: DEPTH + LEAVES ---
# IMPORTANT: For n=25, constrain depth hard (max_depth <= 5, min_samples_leaf >= 2)
# otherwise RF memorizes training data entirely
rf_pipe = Pipeline([('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(n_estimators=300, random_state=42))])
rf_grid = GridSearchCV(rf_pipe,
                        param_grid={
                            'rf__max_depth': [2, 3, 4],
                            'rf__min_samples_leaf': [2, 3],
                            'rf__max_features': [0.5, 1.0],
                        },
                        cv=LeaveOneOut(), scoring=rmse_scorer)
rf_grid.fit(X, y_burst)
print(f"Best RF params: {rf_grid.best_params_} | LOO RMSE: {-rf_grid.best_score_:.2f}%")

# --- XGBOOST: REGULARIZED SEARCH ---
xgb_pipe = Pipeline([('scaler', StandardScaler()),
                     ('xgb', xgb.XGBRegressor(verbosity=0, random_state=42))])
xgb_grid = GridSearchCV(xgb_pipe,
                         param_grid={
                             'xgb__max_depth': [2, 3],
                             'xgb__learning_rate': [0.05, 0.1],
                             'xgb__n_estimators': [100, 200],
                             'xgb__reg_alpha': [0, 1.0],
                         },
                         cv=LeaveOneOut(), scoring=rmse_scorer)
xgb_grid.fit(X, y_burst)
print(f"Best XGB params: {xgb_grid.best_params_} | LOO RMSE: {-xgb_grid.best_score_:.2f}%")

# --- ENSEMBLE: WEIGHTED AVERAGE ---
# Combine best models -- validated with LOO
loo = LeaveOneOut()

y_pred_ridge = np.zeros(len(y_burst))
y_pred_rf = np.zeros(len(y_burst))
y_pred_xgb = np.zeros(len(y_burst))

best_ridge = ridge_grid.best_estimator_
best_rf = rf_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_

for train_idx, test_idx in loo.split(X):
    best_ridge.fit(X.iloc[train_idx], y_burst[train_idx])
    best_rf.fit(X.iloc[train_idx], y_burst[train_idx])
    best_xgb.fit(X.iloc[train_idx], y_burst[train_idx])

    y_pred_ridge[test_idx] = best_ridge.predict(X.iloc[test_idx])
    y_pred_rf[test_idx] = best_rf.predict(X.iloc[test_idx])
    y_pred_xgb[test_idx] = best_xgb.predict(X.iloc[test_idx])

# Equal ensemble
y_ensemble = (y_pred_ridge + y_pred_rf + y_pred_xgb) / 3
rmse_ens = np.sqrt(mean_squared_error(y_burst, y_ensemble))
r2_ens = r2_score(y_burst, y_ensemble)
print(f"\nEnsemble LOO-CV | RMSE: {rmse_ens:.2f}% | R2: {r2_ens:.3f}")
print("[OK] Hyperparameter tuning complete.")
