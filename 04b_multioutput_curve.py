# OpenCode Prompt — STEP 4.2
# File: 04b_multioutput_curve.py

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

X = pd.read_csv('X_features.csv')
Y_curve = pd.read_csv('Y_release_curve.csv')

time_labels = [0.25/24, 0.5/24, 1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 42, 56, 70, 84, 98, 112]

def evaluate_multioutput_loo(model, X, Y):
    loo = LeaveOneOut()
    scaler = StandardScaler()
    Y_pred = np.zeros_like(Y.values, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_te = scaler.transform(X.iloc[test_idx])
        model.fit(X_tr, Y.iloc[train_idx].values)
        Y_pred[test_idx] = model.predict(X_te)

    # Per-timepoint RMSE
    rmse_per_tp = [np.sqrt(mean_squared_error(Y.values[:, i], Y_pred[:, i]))
                   for i in range(Y.shape[1])]
    r2_per_tp = [r2_score(Y.values[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    overall_rmse = np.sqrt(mean_squared_error(Y.values, Y_pred))
    return rmse_per_tp, r2_per_tp, overall_rmse, Y_pred

mo_models = {
    'Ridge (Multi)':    MultiOutputRegressor(Ridge(alpha=10.0)),
    'RF (Multi)':       MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=4,
                                                                    min_samples_leaf=3, random_state=42)),
    'XGB (Multi)':      MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                                                learning_rate=0.05, subsample=0.8,
                                                                verbosity=0, random_state=42)),
}

mo_results = {}
for name, model in mo_models.items():
    rmse_tp, r2_tp, overall, y_pred = evaluate_multioutput_loo(model, X, Y_curve)
    mo_results[name] = {'rmse_per_tp': rmse_tp, 'r2_per_tp': r2_tp,
                         'overall_rmse': overall, 'y_pred': y_pred}
    print(f"{name:20s} | Overall RMSE: {overall:.2f}% | Mean R2: {np.mean(r2_tp):.3f}")

# Plot: RMSE across time points
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, res in mo_results.items():
    axes[0].plot(time_labels, res['rmse_per_tp'], marker='o', label=name, markersize=4)
axes[0].set_xscale('log')
axes[0].set_xlabel('Time (days, log scale)')
axes[0].set_ylabel('RMSE (%)')
axes[0].set_title('RMSE per Time Point (LOO-CV)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Best model predicted vs actual (overlay all 25 formulations)
best_model_name = min(mo_results, key=lambda x: mo_results[x]['overall_rmse'])
Y_pred_best = mo_results[best_model_name]['y_pred']

for i in range(len(Y_curve)):
    actual = Y_curve.values[i]
    pred = Y_pred_best[i]
    axes[1].plot(time_labels, actual, 'b-', alpha=0.4, lw=1)
    axes[1].plot(time_labels, pred, 'r--', alpha=0.4, lw=1)
axes[1].plot([], [], 'b-', label='Actual')
axes[1].plot([], [], 'r--', label='Predicted (LOO-CV)')
axes[1].set_xscale('log')
axes[1].set_xlabel('Time (days, log scale)')
axes[1].set_ylabel('Cumulative Release (%)')
axes[1].set_title(f'Predicted vs Actual Release Curves\n({best_model_name})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_multioutput_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Multi-output curve modeling complete.")
