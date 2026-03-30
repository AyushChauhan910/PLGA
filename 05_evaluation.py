# OpenCode Prompt — STEP 5.1
# File: 05_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

X = pd.read_csv('X_features.csv')
Y_summary = pd.read_csv('Y_release_summary.csv')
targets = ['Burst Release 24h percent', 'T50 days', 'T90 days', 'Max Release percent']

def full_evaluation(model, X, y, name, target):
    loo = LeaveOneOut()
    scaler = StandardScaler()
    y_pred_all = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_te = scaler.transform(X.iloc[test_idx])
        model.fit(X_tr, y[train_idx])
        y_pred_all[test_idx] = model.predict(X_te)

    return {
        'Model': name, 'Target': target,
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_all)),
        'MAE': mean_absolute_error(y, y_pred_all),
        'R2': r2_score(y, y_pred_all),
    }

models = {
    'Ridge (a=10)':  Ridge(alpha=10.0),
    'RF':            RandomForestRegressor(n_estimators=200, max_depth=4,
                                           min_samples_leaf=3, random_state=42),
    'XGBoost':       xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, verbosity=0, random_state=42),
}

all_results = []
for target in targets:
    y = Y_summary[target].values
    for name, model in models.items():
        r = full_evaluation(model, X, y, name, target)
        all_results.append(r)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
results_df.to_csv('05_evaluation_results.csv', index=False)

# --- LEARNING CURVE ANALYSIS (Overfitting Check) ---
# With n=25, train RMSE << test RMSE indicates severe overfitting
# Expected pattern: RF will overfit; Ridge most stable

from sklearn.model_selection import learning_curve

y_burst = Y_summary['Burst Release 24h percent'].values
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, model) in zip(axes, models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_s, y_burst,
        train_sizes=np.linspace(0.3, 1.0, 8),
        cv=LeaveOneOut(), scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    ax.plot(train_sizes, -train_scores.mean(axis=1), 'b-o', label='Train RMSE')
    ax.plot(train_sizes, -val_scores.mean(axis=1), 'r-o', label='LOO-CV RMSE')
    ax.fill_between(train_sizes,
                    -train_scores.mean(axis=1) - train_scores.std(axis=1),
                    -train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='b')
    ax.set_title(f'Learning Curve - {name}')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('RMSE (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_learning_curves.png', dpi=150)
plt.close()
print("[OK] Evaluation complete.")
