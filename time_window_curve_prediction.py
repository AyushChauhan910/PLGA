# File: time_window_curve_prediction.py
# Run: python time_window_curve_prediction.py
# Requires: X_features.csv and Y_release_curve.csv from Phase 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

X = pd.read_csv('X_features.csv')
Y_full = pd.read_csv('Y_release_curve.csv')

# All 18 column names exactly as they appear in Y_release_curve.csv
ALL_COLS = [
    'Release D0.25 6h',   # ~0.01 days
    'Release D0.5 12h',   # ~0.02 days
    'Release D1',         # 1 day
    'Release D2',         # 2 days
    'Release D3',         # 3 days
    'Release D5',         # 5 days
    'Release D7',         # 7 days
    'Release D10',        # 10 days
    'Release D14',        # 14 days
    'Release D21',        # 21 days
    'Release D28',        # 28 days
    'Release D35',        # 35 days
    'Release D42',        # 42 days
    'Release D56',        # 56 days
    'Release D70',        # 70 days
    'Release D84',        # 84 days
    'Release D98',        # 98 days
    'Release D112',       # 112 days
]

# Actual time values in days (for plotting x-axis)
ALL_TIMES = [
    0.25/24, 0.5/24, 1, 2, 3, 5, 7, 10, 14,
    21, 28, 35, 42, 56, 70, 84, 98, 112
]

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: DEFINE THE THREE TIME WINDOWS
# ═══════════════════════════════════════════════════════════════════════════

WINDOWS = {
    'Window 1: D0.25h → D15 days': {
        'cols':  ALL_COLS[:9],       # D0.25h to D14
        'times': ALL_TIMES[:9],
        'color': '#2196F3',          # blue
        'label': 'D0.25h–D15'
    },
    'Window 2: D0.25h → D30 days': {
        'cols':  ALL_COLS[:11],      # D0.25h to D28
        'times': ALL_TIMES[:11],
        'color': '#FF9800',          # orange
        'label': 'D0.25h–D30'
    },
    'Window 3: D0.25h → D60 days': {
        'cols':  ALL_COLS[:14],      # D0.25h to D56
        'times': ALL_TIMES[:14],
        'color': '#4CAF50',          # green
        'label': 'D0.25h–D60'
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: DEFINE MODELS
# ═══════════════════════════════════════════════════════════════════════════

def get_models():
    """Returns a fresh set of models each time (avoids state leakage between windows)."""
    return {
        'Ridge': MultiOutputRegressor(
            Ridge(alpha=10.0)
        ),
        'Random Forest': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                max_features=0.5,
                random_state=42
            )
        ),
        'XGBoost': MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=200,
                max_depth=2,
                learning_rate=0.1,
                subsample=0.8,
                reg_alpha=1.0,
                verbosity=0,
                random_state=42
            )
        ),
    }

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: LOO-CV EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_window_loo(model, X, Y):
    """
    Runs full LOO-CV for a multi-output model on a given Y matrix.
    Returns:
        Y_pred     : (n_samples, n_outputs) predicted values
        rmse_total : single overall RMSE across all samples + timepoints
        rmse_per_tp: list of RMSE per time point
        r2_per_tp  : list of R² per time point
        r2_mean    : mean R² across all time points
    """
    loo = LeaveOneOut()
    scaler = StandardScaler()
    Y_pred = np.zeros_like(Y.values, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_test  = scaler.transform(X.iloc[test_idx])
        model.fit(X_train, Y.iloc[train_idx].values)
        Y_pred[test_idx] = model.predict(X_test)

    # Clip predictions to [0, 100] — release % must be in this range
    Y_pred = np.clip(Y_pred, 0, 100)

    # Overall RMSE (treats all time points equally)
    rmse_total = np.sqrt(mean_squared_error(Y.values, Y_pred))

    # Per time-point metrics
    n_tp = Y.shape[1]
    rmse_per_tp = [
        np.sqrt(mean_squared_error(Y.values[:, i], Y_pred[:, i]))
        for i in range(n_tp)
    ]
    r2_per_tp = [
        r2_score(Y.values[:, i], Y_pred[:, i])
        for i in range(n_tp)
    ]
    r2_mean = np.mean(r2_per_tp)

    return Y_pred, rmse_total, rmse_per_tp, r2_per_tp, r2_mean

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: RUN ALL WINDOWS × ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("TIME-WINDOW CURVE PREDICTION — LOO-CV RESULTS")
print("=" * 65)

# Storage for all results
all_results = {}   # all_results[window_name][model_name] = dict of metrics
summary_rows = []  # for final comparison table

for win_name, win_cfg in WINDOWS.items():
    print(f"\n{'─'*65}")
    print(f"  {win_name}  ({len(win_cfg['cols'])} time points)")
    print(f"{'─'*65}")

    Y_win = Y_full[win_cfg['cols']].fillna(
        Y_full[win_cfg['cols']].median(numeric_only=True)
    )
    all_results[win_name] = {}

    for model_name, model in get_models().items():
        Y_pred, rmse, rmse_tp, r2_tp, r2_mean = evaluate_window_loo(
            model, X, Y_win
        )

        all_results[win_name][model_name] = {
            'Y_pred':    Y_pred,
            'rmse':      rmse,
            'rmse_tp':   rmse_tp,
            'r2_tp':     r2_tp,
            'r2_mean':   r2_mean,
            'times':     win_cfg['times'],
            'cols':      win_cfg['cols'],
            'color':     win_cfg['color'],
        }

        print(f"  {model_name:15s} | RMSE: {rmse:5.2f}%  | Mean R²: {r2_mean:.3f}")

        summary_rows.append({
            'Window':    win_cfg['label'],
            'N timepoints': len(win_cfg['cols']),
            'Model':     model_name,
            'RMSE (%)':  round(rmse, 2),
            'Mean R²':   round(r2_mean, 3),
        })

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: PRINT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 65)
print("FULL COMPARISON TABLE")
print("=" * 65)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv('window_results_summary.csv', index=False)
print("\nSaved: window_results_summary.csv")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: PLOT 1 — PREDICTED VS ACTUAL CURVES (per window, best model)
# Each window gets its own row. Columns = 5 sampled formulations.
# ═══════════════════════════════════════════════════════════════════════════

# Pick best model per window (lowest RMSE)
def best_model_for_window(win_name):
    models = all_results[win_name]
    return min(models, key=lambda m: models[m]['rmse'])

# Pick 5 formulations spread across the dataset to show diversity
sample_idx = [0, 5, 10, 15, 20]

fig, axes = plt.subplots(3, 5, figsize=(20, 13))
fig.suptitle(
    'Predicted vs Actual Release Curves — Best Model per Window (LOO-CV)',
    fontsize=14, fontweight='bold', y=1.01
)

for row_idx, (win_name, win_cfg) in enumerate(WINDOWS.items()):
    best_m = best_model_for_window(win_name)
    res = all_results[win_name][best_m]
    Y_win = Y_full[win_cfg['cols']].fillna(
        Y_full[win_cfg['cols']].median(numeric_only=True)
    )

    for col_idx, form_idx in enumerate(sample_idx):
        ax = axes[row_idx, col_idx]

        actual = Y_win.values[form_idx]
        pred   = res['Y_pred'][form_idx]
        times  = win_cfg['times']

        ax.plot(times, actual, 'o-',
                color='#333333', lw=1.8, markersize=5, label='Actual')
        ax.plot(times, pred, 's--',
                color=win_cfg['color'], lw=1.8, markersize=5, label=f'Pred ({best_m})')

        # Shade the error band
        ax.fill_between(times,
                        np.minimum(actual, pred),
                        np.maximum(actual, pred),
                        alpha=0.15, color=win_cfg['color'])

        ax.set_ylim(-5, 105)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Titles and labels
        if row_idx == 0:
            ax.set_title(f'Formulation #{form_idx+1}', fontsize=10, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(f'{win_cfg["label"]}\nRelease (%)', fontsize=9)
        if row_idx == 2:
            ax.set_xlabel('Time (days, log)', fontsize=9)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

        # Annotate RMSE for this formulation
        form_rmse = np.sqrt(mean_squared_error(actual, pred))
        ax.text(0.97, 0.05, f'RMSE={form_rmse:.1f}%',
                transform=ax.transAxes, fontsize=7.5,
                ha='right', color=win_cfg['color'])

plt.tight_layout()
plt.savefig('plot1_predicted_vs_actual_per_window.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plot1_predicted_vs_actual_per_window.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: PLOT 2 — RMSE PER TIME POINT (all 3 windows, all 3 models)
# Shows WHERE in the release curve each model struggles most
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'RMSE per Time Point — All Windows × All Models (LOO-CV)',
    fontsize=13, fontweight='bold'
)

model_styles = {
    'Ridge':         {'ls': '-',  'marker': 'o'},
    'Random Forest': {'ls': '--', 'marker': 's'},
    'XGBoost':       {'ls': ':',  'marker': '^'},
}
model_colors = {
    'Ridge':         '#E53935',
    'Random Forest': '#1E88E5',
    'XGBoost':       '#43A047',
}

for ax_idx, (win_name, win_cfg) in enumerate(WINDOWS.items()):
    ax = axes[ax_idx]

    for model_name in ['Ridge', 'Random Forest', 'XGBoost']:
        res = all_results[win_name][model_name]
        style = model_styles[model_name]

        ax.plot(win_cfg['times'], res['rmse_tp'],
                color=model_colors[model_name],
                linestyle=style['ls'],
                marker=style['marker'],
                markersize=6, lw=2,
                label=f"{model_name} (mean R²={res['r2_mean']:.3f})")

    # Mark the worst time point
    best_m = best_model_for_window(win_name)
    worst_tp_idx = np.argmax(all_results[win_name][best_m]['rmse_tp'])
    worst_time   = win_cfg['times'][worst_tp_idx]
    worst_rmse   = all_results[win_name][best_m]['rmse_tp'][worst_tp_idx]
    ax.annotate(f'Hardest:\n{win_cfg["cols"][worst_tp_idx]}',
                xy=(worst_time, worst_rmse),
                xytext=(worst_time * 2, worst_rmse + 1.5),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    ax.set_xscale('log')
    ax.set_title(win_cfg['label'], fontweight='bold', fontsize=11)
    ax.set_xlabel('Time (days, log scale)')
    ax.set_ylabel('RMSE (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('plot2_rmse_per_timepoint.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot2_rmse_per_timepoint.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: PLOT 3 — WINDOW COMPARISON BAR CHART
# Overall RMSE and Mean R² grouped by model, across 3 windows
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Overall Performance vs Window Length — RMSE and Mean R²',
    fontsize=13, fontweight='bold'
)

win_labels  = [cfg['label']  for cfg in WINDOWS.values()]
win_colors  = [cfg['color']  for cfg in WINDOWS.values()]
model_names = ['Ridge', 'Random Forest', 'XGBoost']
x = np.arange(len(model_names))
bar_width = 0.25

for metric_idx, (metric_key, metric_label, ax) in enumerate([
    ('rmse',    'Overall RMSE (%)', axes[0]),
    ('r2_mean', 'Mean R²',          axes[1]),
]):
    for win_idx, (win_name, win_cfg) in enumerate(WINDOWS.items()):
        values = [
            all_results[win_name][m][metric_key]
            for m in model_names
        ]
        offset = (win_idx - 1) * bar_width
        bars = ax.bar(
            x + offset, values,
            width=bar_width,
            label=win_cfg['label'],
            color=win_cfg['color'],
            alpha=0.85,
            edgecolor='white', linewidth=0.8
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.05 if metric_key == 'r2_mean' else 0.1),
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(metric_label, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # Add reference line for R² plot
    if metric_key == 'r2_mean':
        ax.axhline(0.7, color='red', linestyle='--', lw=1.2, alpha=0.6,
                   label='R² = 0.70 threshold')

plt.tight_layout()
plt.savefig('plot3_window_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot3_window_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: PLOT 4 — ALL 25 FORMULATIONS OVERLAY (best model, all 3 windows)
# One subplot per window, showing all actual (blue) vs predicted (red dashed)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'All 25 Formulations — Actual vs Predicted (Best Model per Window)',
    fontsize=13, fontweight='bold'
)

for ax, (win_name, win_cfg) in zip(axes, WINDOWS.items()):
    best_m = best_model_for_window(win_name)
    res    = all_results[win_name][best_m]
    Y_win  = Y_full[win_cfg['cols']].fillna(
        Y_full[win_cfg['cols']].median(numeric_only=True)
    )
    times  = win_cfg['times']

    for i in range(len(Y_win)):
        actual = Y_win.values[i]
        pred   = res['Y_pred'][i]
        ax.plot(times, actual, '-', color='#1565C0', alpha=0.35, lw=1.2)
        ax.plot(times, pred,   '--', color=win_cfg['color'], alpha=0.35, lw=1.2)

    # Legend proxies
    ax.plot([], [], '-',  color='#1565C0',       lw=1.8, label='Actual (all 25)')
    ax.plot([], [], '--', color=win_cfg['color'], lw=1.8,
            label=f'Predicted — {best_m}')

    ax.set_xscale('log')
    ax.set_ylim(-5, 105)
    ax.set_xlabel('Time (days, log scale)', fontsize=10)
    ax.set_ylabel('Cumulative Release (%)', fontsize=10)
    ax.set_title(
        f"{win_cfg['label']}\n"
        f"Best: {best_m} | RMSE={res['rmse']:.2f}% | R²={res['r2_mean']:.3f}",
        fontsize=10, fontweight='bold'
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot4_all_formulations_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot4_all_formulations_overlay.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 11: PRINT FINAL SCIENTIFIC INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SCIENTIFIC INTERPRETATION")
print("=" * 65)

for win_name, win_cfg in WINDOWS.items():
    print(f"\n{win_cfg['label']}:")
    for model_name in model_names:
        res = all_results[win_name][model_name]
        worst_tp = np.argmax(res['rmse_tp'])
        print(f"  {model_name:15s} → RMSE={res['rmse']:.2f}%  "
              f"Mean R²={res['r2_mean']:.3f}  "
              f"Hardest timepoint: {win_cfg['cols'][worst_tp]} "
              f"(RMSE={res['rmse_tp'][worst_tp]:.2f}%)")

print("\n✅ All done. Output files:")
print("   window_results_summary.csv")
print("   plot1_predicted_vs_actual_per_window.png")
print("   plot2_rmse_per_timepoint.png")
print("   plot3_window_comparison.png")
print("   plot4_all_formulations_overlay.png")