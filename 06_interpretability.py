# OpenCode Prompt — STEP 6.1
# File: 06_interpretability.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

X = pd.read_csv('X_features.csv')
Y_summary = pd.read_csv('Y_release_summary.csv')
y_burst = Y_summary['Burst Release 24h percent'].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Train on full dataset for SHAP (interpretation, not prediction)
rf = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=2, random_state=42)
rf.fit(X_s, y_burst)

# --- SHAP ANALYSIS ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_s)

# Beeswarm plot
shap.summary_plot(shap_values, X, feature_names=X.columns.tolist(),
                  show=False, max_display=20)
plt.title('SHAP Values - Burst Release Prediction\n(higher |SHAP| = stronger influence)')
plt.tight_layout()
plt.savefig('06_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# Bar plot (mean |SHAP|)
shap.summary_plot(shap_values, X, feature_names=X.columns.tolist(),
                  plot_type='bar', show=False, max_display=15)
plt.title('Feature Importance - Mean |SHAP| (Burst Release)')
plt.tight_layout()
plt.savefig('06_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# --- PHARMACEUTICAL INTERPRETATION TABLE ---
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

feat_imp['Pharmaceutical_Meaning'] = feat_imp['Feature'].map({
    'Drug LogP':              'Lipophilicity -> drug-polymer affinity -> burst modulation',
    'Hydrophilicity_Index':   'Engineered: drug-PLGA phase partition predictor',
    'GA_fraction':            'Glycolide % -> hydrolysis rate -> degradation speed',
    'Tg_Offset':              'Polymer chain mobility at 37C -> diffusion rate',
    'Diffusion_Barrier':      'Particle size x PDI -> effective diffusion path length',
    'Particle Size um':       'Physical barrier for drug escape -> burst control',
    'Drug Loading percent':   'Drug density in matrix -> percolation threshold',
    'Hydrolysis_Index':       'Engineered: combined degradation susceptibility score',
    'Polymer MW kDa':         'Chain length -> matrix integrity during degradation',
    'EE_DL_Interaction':      'Actual drug packing density in microsphere',
    'Polymer Tg C':           'Raw Tg -> chain flexibility at body temperature',
    'MW_Viscosity':           'Entanglement density -> diffusion resistance',
})

print("\n=== TOP 12 FEATURES - PHARMACEUTICAL INTERPRETATION ===")
print(feat_imp.head(12).to_string(index=False))
feat_imp.to_csv('06_feature_importance.csv', index=False)

print("\n[OK] SHAP analysis complete. Outputs: 06_shap_beeswarm.png, 06_shap_bar.png")
