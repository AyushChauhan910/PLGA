# OpenCode Prompt — STEP 2.1
# File: 01_data_profiling.py
# Run: python 01_data_profiling.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_excel('PLGA dataset for ML.xlsx')
print(f"Dataset shape: {df.shape}")

# ─── CLEAN KNOWN ISSUES ─────────────────────────────────────────────────────
# Fix trailing whitespace in Species
df['Species'] = df['Species'].str.strip()

# Drop non-informative columns
DROP_COLS = ['Formulation name', 'Remarks', 'Dosage Form']  # All microspheres
df.drop(columns=DROP_COLS, inplace=True)

# ─── DEFINE COLUMN GROUPS ───────────────────────────────────────────────────
DRUG_PROPS = ['Drug MW Da', 'Drug LogP', 'Drug pKa', 'Drug Solubility mg per mL']

POLYMER_PROPS = ['Lactide Glycolide Ratio', 'Polymer MW kDa', 'Inherent Viscosity dL per g',
                 'Polymer Endcapping', 'Polymer Tg C', 'Polymer Crystallinity']

PROCESS_PARAMS = ['Encapsulation Method', 'Solvent System', 'Stabilizer Type',
                  'Stabilizer Concentration percent', 'Drug Loading percent',
                  'Entrapment Efficiency percent', 'Polymer to Drug Ratio', 'Drying Method']

PARTICLE_CHARS = ['Particle Size um', 'PDI', 'ZetaPotential mV']

RELEASE_CONDITIONS = ['Release Test Method', 'Release Medium', 'Release Medium pH',
                      'Temperature C', 'Agitation RPM', 'Dose mg']

PK_PARAMS = ['Cmax', 'Tmax', 'AUC', 'Half Life', 'MRT']

RELEASE_SUMMARY = ['Burst Release 24h percent', 'Max Release percent',
                   'T50 days', 'T90 days', 'Total Release Duration days']

RELEASE_CURVE = ['Release D0.25 6h', 'Release D0.5 12h', 'Release D1', 'Release D2',
                 'Release D3', 'Release D5', 'Release D7', 'Release D10', 'Release D14',
                 'Release D21', 'Release D28', 'Release D35', 'Release D42',
                 'Release D56', 'Release D70', 'Release D84', 'Release D98', 'Release D112']

# ─── MISSING VALUE ANALYSIS ─────────────────────────────────────────────────
print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
print(missing[missing > 0])
# Impute T90 missing with median (1 value only)
df['T90 days'].fillna(df['T90 days'].median(), inplace=True)

# ─── STATISTICAL SUMMARY ────────────────────────────────────────────────────
print("\n=== NUMERICAL SUMMARY ===")
print(df[DRUG_PROPS + POLYMER_PROPS[1:] + PARTICLE_CHARS].describe().round(2))

# ─── DISTRIBUTION PLOTS ─────────────────────────────────────────────────────
key_num_cols = ['Drug LogP', 'Drug Solubility mg per mL', 'Polymer MW kDa',
                'Polymer Tg C', 'Particle Size um', 'Drug Loading percent',
                'Burst Release 24h percent', 'T50 days']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flatten(), key_num_cols):
    ax.hist(df[col].dropna(), bins=8, color='steelblue', edgecolor='white', alpha=0.85)
    # Overlay KDE
    col_data = df[col].dropna()
    kde_x = np.linspace(col_data.min(), col_data.max(), 100)
    kde = stats.gaussian_kde(col_data)
    ax2 = ax.twinx()
    ax2.plot(kde_x, kde(kde_x), color='crimson', lw=1.5)
    ax2.set_yticks([])
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('')
plt.suptitle('Feature Distributions (PLGA Dataset, n=25)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('01_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── CORRELATION HEATMAP (Numerical features only) ──────────────────────────
num_df = df[DRUG_PROPS + POLYMER_PROPS[1:] + PARTICLE_CHARS + RELEASE_SUMMARY].select_dtypes(include=np.number)
corr = num_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, annot_kws={'size': 7}, ax=ax)
ax.set_title('Pearson Correlation — Numerical Features vs Release Summary', fontsize=12)
plt.tight_layout()
plt.savefig('01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── RELEASE CURVE VISUALIZATION ────────────────────────────────────────────
time_points = [0.25/24, 0.5/24, 1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 42, 56, 70, 84, 98, 112]

fig, ax = plt.subplots(figsize=(12, 6))
for i, row in df.iterrows():
    release = row[RELEASE_CURVE].values
    drug = row['API Name']
    ax.plot(time_points, release, alpha=0.6, marker='o', markersize=3, lw=1.2, label=drug)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Cumulative Release (%)')
ax.set_title('All 25 PLGA Release Profiles')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_release_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[OK] Profiling complete. Outputs: 01_distributions.png, 01_correlation_heatmap.png, 01_release_curves.png")
