# OpenCode Prompt — STEP 3.1
# File: 03_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_excel('PLGA dataset for ML.xlsx')
df['Species'] = df['Species'].str.strip()
df['T90 days'].fillna(df['T90 days'].median(), inplace=True)

# ======================================================================
# DOMAIN-INFORMED FEATURE ENGINEERING
# Each feature is justified by polymer science / pharmacokinetics
# ======================================================================

# --- FEATURE 1: Hydrophilicity Index ---
# WHY: Combines drug solubility (log-scaled) with LogP to quantify drug-polymer
# affinity. High solubility + low LogP -> drug migrates to aqueous phase quickly
# -> predicts burst release behavior
df['Hydrophilicity_Index'] = np.log1p(df['Drug Solubility mg per mL']) - df['Drug LogP']

# --- FEATURE 2: Degradation Rate Proxy ---
# WHY: LA:GA ratio is the primary determinant of PLGA degradation rate
# 50:50 -> fastest; 85:15 -> slowest. Encoded as numeric glycolide fraction.
la_ga_map = {'50:50': 50.0, '65:35': 65.0, '75:25': 75.0, '85:15': 85.0}
df['LA_ratio_num'] = df['Lactide Glycolide Ratio'].map(la_ga_map).fillna(65.0)
df['GA_fraction'] = (100 - df['LA_ratio_num']) / 100  # glycolide fraction

# --- FEATURE 3: Thermal Distance from Body Temperature ---
# WHY: Release rate depends on polymer chain mobility. When Tg >> 37C, polymer
# is glassy at body temperature -> diffusion is severely restricted.
# When Tg ~ 37C -> rubber-glass transition -> unpredictable release.
df['Tg_Offset'] = df['Polymer Tg C'] - 37.0  # positive = glassy; negative = rubbery

# --- FEATURE 4: Diffusion Barrier Indicator ---
# WHY: Particle size determines diffusion path length. Drug must travel from
# center to surface. Larger particles -> longer path -> slower burst + longer sustained phase
# PDI modifies the heterogeneity of this barrier.
df['Diffusion_Barrier'] = df['Particle Size um'] * (1 + df['PDI'])

# --- FEATURE 5: Polymer-Drug Mass Ratio (log-scale) ---
# WHY: Drug loading and polymer:drug ratio directly determine matrix tortuosity.
# Higher drug loading -> higher percolation probability -> faster release.
# Log-scale because effect is typically non-linear.
df['Log_PolymerDrug_Ratio'] = np.log1p(df['Polymer to Drug Ratio'])
df['Drug_Load_Squared'] = df['Drug Loading percent'] ** 2  # captures nonlinearity

# --- FEATURE 6: Hydrolytic Degradation Index ---
# WHY: Higher MW -> longer chains -> slower hydrolysis -> slower bulk erosion.
# Acid endcapping accelerates autocatalytic degradation (self-catalysis by COOH groups).
# Ester-capped polymers degrade more slowly and uniformly.
endcap_map = {'Acid': 1.5, 'Ester': 1.0}
df['Endcap_factor'] = df['Polymer Endcapping'].map(endcap_map).fillna(1.0)
df['Hydrolysis_Index'] = df['GA_fraction'] * df['Endcap_factor'] / np.log1p(df['Polymer MW kDa'])

# --- FEATURE 7: Encapsulation Efficiency x Drug Loading ---
# WHY: EE determines actual drug in particle vs nominal. High EE with high
# loading means dense drug packing -> complex release kinetics.
df['EE_DL_Interaction'] = (df['Entrapment Efficiency percent'] / 100) * df['Drug Loading percent']

# --- FEATURE 8: Polymer Molecular Weight x Viscosity ---
# WHY: Both MW and inherent viscosity correlate with chain entanglement density.
# Together they better predict chain diffusion resistance than either alone.
df['MW_Viscosity'] = df['Polymer MW kDa'] * df['Inherent Viscosity dL per g']

# --- FEATURE 9: Drug Size Normalized by Particle ---
# WHY: Drug molecular weight relative to particle size influences release mechanism.
# Small drug in large particle -> diffusion controlled. Large drug -> erosion controlled.
df['Drug_Particle_Size_Ratio'] = df['Drug MW Da'] / (df['Particle Size um'] * 1000)

# --- FEATURE 10: Zeta Potential Absolute (electrostatic stability) ---
# WHY: Absolute zeta potential indicates surface charge. Highly charged particles
# maintain dispersion better, affecting aggregation and actual release medium interaction.
df['Abs_Zeta'] = df['ZetaPotential mV'].abs()

# ======================================================================
# CATEGORICAL ENCODING
# ======================================================================

# Ordinal Encoding for features with natural order
df['Encapsulation_enc'] = LabelEncoder().fit_transform(df['Encapsulation Method'].fillna('Unknown'))
df['Drying_enc'] = LabelEncoder().fit_transform(df['Drying Method'].fillna('Unknown'))
df['Stabilizer_enc'] = LabelEncoder().fit_transform(df['Stabilizer Type'].fillna('Unknown'))
df['Admin_enc'] = LabelEncoder().fit_transform(df['Administration Route'].fillna('IM'))
df['Species_enc'] = LabelEncoder().fit_transform(df['Species'].fillna('Rabbit'))
df['Release_Method_enc'] = LabelEncoder().fit_transform(df['Release Test Method'].fillna('Unknown'))

# Drug category encoding -- map to continuous hydrophilicity scale
drug_cat_order = {
    'Small hydrophilic': 1, 'Hydrophilic antibiotic': 2, 'Hydrophilic salt': 3,
    'Large hydrophilic antibiotic': 4, 'Peptide hydrophilic': 5, 'Peptide': 6,
    'Ionizable acidic drug': 7, 'Moderately soluble': 7, 'Local anesthetic': 8,
    'Hydrophilic salt': 3, 'Moderately lipophilic steroid': 9, 'Steroid': 9,
    'Lipophilic base': 10, 'Poorly soluble base': 10, 'BCS II lipophilic': 11,
    'Large lipophilic': 12, 'Lipophilic immunosuppressant': 13,
    'Highly lipophilic': 14, 'Extremely lipophilic': 15
}
df['Drug_Cat_enc'] = df['Drug Category'].map(drug_cat_order).fillna(8)

# ======================================================================
# FINAL FEATURE SET DEFINITION
# ======================================================================

BASE_FEATURES = [
    # Drug properties
    'Drug MW Da', 'Drug LogP', 'Drug pKa', 'Drug Solubility mg per mL',
    # Polymer properties
    'Polymer MW kDa', 'Inherent Viscosity dL per g', 'Polymer Tg C',
    # Process parameters
    'Stabilizer Concentration percent', 'Drug Loading percent',
    'Entrapment Efficiency percent', 'Polymer to Drug Ratio',
    # Particle characteristics
    'Particle Size um', 'PDI', 'ZetaPotential mV',
    # Release conditions
    'Release Medium pH', 'Temperature C', 'Agitation RPM', 'Dose mg',
]

ENGINEERED_FEATURES = [
    'Hydrophilicity_Index', 'GA_fraction', 'Tg_Offset', 'Diffusion_Barrier',
    'Log_PolymerDrug_Ratio', 'Drug_Load_Squared', 'Hydrolysis_Index',
    'EE_DL_Interaction', 'MW_Viscosity', 'Drug_Particle_Size_Ratio', 'Abs_Zeta',
]

ENCODED_FEATURES = [
    'Encapsulation_enc', 'Drying_enc', 'Stabilizer_enc', 'Admin_enc',
    'Species_enc', 'Release_Method_enc', 'Drug_Cat_enc', 'Endcap_factor',
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES + ENCODED_FEATURES

TARGET_CURVE = ['Release D0.25 6h', 'Release D0.5 12h', 'Release D1', 'Release D2',
                'Release D3', 'Release D5', 'Release D7', 'Release D10', 'Release D14',
                'Release D21', 'Release D28', 'Release D35', 'Release D42',
                'Release D56', 'Release D70', 'Release D84', 'Release D98', 'Release D112']
TARGET_SUMMARY = ['Burst Release 24h percent', 'T50 days', 'T90 days', 'Max Release percent']

# Build X and Y matrices
X = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median(numeric_only=True))
Y_curve = df[TARGET_CURVE].fillna(df[TARGET_CURVE].median(numeric_only=True))
Y_summary = df[TARGET_SUMMARY].fillna(df[TARGET_SUMMARY].median(numeric_only=True))

print(f"Feature matrix X: {X.shape}")
print(f"Release curve Y: {Y_curve.shape}")
print(f"Summary targets Y: {Y_summary.shape}")
print(f"\nEngineered features sample stats:")
print(X[ENGINEERED_FEATURES].describe().round(3))

# Save preprocessed data
X.to_csv('X_features.csv', index=False)
Y_curve.to_csv('Y_release_curve.csv', index=False)
Y_summary.to_csv('Y_release_summary.csv', index=False)

print("\n[OK] Feature engineering complete. Saved: X_features.csv, Y_release_curve.csv, Y_release_summary.csv")
