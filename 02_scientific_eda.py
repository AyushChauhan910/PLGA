# OpenCode Prompt — STEP 2.2
# File: 02_scientific_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('PLGA dataset for ML.xlsx')
df['Species'] = df['Species'].str.strip()

# ─── LA:GA RATIO vs BURST RELEASE ───────────────────────────────────────────
# SCIENCE: Higher GA content → more hydrophilic → faster water penetration → faster degradation
# 50:50 degrades in weeks; 75:25 takes months; 85:15 may take years
# Expected: Higher GA content (lower LA:GA ratio) → Higher burst + faster T50

la_ga_map = {'50:50': 50, '65:35': 65, '75:25': 75, '85:15': 85}
df['LA_ratio_num'] = df['Lactide Glycolide Ratio'].map(la_ga_map)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(df['LA_ratio_num'], df['Burst Release 24h percent'],
                c=df['Drug LogP'], cmap='RdYlGn', s=80, edgecolors='k', linewidths=0.5)
axes[0].set_xlabel('Lactide % in LA:GA Ratio')
axes[0].set_ylabel('Burst Release at 24h (%)')
axes[0].set_title('LA:GA Ratio vs Burst Release\n(color = Drug LogP)')

# ─── Tg vs T50 ─────────────────────────────────────────────────────────────
# SCIENCE: Tg (glass transition temp) determines chain mobility at 37°C (body temp)
# Tg > 37°C → glassy state → slower diffusion → slower release
# Tg < 37°C → rubbery state → faster diffusion
axes[1].scatter(df['Polymer Tg C'], df['T50 days'],
                c=df['Polymer MW kDa'], cmap='Blues', s=80, edgecolors='k', linewidths=0.5)
axes[1].axvline(37, color='red', linestyle='--', lw=1.5, label='Body Temp (37°C)')
axes[1].set_xlabel('Polymer Tg (°C)')
axes[1].set_ylabel('T50 (days)')
axes[1].set_title('Polymer Tg vs T50\n(color = Polymer MW, red line = 37°C)')
axes[1].legend()

# ─── Drug LogP vs Burst Release ─────────────────────────────────────────────
# SCIENCE: Hydrophilic drugs (low LogP) have poor interaction with hydrophobic PLGA matrix
# → They tend to migrate to particle surface → Higher burst release
# Lipophilic drugs (high LogP) are entrapped in the polymer matrix → Lower burst
axes[2].scatter(df['Drug LogP'], df['Burst Release 24h percent'],
                c='steelblue', s=80, edgecolors='k', linewidths=0.5)
z = np.polyfit(df['Drug LogP'].dropna(), df['Burst Release 24h percent'].dropna(), 1)
p = np.poly1d(z)
x_line = np.linspace(df['Drug LogP'].min(), df['Drug LogP'].max(), 100)
axes[2].plot(x_line, p(x_line), 'r--', lw=1.5, label=f'Trend')
axes[2].set_xlabel('Drug LogP')
axes[2].set_ylabel('Burst Release at 24h (%)')
axes[2].set_title('Drug LogP vs Burst Release\n(lipophilic -> lower burst expected)')

plt.tight_layout()
plt.savefig('02_scientific_eda.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Endcapping Effect ───────────────────────────────────────────────────────
# SCIENCE: Acid-endcapped (free carboxylic end) → autocatalytic degradation
# Ester-endcapped → slower, more linear release; less acidic microenvironment
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='Polymer Endcapping', y='Burst Release 24h percent',
            palette='Set2', ax=ax)
ax.set_title('Endcapping Type vs Burst Release\n(Acid-endcapped -> autocatalytic degradation -> higher burst expected)')
plt.tight_layout()
plt.savefig('02_endcapping_effect.png', dpi=150)
plt.close()

print("[OK] Scientific EDA complete.")
