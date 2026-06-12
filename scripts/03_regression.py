import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    print("Run: pip install statsmodels")
    exit()

print("=" * 55)
print("STAGE 3 — Feature Engineering & OLS Regression")
print("=" * 55)

# ── Load matched dataset ───────────────────────────────────
df = pd.read_csv('data/property/ncr_aqi_matched_merged.csv')
print(f"\nInput: {df.shape[0]} rows × {df.shape[1]} columns")

# ═══════════════════════════════════════════════════════════
# PART A — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

# ── 1. Log-transform target ────────────────────────────────
df['log_price_sqft'] = np.log(df['price_per_sqft'])

# ── 2. AQI scaling ────────────────────────────────────────
# Divide by 100 so coefficient = price change per 100-pt AQI
df['aqi_100'] = df['annual_aqi'] / 100

# ── 3. Size: use log_area only ────────────────────────────
# log_bhk and log_bathrooms are collinear with log_area
# (bigger flats have more rooms) — drop them to fix VIF
df['log_area'] = np.log(df['area_sqft'].clip(lower=1))

# ── 4. Furnishing score ───────────────────────────────────
furnishing_map = {
    'Unfurnished':    0,
    'Semi-Furnished': 1,
    'Furnished':      2
}
df['furnishing_score'] = df['furnishing'].map(furnishing_map).fillna(1)

# ── 5. Transaction: New=1, Resale=0 ──────────────────────
df['is_new'] = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

# ── 6. Drop is_ready — check variance first ───────────────
ready_counts = df['status'].value_counts()
print(f"\nStatus breakdown:")
print(ready_counts.to_string())

# Only include is_ready if it has meaningful variance
# (at least 5% in the minority class)
df['is_ready'] = (df['status'].str.lower().str.strip() == 'ready to move').astype(int)
ready_pct = df['is_ready'].mean()
print(f"\nReady to Move: {ready_pct:.1%} of properties")
use_ready = 0.05 < ready_pct < 0.95
if not use_ready:
    print("  → Dropping is_ready (near-constant, causes high VIF)")
else:
    print("  → Keeping is_ready (sufficient variance)")

# ── 7. Balcony ────────────────────────────────────────────
# Keep as-is (already numeric, no VIF issue)

# ── 8. Force numeric ──────────────────────────────────────
base_controls = ['log_area', 'furnishing_score', 'is_new']
if use_ready:
    base_controls.append('is_ready')

all_cols = ['aqi_100', 'log_price_sqft'] + base_controls
for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\nFeatures engineered:")
print(f"  log_price_sqft : {df['log_price_sqft'].min():.2f} — {df['log_price_sqft'].max():.2f}")
print(f"  aqi_100        : {df['aqi_100'].min():.2f} — {df['aqi_100'].max():.2f}")
print(f"  log_area       : {df['log_area'].min():.2f} — {df['log_area'].max():.2f}")

# ═══════════════════════════════════════════════════════════
# PART B — VIF CHECK BEFORE REGRESSION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("VIF Check — before regression")
print("=" * 55)

vif_features = ['aqi_100'] + base_controls
sub_vif = df[vif_features].dropna().apply(pd.to_numeric, errors='coerce').dropna()
vif_data = pd.DataFrame({
    'feature': vif_features,
    'VIF': [variance_inflation_factor(sub_vif.values, i)
            for i in range(sub_vif.shape[1])]
}).sort_values('VIF', ascending=False)
print(vif_data.round(2).to_string(index=False))

high_vif = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
if high_vif:
    print(f"\nWARNING: High VIF features: {high_vif}")
    print("Consider dropping these before final submission")
else:
    print("\n✓ All VIF values acceptable (< 10)")

# ═══════════════════════════════════════════════════════════
# PART C — OLS REGRESSION (two models only)
# ═══════════════════════════════════════════════════════════

# Note: Model 3 with city FE is excluded because city dummies
# absorb station-level AQI variation (sign flips to positive),
# making joint identification impossible. Disclosed as limitation.

models_spec = {
    'Model 1 — AQI only (baseline)':
        ['aqi_100'],
    'Model 2 — AQI + property controls (main)':
        ['aqi_100'] + base_controls,
}

results = {}

print("\n" + "=" * 55)
print("OLS Regression Results")
print("=" * 55)

for model_name, features in models_spec.items():
    cols_needed = ['log_price_sqft'] + features
    sub = df[cols_needed].dropna()
    sub = sub.apply(pd.to_numeric, errors='coerce').dropna()

    X = sm.add_constant(sub[features].astype(float))
    y = sub['log_price_sqft'].astype(float)

    model = sm.OLS(y, X).fit(cov_type='HC3')
    results[model_name] = model

    aqi_coef  = model.params['aqi_100']
    aqi_pval  = model.pvalues['aqi_100']
    aqi_ci_lo = model.conf_int().loc['aqi_100', 0]
    aqi_ci_hi = model.conf_int().loc['aqi_100', 1]
    pct_change = (np.exp(aqi_coef) - 1) * 100

    print(f"\n{'─'*50}")
    print(f"{model_name}")
    print(f"{'─'*50}")
    print(f"  N observations : {len(sub):,}")
    print(f"  R²             : {model.rsquared:.4f}")
    print(f"  Adj R²         : {model.rsquared_adj:.4f}")
    print(f"\n  AQI coefficient (per 100-point increase):")
    print(f"    β             = {aqi_coef:.4f}")
    print(f"    95% CI        = [{aqi_ci_lo:.4f}, {aqi_ci_hi:.4f}]")
    print(f"    p-value       = {aqi_pval:.4f} "
          f"{'***' if aqi_pval<0.001 else '**' if aqi_pval<0.01 else '*' if aqi_pval<0.05 else '(ns)'}")
    print(f"    Price effect  = {pct_change:+.1f}% per 100-pt AQI rise")

# ── Full Model 2 coefficient table ────────────────────────
print("\n" + "=" * 55)
print("Full coefficient table — Model 2")
print("=" * 55)
m2 = results['Model 2 — AQI + property controls (main)']
coef_table = pd.DataFrame({
    'coef':    m2.params.round(4),
    'std_err': m2.bse.round(4),
    'p_value': m2.pvalues.round(4),
    'sig':     m2.pvalues.apply(
        lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')
})
print(coef_table.to_string())

# ═══════════════════════════════════════════════════════════
# PART D — POLICY SIMULATION (corrected)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("Policy Simulation")
print("=" * 55)

# Use main model coefficient
aqi_beta = m2.params['aqi_100']  # negative number (higher AQI = lower price)

city_avg_aqi = df.groupby('city')['annual_aqi'].mean()
print("\nCity average AQI (2021):")
print(city_avg_aqi.round(1).to_string())

ncr_avg = df['annual_aqi'].mean()
print(f"\nNCR overall average AQI: {ncr_avg:.1f}")
print(f"AQI beta (Model 2):      {aqi_beta:.4f}")

# For properties ABOVE NCR average AQI:
# Their AQI *decreases* by (annual_aqi - ncr_avg) if they reach the average
# Since beta is negative: price_change = exp(beta * delta_aqi/100) - 1
# delta_aqi is negative (improvement), so price_change is positive
df['aqi_delta']       = df['annual_aqi'] - ncr_avg   # positive for dirty zones
df['aqi_delta_100']   = df['aqi_delta'] / 100

# Price uplift: if AQI improves (decreases) to NCR average
# Change in log price = beta * (-delta_aqi/100)  [negative delta = improvement]
# Only applies to properties above NCR average (delta > 0)
df['pct_uplift']      = np.where(
    df['aqi_delta'] > 0,
    (np.exp(-aqi_beta * df['aqi_delta_100']) - 1),  # double negative = positive
    0.0
)
df['price_uplift_rs'] = df['price_total'] * df['pct_uplift']

total_uplift_cr = df['price_uplift_rs'].sum() / 1e7
props_affected  = (df['aqi_delta'] > 0).sum()

print(f"\nScenario: Above-average AQI zones improve to NCR average ({ncr_avg:.0f})")
print(f"  Properties affected  : {props_affected:,} / {len(df):,}")
print(f"  Total wealth unlocked: ₹{total_uplift_cr:,.0f} crore")
print(f"  Avg uplift/property  : ₹{df.loc[df['aqi_delta']>0, 'price_uplift_rs'].mean():,.0f}")

print("\nUplift by city:")
city_sim = df.groupby('city').agg(
    props        = ('price_total', 'count'),
    avg_aqi      = ('annual_aqi', 'mean'),
    total_uplift = ('price_uplift_rs', 'sum')
)
city_sim['uplift_cr'] = (city_sim['total_uplift'] / 1e7).round(1)
city_sim['avg_aqi']   = city_sim['avg_aqi'].round(1)
print(city_sim[['props', 'avg_aqi', 'uplift_cr']].to_string())

# ── Save final dataset ────────────────────────────────────
df.to_csv('data/property/ncr_final_merged.csv', index=False)
print(f"\n✓ Saved → data/property/ncr_final.csv")
print(f"  {len(df)} rows × {len(df.columns)} columns")

