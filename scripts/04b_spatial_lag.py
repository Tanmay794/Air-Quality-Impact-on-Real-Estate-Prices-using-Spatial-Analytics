import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import libpysal
from libpysal.weights import KNN
import spreg
from spreg import ML_Lag


print("=" * 55)
print("STAGE 4B — Spatial Lag Model")
print("=" * 55)

# ── Load and rebuild features ─────────────────────────────
df = pd.read_csv('data/property/ncr_aqi_matched_merged.csv')

df['log_price_sqft']   = np.log(df['price_per_sqft'])
df['aqi_100']          = df['annual_aqi'] / 100
df['log_area']         = np.log(df['area_sqft'].clip(lower=1))
df['furnishing_score'] = df['furnishing'].map({
    'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}).fillna(1)
df['is_new'] = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

features = ['aqi_100', 'log_area', 'furnishing_score', 'is_new']
cols     = ['log_price_sqft'] + features + ['latitude', 'longitude', 'nearest_station']

sub = df[cols].dropna()
sub = sub.copy()
for col in ['log_price_sqft'] + features:
    sub[col] = pd.to_numeric(sub[col], errors='coerce')
sub = sub.dropna()

print(f"\nObservations: {len(sub):,}")

# ── Build spatial weights at property level ───────────────
# Use k=10 nearest neighbours on property coordinates
# (duplicate coords handled by KNN — neighbours assigned
#  to closest unique points)
print("\nBuilding spatial weights matrix (k=10 KNN)...")
coords = sub[['latitude', 'longitude']].values

w = KNN.from_array(coords, k=10)
w.transform = 'r'  # row-standardise
print(f"Weights matrix: {w.n} observations, {w.mean_neighbors:.1f} avg neighbours")

# ── OLS benchmark (refit for comparison) ─────────────────
print("\n" + "=" * 55)
print("OLS (benchmark)")
print("=" * 55)

X_ols = sm.add_constant(sub[features].astype(float))
y_ols = sub['log_price_sqft'].astype(float)
ols   = sm.OLS(y_ols, X_ols).fit(cov_type='HC3')

aqi_ols   = ols.params['aqi_100']
pval_ols  = ols.pvalues['aqi_100']
pct_ols   = (np.exp(aqi_ols) - 1) * 100

print(f"  R²           : {ols.rsquared:.4f}")
print(f"  AQI β        : {aqi_ols:.4f}")
print(f"  AQI p-value  : {pval_ols:.4f}")
print(f"  Price effect : {pct_ols:+.1f}% per 100-pt AQI rise")

# ── Spatial Lag Model ─────────────────────────────────────
print("\n" + "=" * 55)
print("Spatial Lag Model (ML_Lag)")
print("=" * 55)
print("Fitting... (may take 30-60 seconds)")

y_arr = sub['log_price_sqft'].values.reshape(-1, 1)
X_arr = sub[features].values

# spreg requires numpy arrays, no constant needed (added internally)
lag_model = ML_Lag(
    y_arr,
    X_arr,
    w=w,
    name_y='log_price_sqft',
    name_x=features,
    name_ds='NCR Property Dataset'
)

print(lag_model.summary)

# ── Extract key results ───────────────────────────────────
print("\n" + "=" * 55)
print("Comparison: OLS vs Spatial Lag")
print("=" * 55)

# spreg stores betas as [const, x1, x2...] in betas array
# and rho (spatial lag coeff) separately
betas     = lag_model.betas.flatten()
std_errs  = np.sqrt(np.diag(lag_model.vm))
z_stats   = lag_model.z_stat

# Find AQI index (first feature after constant)
aqi_idx   = 1  # const=0, aqi_100=1
aqi_lag   = betas[aqi_idx]
aqi_se    = std_errs[aqi_idx]
aqi_z     = z_stats[aqi_idx][0]
aqi_pval  = z_stats[aqi_idx][1]
pct_lag   = (np.exp(aqi_lag) - 1) * 100
rho       = lag_model.rho

print(f"\n{'':30s} {'OLS':>12s} {'Spatial Lag':>12s}")
print(f"{'─'*55}")
print(f"{'AQI β (per 100-pt)':30s} {aqi_ols:>12.4f} {aqi_lag:>12.4f}")
print(f"{'AQI price effect':30s} {pct_ols:>11.1f}% {pct_lag:>11.1f}%")
print(f"{'AQI p-value':30s} {pval_ols:>12.4f} {aqi_pval:>12.4f}")
print(f"{'Spatial lag ρ (rho)':30s} {'—':>12s} {rho:>12.4f}")
print(f"{'R² / Pseudo-R²':30s} {ols.rsquared:>12.4f} {lag_model.pr2:>12.4f}")

print(f"\nSpatial lag coefficient ρ = {rho:.4f}")
if abs(rho) > 0.1 and lag_model.z_stat[-1][1] < 0.05:
    print("  → ρ is significant: spatial dependence confirmed")
    print("  → Spatial lag model is preferred specification")
    print("  → Use spatial lag AQI coefficient as main finding")
else:
    print("  → ρ is not significant: OLS remains valid")

# ── Interpretation ────────────────────────────────────────
print("\n" + "=" * 55)
print("Final Interpretation")
print("=" * 55)

main_beta = aqi_lag
main_pct  = pct_lag
main_se   = aqi_se

print(f"""
Main finding (Spatial Lag Model):
  A 100-point increase in annual AQI is associated with a
  {main_pct:+.1f}% change in property price per sqft
  (β = {main_beta:.4f}, SE = {main_se:.4f}, p = {aqi_pval:.4f})

Comparison with OLS:
  OLS estimate : {pct_ols:+.1f}% (β = {aqi_ols:.4f})
  Spatial Lag  : {main_pct:+.1f}% (β = {main_beta:.4f})
  Difference   : {abs(pct_lag - pct_ols):.1f} percentage points

The spatial lag model accounts for the positive spatial
autocorrelation identified in Stage 4 (Moran's I = 0.50,
p < 0.001). The AQI coefficient {"remains" if abs(pct_lag - pct_ols) < 5 else "shifts"} 
{"stable" if abs(pct_lag - pct_ols) < 5 else "materially"} across specifications,
{"supporting" if abs(pct_lag - pct_ols) < 5 else "qualifying"} the robustness of the pollution discount finding.
""")

# ── Policy simulation with spatial lag beta ───────────────
print("=" * 55)
print("Policy Simulation (Spatial Lag Model)")
print("=" * 55)

df2 = pd.read_csv('data/property/ncr_final_merged.csv')
ncr_avg = df2['annual_aqi'].mean()

df2['aqi_delta']     = df2['annual_aqi'] - ncr_avg
df2['aqi_delta_100'] = df2['aqi_delta'] / 100
df2['pct_uplift_sl'] = np.where(
    df2['aqi_delta'] > 0,
    (np.exp(-main_beta * df2['aqi_delta_100']) - 1),
    0.0
)
df2['price_uplift_sl'] = df2['price_total'] * df2['pct_uplift_sl']

total_cr      = df2['price_uplift_sl'].sum() / 1e7
props_hit     = (df2['aqi_delta'] > 0).sum()

print(f"\nScenario: Above-average AQI zones → NCR average ({ncr_avg:.0f})")
print(f"  Properties affected   : {props_hit:,} / {len(df2):,}")
print(f"  Total wealth unlocked : ₹{total_cr:,.0f} crore")
print(f"  Avg uplift/property   : "
      f"₹{df2.loc[df2['aqi_delta']>0,'price_uplift_sl'].mean():,.0f}")

print("\nBy city:")
city_sim = df2.groupby('city').agg(
    props          = ('price_total','count'),
    avg_aqi        = ('annual_aqi','mean'),
    uplift_cr_sl   = ('price_uplift_sl','sum')
)
city_sim['uplift_cr_sl'] = (city_sim['uplift_cr_sl'] / 1e7).round(1)
city_sim['avg_aqi']      = city_sim['avg_aqi'].round(1)
print(city_sim[['props','avg_aqi','uplift_cr_sl']].to_string())

# ── Save ──────────────────────────────────────────────────
df2.to_csv('data/property/ncr_final_merged.csv', index=False)
print(f"\n✓ Saved → data/property/ncr_final.csv")

