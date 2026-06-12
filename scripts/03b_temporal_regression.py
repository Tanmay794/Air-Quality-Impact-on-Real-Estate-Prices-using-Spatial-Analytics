import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("=" * 55)
print("STAGE 3B — Temporal & Seasonal Regression Analysis")
print("=" * 55)

df = pd.read_csv('data/property/ncr_dated_monthly_aqi.csv')
print(f"\nLoaded: {df.shape[0]} properties")

# ── Feature engineering ───────────────────────────────────
df['log_price_sqft']   = np.log(df['price_per_sqft'])
df['monthly_aqi_100']  = df['monthly_aqi'] / 100
df['annual_aqi_100']   = df['annual_aqi'] / 100
df['log_area']         = np.log(df['area_sqft'].clip(lower=1))
df['furnishing_score'] = df['furnishing'].map({
    'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}).fillna(1)
df['is_new'] = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

# Season dummies
df['is_winter']  = (df['listing_season'] == 'Winter').astype(int)
df['is_monsoon'] = (df['listing_season'] == 'Monsoon').astype(int)
df['is_spring']  = (df['listing_season'] == 'Spring').astype(int)
# Autumn = base category

controls = ['log_area', 'furnishing_score', 'is_new']

for col in ['log_price_sqft','monthly_aqi_100','annual_aqi_100'] + controls:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['log_price_sqft','monthly_aqi_100','annual_aqi_100'] + controls)
print(f"After cleaning: {len(df)} properties")

# ═══════════════════════════════════════════════════════════
# ANALYSIS 1: Annual vs Monthly AQI comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ANALYSIS 1 — Annual AQI vs Monthly AQI (same controls)")
print("=" * 55)

results = {}

for aqi_col, label in [('annual_aqi_100', 'Annual AQI (baseline)'),
                        ('monthly_aqi_100','Monthly AQI (listing month)')]:
    feats = [aqi_col] + controls
    X = sm.add_constant(df[feats].astype(float))
    y = df['log_price_sqft'].astype(float)
    m = sm.OLS(y, X).fit(cov_type='HC3')
    results[label] = m

    b    = m.params[aqi_col]
    p    = m.pvalues[aqi_col]
    ci_l = m.conf_int().loc[aqi_col, 0]
    ci_h = m.conf_int().loc[aqi_col, 1]
    pct  = (np.exp(b) - 1) * 100
    sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    print(f"\n  {label}")
    print(f"  {'─'*45}")
    print(f"  β = {b:.4f}  |  95% CI [{ci_l:.4f}, {ci_h:.4f}]")
    print(f"  p = {p:.4f} {sig}  |  R² = {m.rsquared:.4f}")
    print(f"  Price effect: {pct:+.1f}% per 100-pt AQI rise")

# ═══════════════════════════════════════════════════════════
# ANALYSIS 2: Seasonal regression — does discount vary by season?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ANALYSIS 2 — AQI Discount by Season (split sample)")
print("=" * 55)

seasons = {
    'Winter (Oct-Mar, HIGH pollution)': df['listing_season'].isin(['Winter','Autumn']),
    'Summer (Apr-Sep, LOW pollution)':  df['listing_season'].isin(['Spring','Monsoon']),
}

season_results = {}
for season_label, mask in seasons.items():
    sub = df[mask].copy()
    feats = ['monthly_aqi_100'] + controls
    X = sm.add_constant(sub[feats].astype(float))
    y = sub['log_price_sqft'].astype(float)
    m = sm.OLS(y, X).fit(cov_type='HC3')
    season_results[season_label] = m

    b   = m.params['monthly_aqi_100']
    p   = m.pvalues['monthly_aqi_100']
    pct = (np.exp(b) - 1) * 100
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    print(f"\n  {season_label}")
    print(f"  N = {len(sub):,}  |  R² = {m.rsquared:.4f}")
    print(f"  AQI β = {b:.4f}  |  p = {p:.4f} {sig}")
    print(f"  Price effect: {pct:+.1f}% per 100-pt AQI rise")

# Key finding
w_pct = (np.exp(season_results['Winter (Oct-Mar, HIGH pollution)'].params['monthly_aqi_100'])-1)*100
s_pct = (np.exp(season_results['Summer (Apr-Sep, LOW pollution)'].params['monthly_aqi_100'])-1)*100
print(f"\n  ┌─ Key finding: Winter discount = {w_pct:+.1f}%, Summer discount = {s_pct:+.1f}%")
if abs(w_pct) > abs(s_pct):
    print(f"  └─ Pollution discount is LARGER when pollution is visibly bad (winter)")
    print(f"     → Buyers respond to salient current pollution, not just annual reputation")
else:
    print(f"  └─ Pollution discount is similar across seasons")
    print(f"     → Buyers price location reputation, not current pollution levels")

# ═══════════════════════════════════════════════════════════
# ANALYSIS 3: New Property vs Resale — COVID hypothesis test
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ANALYSIS 3 — COVID Hypothesis: New Property vs Resale")
print("=" * 55)
print("H1: New buyers in 2021 discount pollution MORE than resale buyers")

txn_results = {}
for txn_label, txn_val in [('New Property (post-COVID buyers)', 1),
                             ('Resale (pre-COVID anchored)',      0)]:
    sub = df[df['is_new'] == txn_val].copy()
    feats = ['monthly_aqi_100'] + ['log_area', 'furnishing_score']
    X = sm.add_constant(sub[feats].astype(float))
    y = sub['log_price_sqft'].astype(float)
    m = sm.OLS(y, X).fit(cov_type='HC3')
    txn_results[txn_label] = m

    b   = m.params['monthly_aqi_100']
    p   = m.pvalues['monthly_aqi_100']
    pct = (np.exp(b) - 1) * 100
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    print(f"\n  {txn_label}")
    print(f"  N = {len(sub):,}  |  R² = {m.rsquared:.4f}")
    print(f"  AQI β = {b:.4f}  |  p = {p:.4f} {sig}")
    print(f"  Price effect: {pct:+.1f}% per 100-pt AQI rise")

new_pct    = (np.exp(txn_results['New Property (post-COVID buyers)'].params['monthly_aqi_100'])-1)*100
resale_pct = (np.exp(txn_results['Resale (pre-COVID anchored)'].params['monthly_aqi_100'])-1)*100
diff       = abs(new_pct) - abs(resale_pct)

print(f"\n  ┌─ New property discount: {new_pct:+.1f}%")
print(f"  ├─ Resale discount:       {resale_pct:+.1f}%")
print(f"  └─ Difference:            {diff:+.1f} pp")
if diff > 2:
    print(f"\n  RESULT: COVID clean-air awakening CONFIRMED")
    print(f"  New buyers discount pollution {diff:.1f}pp more than resale buyers")
elif diff < -2:
    print(f"\n  RESULT: Resale buyers more pollution-sensitive (unexpected)")
else:
    print(f"\n  RESULT: No significant difference — hypothesis not confirmed")

# ═══════════════════════════════════════════════════════════
# ANALYSIS 4: Days on market regression
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ANALYSIS 4 — Does Pollution Affect Time to Sell?")
print("=" * 55)

feats_dom = ['monthly_aqi_100', 'log_area', 'furnishing_score', 'is_new']
X_dom = sm.add_constant(df[feats_dom].astype(float))
y_dom = df['days_on_market'].astype(float)
m_dom = sm.OLS(y_dom, X_dom).fit(cov_type='HC3')

b_dom   = m_dom.params['monthly_aqi_100']
p_dom   = m_dom.pvalues['monthly_aqi_100']
sig_dom = '***' if p_dom<0.001 else '**' if p_dom<0.01 else '*' if p_dom<0.05 else 'ns'

print(f"\n  Dependent variable: days_on_market")
print(f"  N = {len(df):,}  |  R² = {m_dom.rsquared:.4f}")
print(f"\n  AQI effect on days on market:")
print(f"  β = {b_dom:.4f}  |  p = {p_dom:.4f} {sig_dom}")
print(f"  Interpretation: A 100-pt AQI increase adds {b_dom:.1f} days to time-to-sell")
print()
print(f"  Full coefficient table:")
coef_tbl = pd.DataFrame({
    'coef':   m_dom.params.round(3),
    'p':      m_dom.pvalues.round(4),
    'sig':    m_dom.pvalues.apply(lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')
})
print(coef_tbl.to_string())

if p_dom < 0.05:
    print(f"\n  FINDING: Higher pollution significantly INCREASES time to sell")
    print(f"  Polluted zones impose a dual penalty: lower prices AND slower sales")
else:
    print(f"\n  FINDING: AQI does not significantly affect time to sell (p={p_dom:.3f})")
    print(f"  The pollution discount is purely a price effect, not a velocity effect")

# ═══════════════════════════════════════════════════════════
# ANALYSIS 5: Quarterly price pattern
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ANALYSIS 5 — Quarterly Price Pattern by AQI Zone")
print("=" * 55)

# Split into high vs low AQI zones (above/below median)
aqi_median = df['annual_aqi'].median()
df['aqi_zone'] = np.where(df['annual_aqi'] > aqi_median, 'High AQI Zone', 'Low AQI Zone')

quarterly = df.groupby(['listing_quarter','aqi_zone']).agg(
    avg_price    = ('price_per_sqft', 'mean'),
    avg_monthly_aqi = ('monthly_aqi', 'mean'),
    n_props      = ('price_per_sqft', 'count')
).round(1).reset_index()

print(f"\nAQI zone split (median annual AQI = {aqi_median:.0f}):")
print(f"  High AQI zone: {(df['aqi_zone']=='High AQI Zone').sum()} properties")
print(f"  Low AQI zone:  {(df['aqi_zone']=='Low AQI Zone').sum()} properties")

print(f"\nAvg price/sqft by quarter and AQI zone:")
pivot = quarterly.pivot(index='listing_quarter', columns='aqi_zone', values='avg_price')
pivot['Price Gap (INR)'] = (pivot['Low AQI Zone'] - pivot['High AQI Zone']).round(0)
pivot['Gap %'] = ((pivot['Low AQI Zone'] - pivot['High AQI Zone']) / pivot['High AQI Zone'] * 100).round(1)
print(pivot.to_string())

print(f"\nAvg monthly AQI by quarter (confirms seasonality):")
aqi_pivot = quarterly.pivot(index='listing_quarter', columns='aqi_zone', values='avg_monthly_aqi')
print(aqi_pivot.to_string())

# ═══════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("SUMMARY — All Findings")
print("=" * 55)

print(f"""
┌─────────────────────────────────────────────────────────┐
│  Finding                          │ AQI β  │ Effect    │
├─────────────────────────────────────────────────────────┤
│  Annual AQI (baseline)            │ {results['Annual AQI (baseline)'].params['annual_aqi_100']:+.4f} │ {(np.exp(results['Annual AQI (baseline)'].params['annual_aqi_100'])-1)*100:+.1f}%    │
│  Monthly AQI (listing month)      │ {results['Monthly AQI (listing month)'].params['monthly_aqi_100']:+.4f} │ {(np.exp(results['Monthly AQI (listing month)'].params['monthly_aqi_100'])-1)*100:+.1f}%    │
│  Winter listings                  │ {season_results['Winter (Oct-Mar, HIGH pollution)'].params['monthly_aqi_100']:+.4f} │ {w_pct:+.1f}%    │
│  Summer listings                  │ {season_results['Summer (Apr-Sep, LOW pollution)'].params['monthly_aqi_100']:+.4f} │ {s_pct:+.1f}%    │
│  New property buyers              │ {txn_results['New Property (post-COVID buyers)'].params['monthly_aqi_100']:+.4f} │ {new_pct:+.1f}%    │
│  Resale buyers                    │ {txn_results['Resale (pre-COVID anchored)'].params['monthly_aqi_100']:+.4f} │ {resale_pct:+.1f}%    │
│  Days on market (AQI effect)      │ {b_dom:+.4f} │ {b_dom:+.1f} days  │
└─────────────────────────────────────────────────────────┘
""")

# Save enriched dataset
df.to_csv('data/property/ncr_temporal_analysis.csv', index=False)
print(f"✓ Saved → data/property/ncr_temporal_analysis.csv")

