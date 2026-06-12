import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("ANALYSIS — Pollution Justice: Who Bears the Burden?")
print("=" * 60)

df = pd.read_csv('data/property/ncr_final_merged.csv')

df['log_price_sqft']   = np.log(df['price_per_sqft'])
df['log_area']         = np.log(df['area_sqft'].clip(lower=1))
df['annual_aqi_100']   = df['annual_aqi'] / 100
df['furnishing_score'] = df['furnishing'].map({
    'Unfurnished':0,'Semi-Furnished':1,'Furnished':2}).fillna(1)
df['is_new'] = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

# ── Price segments ────────────────────────────────────────
df['price_segment'] = pd.cut(
    df['price_per_sqft'],
    bins=[0, 4000, 7000, 12000, 999999],
    labels=['Budget\n(<Rs4k)', 'Mid\n(Rs4k-7k)',
            'Premium\n(Rs7k-12k)', 'Luxury\n(Rs12k+)']
)

print("\nPrice segment breakdown:")
seg_summary = df.groupby('price_segment', observed=True).agg(
    n=('price_per_sqft','count'),
    avg_price=('price_per_sqft','mean'),
    avg_aqi=('annual_aqi','mean'),
    pct_ghaziabad=('city', lambda x: (x=='Ghaziabad').mean()*100),
    pct_gurgaon=('city', lambda x: (x=='Gurgaon').mean()*100),
).round(1)
print(seg_summary.to_string())

# ── Regression per segment ────────────────────────────────
print("\n" + "=" * 60)
print("AQI Coefficient by Price Segment")
print("=" * 60)

segments  = ['Budget\n(<Rs4k)', 'Mid\n(Rs4k-7k)',
             'Premium\n(Rs7k-12k)', 'Luxury\n(Rs12k+)']
controls  = ['log_area', 'furnishing_score', 'is_new']
seg_results = []

for seg in segments:
    sub = df[df['price_segment'] == seg].copy()
    if len(sub) < 50:
        print(f"\n  {seg}: insufficient observations ({len(sub)}), skipping")
        continue

    sub_clean = sub[['log_price_sqft','annual_aqi_100'] + controls].dropna()
    sub_clean = sub_clean.apply(pd.to_numeric, errors='coerce').dropna()

    X = sm.add_constant(sub_clean[['annual_aqi_100'] + controls].astype(float))
    y = sub_clean['log_price_sqft'].astype(float)
    m = sm.OLS(y, X).fit(cov_type='HC3')

    b   = m.params['annual_aqi_100']
    p   = m.pvalues['annual_aqi_100']
    ci_l = m.conf_int().loc['annual_aqi_100', 0]
    ci_h = m.conf_int().loc['annual_aqi_100', 1]
    pct = (np.exp(b) - 1) * 100
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    avg_aqi = sub['annual_aqi'].mean()

    seg_label = seg.replace('\n', ' ')
    print(f"\n  Segment: {seg_label}  (n={len(sub_clean):,}, avg AQI={avg_aqi:.0f})")
    print(f"  β = {b:.4f}  [{ci_l:.4f}, {ci_h:.4f}]")
    print(f"  p = {p:.4f} {sig}  |  R² = {m.rsquared:.4f}")
    print(f"  Price effect: {pct:+.1f}% per 100-pt AQI rise")

    seg_results.append({
        'segment':    seg_label.strip(),
        'n':          len(sub_clean),
        'avg_aqi':    avg_aqi,
        'beta':       b,
        'ci_low':     ci_l,
        'ci_high':    ci_h,
        'pct_effect': pct,
        'p_value':    p,
        'sig':        sig,
        'r2':         m.rsquared,
    })

results_df = pd.DataFrame(seg_results)

# ── Justice interpretation ────────────────────────────────
print("\n" + "=" * 60)
print("Pollution Justice Interpretation")
print("=" * 60)

print("\nAQI discount magnitude by segment:")
for _, r in results_df.iterrows():
    bar = '█' * int(abs(r['pct_effect']) / 2)
    print(f"  {r['segment']:20s}: {r['pct_effect']:+6.1f}%  {bar} ({r['sig']})")

print()
budget_beta  = results_df[results_df['segment'].str.contains('Budget')]['beta'].values
luxury_beta  = results_df[results_df['segment'].str.contains('Luxury')]['beta'].values
budget_aqi   = results_df[results_df['segment'].str.contains('Budget')]['avg_aqi'].values
luxury_aqi   = results_df[results_df['segment'].str.contains('Luxury')]['avg_aqi'].values

if len(budget_beta) > 0 and len(luxury_beta) > 0:
    b_pct = (np.exp(budget_beta[0])-1)*100
    l_pct = (np.exp(luxury_beta[0])-1)*100
    print(f"Budget buyers:  {b_pct:+.1f}% discount per 100-pt AQI  |  avg AQI = {budget_aqi[0]:.0f}")
    print(f"Luxury buyers:  {l_pct:+.1f}% discount per 100-pt AQI  |  avg AQI = {luxury_aqi[0]:.0f}")
    print()
    if abs(b_pct) < abs(l_pct):
        print("POLLUTION JUSTICE FINDING:")
        print("Budget buyers receive LESS price compensation per unit of pollution")
        print("than luxury buyers. The market under-prices pollution risk for")
        print("lower-income segments — a regressive environmental externality.")
    elif abs(b_pct) > abs(l_pct):
        print("POLLUTION JUSTICE FINDING:")
        print("Budget buyers receive MORE price compensation per unit of pollution.")
        print("This may reflect forced exposure — budget buyers in heavily")
        print("polluted zones demand proportionally larger discounts because")
        print("they lack the option to move to cleaner areas.")
    else:
        print("The pollution discount is uniform across price segments.")

# ── AQI exposure by segment ───────────────────────────────
print("\n" + "=" * 60)
print("Who Lives in the Most Polluted Areas?")
print("=" * 60)
exposure = df.groupby('price_segment', observed=True)['annual_aqi'].agg(
    ['mean','median','min','max','count']).round(1)
print(exposure.to_string())

print()
budget_aqi_mean  = df[df['price_segment']=='Budget\n(<Rs4k)']['annual_aqi'].mean()
luxury_aqi_mean  = df[df['price_segment']=='Luxury\n(Rs12k+)']['annual_aqi'].mean()
if not np.isnan(budget_aqi_mean) and not np.isnan(luxury_aqi_mean):
    print(f"Budget buyers face AQI {budget_aqi_mean:.0f} on average")
    print(f"Luxury buyers face AQI {luxury_aqi_mean:.0f} on average")
    print(f"Budget buyers face {budget_aqi_mean - luxury_aqi_mean:+.0f} AQI points MORE pollution")
    print(f"→ Lower-income buyers are systematically more pollution-exposed")

# ── Visualisation ─────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Pollution Justice Analysis: Who Bears the Air Quality Burden?',
             fontsize=13, fontweight='bold', color='#1a2744')
fig.patch.set_facecolor('white')

NAVY  = '#1a2744'
RED   = '#c0392b'
GREEN = '#27ae60'
BLUE  = '#2980b9'
LGREY = '#f4f6f8'

seg_labels = [r['segment'].replace('(<Rs4k)','').replace('(Rs4k-7k)','').replace('(Rs7k-12k)','').replace('(Rs12k+)','').strip()
              for _, r in results_df.iterrows()]

# Panel 1: AQI coefficient by segment with CI
ax = axes[0]
betas = [r['beta'] for _, r in results_df.iterrows()]
ci_lows  = [r['ci_low']  for _, r in results_df.iterrows()]
ci_highs = [r['ci_high'] for _, r in results_df.iterrows()]
colors_bar = [RED if b < 0 else GREEN for b in betas]

bars = ax.bar(seg_labels, betas, color=colors_bar, alpha=0.8, width=0.6)
for i, (b, cl, ch) in enumerate(zip(betas, ci_lows, ci_highs)):
    ax.errorbar(i, b, yerr=[[b-cl],[ch-b]],
                fmt='none', color=NAVY, capsize=5, linewidth=1.5)
    pct = (np.exp(b)-1)*100
    ax.text(i, b - 0.005 if b < 0 else b + 0.005,
            f'{pct:+.1f}%', ha='center',
            va='top' if b < 0 else 'bottom',
            fontsize=8, fontweight='bold', color='white')

ax.axhline(0, color=NAVY, linewidth=1)
ax.set_ylabel('AQI Coefficient (β)', fontsize=9)
ax.set_title('AQI Discount by\nPrice Segment', fontsize=10, fontweight='bold', color=NAVY)
ax.set_facecolor(LGREY)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3, color='white')

# Panel 2: Avg AQI exposure by segment
ax = axes[1]
exposure_vals = df.groupby('price_segment', observed=True)['annual_aqi'].mean()
seg_order = ['Budget\n(<Rs4k)', 'Mid\n(Rs4k-7k)', 'Premium\n(Rs7k-12k)', 'Luxury\n(Rs12k+)']
labels_short = ['Budget', 'Mid', 'Premium', 'Luxury']
vals = [exposure_vals.get(s, np.nan) for s in seg_order]
bar_colors = [RED if v > exposure_vals.mean() else GREEN for v in vals if not np.isnan(v)]
valid_labels = [l for l, v in zip(labels_short, vals) if not np.isnan(v)]
valid_vals   = [v for v in vals if not np.isnan(v)]

bars2 = ax.bar(valid_labels, valid_vals, color=bar_colors, alpha=0.8, width=0.6)
ax.axhline(np.mean(valid_vals), color=NAVY, linestyle='--',
           linewidth=1.5, label=f'NCR avg ({np.mean(valid_vals):.0f})')
for bar, val in zip(bars2, valid_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'AQI\n{val:.0f}', ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('Avg Annual AQI', fontsize=9)
ax.set_title('Pollution Exposure\nby Income Segment', fontsize=10, fontweight='bold', color=NAVY)
ax.set_facecolor(LGREY)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3, color='white')
ax.legend(fontsize=8)

# Panel 3: Price per unit of AQI improvement
ax = axes[2]
price_per_aqi = []
for seg, label in zip(seg_order, labels_short):
    seg_df = df[df['price_segment']==seg]
    if len(seg_df) < 50:
        continue
    avg_price = seg_df['price_per_sqft'].mean()
    r = results_df[results_df['segment'].str.contains(label.split()[0])]
    if len(r) > 0:
        pct_per_100 = abs((np.exp(r['beta'].values[0])-1)*100)
        rs_per_100_pts = avg_price * pct_per_100 / 100
        price_per_aqi.append((label, rs_per_100_pts, pct_per_100))

if price_per_aqi:
    labels_p = [x[0] for x in price_per_aqi]
    vals_p   = [x[1] for x in price_per_aqi]
    pcts_p   = [x[2] for x in price_per_aqi]
    colors_p = [BLUE] * len(vals_p)
    bars3 = ax.bar(labels_p, vals_p, color=colors_p, alpha=0.8, width=0.6)
    for bar, val, pct in zip(bars3, vals_p, pcts_p):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                f'Rs{val:,.0f}\n({pct:.1f}%)', ha='center', fontsize=7.5)
    ax.set_ylabel('Rs per sqft per 100-pt AQI improvement', fontsize=9)
    ax.set_title('Price Compensation\nper Unit of Pollution', fontsize=10, fontweight='bold', color=NAVY)
    ax.set_facecolor(LGREY)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, color='white')

plt.tight_layout()
plt.savefig('outputs/pollution_justice.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Chart saved → outputs/pollution_justice.png")

