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
print("ANALYSIS — Within-City Station Comparison")
print("=" * 60)

df = pd.read_csv('data/property/ncr_final_merged.csv')

# Feature engineering
df['log_price_sqft']   = np.log(df['price_per_sqft'])
df['log_area']         = np.log(df['area_sqft'].clip(lower=1))
df['furnishing_score'] = df['furnishing'].map({
    'Unfurnished':0,'Semi-Furnished':1,'Furnished':2}).fillna(1)
df['is_new'] = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

# ── City pairs ────────────────────────────────────────────
PAIRS = {
    'Gurgaon': {
        'clean':  ('Vikas Sadan',      178),
        'dirty':  ('Gwal Pahari',       219),
        'gap':    41,
        'color':  ('#27ae60', '#e74c3c')
    },
    'New Delhi': {
        'clean':  ('Sri Aurobindo Marg', 188),
        'dirty':  ('Dwarka Sector 8',    230),
        'gap':    41,
        'color':  ('#2980b9', '#c0392b')
    },
    'Ghaziabad': {
        'clean':  ('Sanjay Nagar',       220),
        'dirty':  ('Anand Vihar',        248),
        'gap':    28,
        'color':  ('#8e44ad', '#e67e22')
    },
    'Noida': {
        'clean':  ('Knowledge Park III', 195),
        'dirty':  ('Sector 62 Noida',    224),
        'gap':    30,
        'color':  ('#1a2744', '#e74c3c')
    },
}

all_results = []

print()
for city, pair in PAIRS.items():
    clean_name, clean_aqi = pair['clean']
    dirty_name, dirty_aqi = pair['dirty']
    c1, c2 = pair['color']

    # Filter to this pair
    sub = df[df['nearest_station'].isin([clean_name, dirty_name])].copy()
    sub['is_clean'] = (sub['nearest_station'] == clean_name).astype(int)

    n_clean = (sub['nearest_station'] == clean_name).sum()
    n_dirty = (sub['nearest_station'] == dirty_name).sum()

    print(f"{'='*55}")
    print(f"City: {city}")
    print(f"  Clean station: {clean_name} (AQI {clean_aqi}, n={n_clean})")
    print(f"  Dirty station: {dirty_name} (AQI {dirty_aqi}, n={n_dirty})")
    print(f"  AQI gap: {dirty_aqi - clean_aqi} points")

    # ── Descriptive comparison ────────────────────────────
    print(f"\n  Descriptive stats:")
    for station in [clean_name, dirty_name]:
        s = sub[sub['nearest_station'] == station]
        label = 'CLEAN' if station == clean_name else 'DIRTY'
        print(f"    [{label}] {station}:")
        print(f"      Avg price/sqft: Rs {s['price_per_sqft'].mean():,.0f}")
        print(f"      Median price/sqft: Rs {s['price_per_sqft'].median():,.0f}")
        print(f"      Avg area: {s['area_sqft'].mean():,.0f} sqft")
        print(f"      Avg BHK: {s['bhk'].mean():.1f}")

    raw_gap = (sub[sub['nearest_station']==clean_name]['price_per_sqft'].mean() -
               sub[sub['nearest_station']==dirty_name]['price_per_sqft'].mean())
    raw_gap_pct = raw_gap / sub[sub['nearest_station']==dirty_name]['price_per_sqft'].mean() * 100
    print(f"\n  Raw price gap: Rs {raw_gap:+,.0f}/sqft ({raw_gap_pct:+.1f}%)")
    print(f"  (positive = clean station is more expensive)")

    # ── Controlled regression ─────────────────────────────
    controls = ['log_area', 'furnishing_score', 'is_new']
    sub_clean = sub[['log_price_sqft','is_clean'] + controls].dropna()
    sub_clean = sub_clean.apply(pd.to_numeric, errors='coerce').dropna()

    X = sm.add_constant(sub_clean[['is_clean'] + controls].astype(float))
    y = sub_clean['log_price_sqft'].astype(float)
    m = sm.OLS(y, X).fit(cov_type='HC3')

    b     = m.params['is_clean']
    p     = m.pvalues['is_clean']
    ci_l  = m.conf_int().loc['is_clean', 0]
    ci_h  = m.conf_int().loc['is_clean', 1]
    pct   = (np.exp(b) - 1) * 100
    sig   = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    print(f"\n  Controlled regression result:")
    print(f"  β(clean station) = {b:.4f}  [{ci_l:.4f}, {ci_h:.4f}]")
    print(f"  p = {p:.4f} {sig}  |  R² = {m.rsquared:.4f}")
    print(f"  Controlled price premium: {pct:+.1f}%")
    print(f"  Implied premium per AQI point: {pct/pair['gap']:+.2f}% per AQI point")

    if p < 0.05:
        direction = "MORE" if pct > 0 else "LESS"
        print(f"\n  ✓ SIGNIFICANT: Clean-air station properties cost {abs(pct):.1f}% {direction}")
        print(f"    than identical flats near the dirty station, even after")
        print(f"    controlling for size, furnishing, and transaction type.")
    else:
        print(f"\n  ✗ Not significant (p={p:.3f}) — within-{city} AQI gap")
        print(f"    does not produce a significant price premium after controls.")

    all_results.append({
        'city': city,
        'clean_station': clean_name,
        'dirty_station': dirty_name,
        'aqi_gap': dirty_aqi - clean_aqi,
        'raw_gap_pct': raw_gap_pct,
        'controlled_pct': pct,
        'p_value': p,
        'significant': p < 0.05,
        'n_clean': n_clean,
        'n_dirty': n_dirty,
    })
    print()

# ── Summary table ─────────────────────────────────────────
print("=" * 60)
print("SUMMARY — Within-City Clean vs Dirty Station")
print("=" * 60)
results_df = pd.DataFrame(all_results)
print()
print(f"{'City':15s} {'AQI Gap':8s} {'Raw Gap':10s} {'Controlled':12s} {'p-value':10s} {'Sig':5s}")
print("─" * 65)
for _, r in results_df.iterrows():
    sig = '***' if r['p_value']<0.001 else '**' if r['p_value']<0.01 else '*' if r['p_value']<0.05 else 'ns'
    print(f"{r['city']:15s} {r['aqi_gap']:8.0f} {r['raw_gap_pct']:+9.1f}% {r['controlled_pct']:+10.1f}%  {r['p_value']:8.4f}  {sig}")

print()
sig_results = results_df[results_df['significant']]
if len(sig_results) > 0:
    avg_pct_per_aqi = (sig_results['controlled_pct'] / sig_results['aqi_gap']).mean()
    print(f"Key finding: In {len(sig_results)}/{len(results_df)} cities, clean-air stations")
    print(f"command a significant price premium over dirty stations in the same city.")
    print(f"Average premium: {avg_pct_per_aqi:+.2f}% per AQI point difference")

# ── Visualisation ─────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Within-City Clean vs Dirty Station: Price Comparison',
             fontsize=14, fontweight='bold', color='#1a2744')
fig.patch.set_facecolor('white')

for ax, (city, pair), result in zip(axes.flatten(), PAIRS.items(), all_results):
    clean_name = pair['clean'][0]
    dirty_name = pair['dirty'][0]
    c1, c2 = pair['color']

    sub = df[df['nearest_station'].isin([clean_name, dirty_name])].copy()
    clean_prices = sub[sub['nearest_station']==clean_name]['price_per_sqft'].dropna()
    dirty_prices = sub[sub['nearest_station']==dirty_name]['price_per_sqft'].dropna()

    # Box plot
    bp = ax.boxplot([clean_prices, dirty_prices],
                    patch_artist=True,
                    labels=[f"Clean\n{clean_name.split()[0]}\nAQI {pair['clean'][1]}",
                            f"Dirty\n{dirty_name.split()[0]}\nAQI {pair['dirty'][1]}"],
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(c1)
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor(c2)
    bp['boxes'][1].set_alpha(0.8)

    ax.set_title(f"{city} (AQI gap = {pair['gap']} pts)",
                 fontsize=11, fontweight='bold', color='#1a2744')
    ax.set_ylabel('Price per Sqft (INR)', fontsize=9)
    ax.set_facecolor('#f4f6f8')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, color='white')

    sig = '***' if result['p_value']<0.001 else '**' if result['p_value']<0.01 else '*' if result['p_value']<0.05 else 'ns'
    ax.text(0.5, 0.95,
            f"Controlled premium: {result['controlled_pct']:+.1f}% ({sig})",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, fontweight='bold',
            color='#27ae60' if result['controlled_pct'] > 0 else '#e74c3c')

plt.tight_layout()
plt.savefig('outputs/station_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Chart saved → outputs/station_comparison.png")

