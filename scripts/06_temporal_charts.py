import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("Run: pip install matplotlib")
    exit()

print("=" * 55)
print("STAGE 6 — Temporal Visualisations")
print("=" * 55)

df = pd.read_csv('data/property/ncr_temporal_analysis.csv')
print(f"Loaded: {len(df)} properties")

os.makedirs('outputs', exist_ok=True)

MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

NAVY   = '#1a2744'
RED    = '#c0392b'
BLUE   = '#2980b9'
GREEN  = '#27ae60'
ORANGE = '#e67e22'
LGREY  = '#f4f6f8'
DGREY  = '#2c3e50'

# ═══════════════════════════════════════════════════════════
# CHART 1: Monthly AQI seasonality per city
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Monthly AQI Seasonality by City (2021)', fontsize=14, fontweight='bold', color=NAVY)
fig.patch.set_facecolor('white')

cities = ['New Delhi','Gurgaon','Noida','Ghaziabad','Greater Noida','Faridabad']
colors = [NAVY, GREEN, BLUE, RED, ORANGE, '#8e44ad']

for ax, city, color in zip(axes.flatten(), cities, colors):
    city_df = df[df['city'] == city]
    monthly_aqi = city_df.groupby('listing_month')['monthly_aqi'].mean()
    months = sorted(monthly_aqi.index)
    values = [monthly_aqi[m] for m in months]
    month_labels = [MONTH_NAMES[m] for m in months]

    ax.plot(range(len(months)), values, color=color, linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(range(len(months)), values,
                    alpha=0.1, color=color)
    ax.axhline(y=city_df['annual_aqi'].mean(), color=color,
               linestyle='--', alpha=0.5, linewidth=1, label='Annual avg')

    ax.set_title(city, fontsize=11, fontweight='bold', color=NAVY)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(month_labels, fontsize=7, rotation=45)
    ax.set_ylabel('AQI', fontsize=8)
    ax.set_facecolor(LGREY)
    ax.grid(True, alpha=0.3, color='white')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('outputs/chart1_monthly_aqi_by_city.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 1 saved — Monthly AQI seasonality by city")

# ═══════════════════════════════════════════════════════════
# CHART 2: Quarterly price gap — High vs Low AQI zones
# ═══════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Quarterly Price Patterns: High vs Low AQI Zones', fontsize=13, fontweight='bold', color=NAVY)
fig.patch.set_facecolor('white')

quarterly = df.groupby(['listing_quarter','aqi_zone']).agg(
    avg_price   = ('price_per_sqft','mean'),
    avg_aqi     = ('monthly_aqi','mean'),
    n_props     = ('price_per_sqft','count')
).reset_index()

quarters  = [1, 2, 3, 4]
q_labels  = ['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)']
x = np.arange(len(quarters))
width = 0.35

high_prices = [quarterly[(quarterly['listing_quarter']==q) & (quarterly['aqi_zone']=='High AQI Zone')]['avg_price'].values[0]
               if len(quarterly[(quarterly['listing_quarter']==q) & (quarterly['aqi_zone']=='High AQI Zone')]) > 0 else 0
               for q in quarters]
low_prices  = [quarterly[(quarterly['listing_quarter']==q) & (quarterly['aqi_zone']=='Low AQI Zone')]['avg_price'].values[0]
               if len(quarterly[(quarterly['listing_quarter']==q) & (quarterly['aqi_zone']=='Low AQI Zone')]) > 0 else 0
               for q in quarters]

bars1 = ax1.bar(x - width/2, high_prices, width, label='High AQI Zone', color=RED, alpha=0.8)
bars2 = ax1.bar(x + width/2, low_prices,  width, label='Low AQI Zone',  color=GREEN, alpha=0.8)
ax1.set_xlabel('Quarter', fontsize=10)
ax1.set_ylabel('Avg Price per Sqft (INR)', fontsize=10)
ax1.set_title('Avg Price by Quarter', fontsize=11, fontweight='bold', color=NAVY)
ax1.set_xticks(x)
ax1.set_xticklabels(q_labels)
ax1.legend()
ax1.set_facecolor(LGREY)
ax1.spines[['top','right']].set_visible(False)
ax1.grid(axis='y', alpha=0.3, color='white')
for bar in bars1:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f'₹{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f'₹{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=7)

# Price gap
gaps = [l - h for l, h in zip(low_prices, high_prices)]
gap_pct = [(l-h)/h*100 if h > 0 else 0 for l, h in zip(low_prices, high_prices)]
bar_colors = [GREEN if g >= 0 else RED for g in gaps]
ax2.bar(x, gap_pct, color=bar_colors, alpha=0.8)
ax2.axhline(0, color=DGREY, linewidth=1)
ax2.set_xlabel('Quarter', fontsize=10)
ax2.set_ylabel('Price Premium of Clean Air Zone (%)', fontsize=10)
ax2.set_title('Clean Air Premium by Quarter\n(Low AQI vs High AQI)', fontsize=11, fontweight='bold', color=NAVY)
ax2.set_xticks(x)
ax2.set_xticklabels(q_labels)
ax2.set_facecolor(LGREY)
ax2.spines[['top','right']].set_visible(False)
ax2.grid(axis='y', alpha=0.3, color='white')
for i, (bar_val, pct) in enumerate(zip(x, gap_pct)):
    ax2.text(bar_val, pct + 0.3 if pct >= 0 else pct - 0.8,
             f'{pct:+.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/chart2_quarterly_price_gap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 2 saved — Quarterly price gap High vs Low AQI")

# ═══════════════════════════════════════════════════════════
# CHART 3: COVID hypothesis — New Property vs Resale AQI discount
# ═══════════════════════════════════════════════════════════
import statsmodels.api as sm

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('COVID Clean-Air Awakening: New Property vs Resale Buyers', fontsize=13, fontweight='bold', color=NAVY)
fig.patch.set_facecolor('white')

df['log_price_sqft']  = np.log(df['price_per_sqft'])
df['monthly_aqi_100'] = df['monthly_aqi'] / 100
df['log_area']        = np.log(df['area_sqft'].clip(lower=1))
df['furnishing_score']= df['furnishing'].map({'Unfurnished':0,'Semi-Furnished':1,'Furnished':2}).fillna(1)
df['is_new']          = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

# Panel 1: Monthly AQI distribution by transaction type
ax = axes[0]
new_aqi    = df[df['is_new']==1]['monthly_aqi']
resale_aqi = df[df['is_new']==0]['monthly_aqi']
bins = np.linspace(df['monthly_aqi'].min(), df['monthly_aqi'].max(), 25)
ax.hist(resale_aqi, bins=bins, alpha=0.6, color=BLUE,  label=f'Resale (n={len(resale_aqi):,})',  density=True)
ax.hist(new_aqi,    bins=bins, alpha=0.6, color=ORANGE, label=f'New Prop (n={len(new_aqi):,})', density=True)
ax.set_xlabel('Monthly AQI at Listing', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('AQI Exposure Distribution\nNew vs Resale', fontsize=10, fontweight='bold', color=NAVY)
ax.legend(fontsize=8)
ax.set_facecolor(LGREY)
ax.spines[['top','right']].set_visible(False)

# Panel 2: Scatter — monthly AQI vs price by transaction type
ax = axes[1]
for txn, color, label in [(0, BLUE, 'Resale'), (1, ORANGE, 'New Property')]:
    sub = df[df['is_new']==txn].sample(min(400, (df['is_new']==txn).sum()), random_state=42)
    ax.scatter(sub['monthly_aqi'], sub['price_per_sqft'],
               alpha=0.3, s=15, color=color, label=label)
    # Trend line
    z = np.polyfit(sub['monthly_aqi'], sub['price_per_sqft'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(sub['monthly_aqi'].min(), sub['monthly_aqi'].max(), 100)
    ax.plot(x_line, p_line(x_line), color=color, linewidth=2)

ax.set_xlabel('Monthly AQI', fontsize=9)
ax.set_ylabel('Price per Sqft (INR)', fontsize=9)
ax.set_title('AQI vs Price\n(with trend lines)', fontsize=10, fontweight='bold', color=NAVY)
ax.legend(fontsize=8)
ax.set_facecolor(LGREY)
ax.spines[['top','right']].set_visible(False)

# Panel 3: AQI coefficient comparison with CI
ax = axes[2]
groups = ['New Property', 'Resale']
betas  = []
cis    = []
pcts   = []
for txn_val in [1, 0]:
    sub = df[df['is_new']==txn_val]
    feats = ['monthly_aqi_100','log_area','furnishing_score']
    X = sm.add_constant(sub[feats].astype(float).dropna())
    y = sub['log_price_sqft'].astype(float).loc[X.index]
    m = sm.OLS(y, X).fit(cov_type='HC3')
    b = m.params['monthly_aqi_100']
    ci = m.conf_int().loc['monthly_aqi_100']
    betas.append(b)
    cis.append(ci)
    pcts.append((np.exp(b)-1)*100)

colors_bar = [ORANGE, BLUE]
bars = ax.bar(groups, betas, color=colors_bar, alpha=0.8, width=0.5)
for i, (b, ci, color) in enumerate(zip(betas, cis, colors_bar)):
    ax.errorbar(i, b, yerr=[[b-ci[0]], [ci[1]-b]],
                fmt='none', color=DGREY, capsize=6, linewidth=2)
    ax.text(i, b - 0.01, f'{(np.exp(b)-1)*100:+.1f}%',
            ha='center', va='top', fontsize=10, fontweight='bold', color='white')

ax.axhline(0, color=DGREY, linewidth=1)
ax.set_ylabel('AQI Coefficient (β)', fontsize=9)
ax.set_title('AQI Discount Coefficient\n(with 95% CI)', fontsize=10, fontweight='bold', color=NAVY)
ax.set_facecolor(LGREY)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3, color='white')

plt.tight_layout()
plt.savefig('outputs/chart3_covid_hypothesis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 3 saved — COVID hypothesis new vs resale")

# ═══════════════════════════════════════════════════════════
# CHART 4: Days on market vs AQI
# ═══════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Time to Sell: Does Pollution Slow the Market?', fontsize=13, fontweight='bold', color=NAVY)
fig.patch.set_facecolor('white')

# Panel 1: Days on market by AQI zone and city
city_dom = df.groupby(['city','aqi_zone'])['days_on_market'].mean().reset_index()
city_names = df['city'].unique()
x_pos = np.arange(len(city_names))
width = 0.35

for i, zone in enumerate(['High AQI Zone','Low AQI Zone']):
    vals = [city_dom[(city_dom['city']==c)&(city_dom['aqi_zone']==zone)]['days_on_market'].values[0]
            if len(city_dom[(city_dom['city']==c)&(city_dom['aqi_zone']==zone)]) > 0 else 0
            for c in city_names]
    color = RED if zone == 'High AQI Zone' else GREEN
    ax1.bar(x_pos + (i-0.5)*width, vals, width, label=zone, color=color, alpha=0.8)

ax1.set_xlabel('City', fontsize=9)
ax1.set_ylabel('Avg Days on Market', fontsize=9)
ax1.set_title('Days on Market by City\nand AQI Zone', fontsize=10, fontweight='bold', color=NAVY)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([c.replace(' ','\n') for c in city_names], fontsize=8)
ax1.legend(fontsize=8)
ax1.set_facecolor(LGREY)
ax1.spines[['top','right']].set_visible(False)
ax1.grid(axis='y', alpha=0.3, color='white')

# Panel 2: Scatter — monthly AQI vs days on market
sample = df.sample(min(1000, len(df)), random_state=42)
ax2.scatter(sample['monthly_aqi'], sample['days_on_market'],
            alpha=0.3, s=10, color=NAVY)
z = np.polyfit(sample['monthly_aqi'], sample['days_on_market'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(sample['monthly_aqi'].min(), sample['monthly_aqi'].max(), 100)
ax2.plot(x_line, p_line(x_line), color=RED, linewidth=2.5, label='Trend')
slope = z[0]
ax2.text(0.05, 0.92, f'Slope: {slope:+.2f} days per AQI point',
         transform=ax2.transAxes, fontsize=9, color=RED, fontweight='bold')
ax2.set_xlabel('Monthly AQI at Listing', fontsize=9)
ax2.set_ylabel('Days on Market', fontsize=9)
ax2.set_title('Monthly AQI vs Days on Market', fontsize=10, fontweight='bold', color=NAVY)
ax2.legend(fontsize=9)
ax2.set_facecolor(LGREY)
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/chart4_days_on_market.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 4 saved — Days on market vs AQI")

# ═══════════════════════════════════════════════════════════
# CHART 5: Seasonal AQI discount — the key seasonal finding
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Seasonal Pollution Discount: Do Buyers React to Current Pollution?', fontsize=12, fontweight='bold', color=NAVY)
fig.patch.set_facecolor('white')

# Panel 1: Monthly avg price vs monthly AQI
monthly_summary = df.groupby('listing_month').agg(
    avg_price   = ('price_per_sqft', 'mean'),
    avg_aqi     = ('monthly_aqi', 'mean'),
    n_props     = ('price_per_sqft', 'count')
).reset_index()

ax1 = axes[0]
ax1_twin = ax1.twinx()

months_sorted = monthly_summary['listing_month'].tolist()
m_labels = [MONTH_NAMES[m] for m in months_sorted]

line1, = ax1.plot(range(12), monthly_summary['avg_price'],
                   color=BLUE, linewidth=2.5, marker='o', markersize=6, label='Avg Price/sqft')
line2, = ax1_twin.plot(range(12), monthly_summary['avg_aqi'],
                        color=RED, linewidth=2.5, marker='s', markersize=6,
                        linestyle='--', label='Avg Monthly AQI')

ax1.set_xticks(range(12))
ax1.set_xticklabels(m_labels, rotation=45, fontsize=8)
ax1.set_ylabel('Avg Price per Sqft (INR)', color=BLUE, fontsize=9)
ax1_twin.set_ylabel('Avg Monthly AQI', color=RED, fontsize=9)
ax1.set_title('Price vs AQI by Listing Month', fontsize=10, fontweight='bold', color=NAVY)
ax1.set_facecolor(LGREY)
ax1.spines[['top','right']].set_visible(False)
lines = [line1, line2]
ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)

# Panel 2: AQI coefficient by month (rolling)
ax2 = axes[1]
month_betas = []
month_pvals = []
months_valid = []

for m in range(1, 13):
    sub = df[df['listing_month'] == m]
    if len(sub) < 50:
        continue
    feats = ['monthly_aqi_100','log_area','furnishing_score','is_new']
    sub_clean = sub[feats + ['log_price_sqft']].dropna()
    if len(sub_clean) < 30:
        continue
    X = sm.add_constant(sub_clean[feats].astype(float))
    y = sub_clean['log_price_sqft'].astype(float)
    mod = sm.OLS(y, X).fit()
    month_betas.append((np.exp(mod.params['monthly_aqi_100'])-1)*100)
    month_pvals.append(mod.pvalues['monthly_aqi_100'])
    months_valid.append(m)

colors_monthly = [RED if p < 0.05 else MGREY_C if p < 0.1 else '#cccccc'
                  for p in month_pvals]
MGREY_C = '#95a5a6'
colors_monthly = [RED if p < 0.05 else '#95a5a6' for p in month_pvals]

bars = ax2.bar([MONTH_NAMES[m] for m in months_valid], month_betas,
               color=colors_monthly, alpha=0.85)
ax2.axhline(0, color=DGREY, linewidth=1)
ax2.set_xlabel('Listing Month', fontsize=9)
ax2.set_ylabel('AQI Price Effect (%)', fontsize=9)
ax2.set_title('Monthly AQI Discount by Listing Month\n(red = significant p<0.05)', fontsize=10, fontweight='bold', color=NAVY)
ax2.tick_params(axis='x', rotation=45)
ax2.set_facecolor(LGREY)
ax2.spines[['top','right']].set_visible(False)
ax2.grid(axis='y', alpha=0.3, color='white')
for bar, val in zip(bars, month_betas):
    ax2.text(bar.get_x()+bar.get_width()/2,
             bar.get_height() - 0.5 if val < 0 else bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='top' if val < 0 else 'bottom',
             fontsize=7)

plt.tight_layout()
plt.savefig('outputs/chart5_seasonal_discount.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 5 saved — Seasonal pollution discount by month")

print(f"\n{'='*55}")
print("All 5 charts saved to outputs/")
print("  chart1_monthly_aqi_by_city.png")
print("  chart2_quarterly_price_gap.png")
print("  chart3_covid_hypothesis.png")
print("  chart4_days_on_market.png")
print("  chart5_seasonal_discount.png")

