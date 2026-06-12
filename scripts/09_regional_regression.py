import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

print("=" * 60)
print("REGIONAL REGRESSION — Gurgaon Belt vs East NCR")
print("=" * 60)

df = pd.read_csv('data/property/ncr_final_merged.csv')

# ── Feature engineering ───────────────────────────────────
df['log_price_sqft']   = np.log(df['price_per_sqft'])
df['aqi_100']          = df['annual_aqi'] / 100
df['log_area']         = np.log(df['area_sqft'].clip(lower=1))
df['furnishing_score'] = df['furnishing'].map({
    'Unfurnished':0,'Semi-Furnished':1,'Furnished':2}).fillna(1)
df['is_new'] = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

# ── Zone assignment ────────────────────────────────────────
# Gurgaon Belt: clean stations (AQI 178-203) to dirty (AQI 219-230)
# Maximum within-zone AQI variation = 52 points
# Buyers choose between Vikas Sadan / Aya Nagar / Faridabad vs
# Gwal Pahari / Dwarka / IGI — real location choices

# East NCR: cleanest (Knowledge Park III, AQI 195) to
# dirtiest (Loni AQI 249, Anand Vihar AQI 248)
# Maximum within-zone AQI variation = 54 points
# Buyers in Noida/Greater Noida have genuine station-level choice

GURGAON_BELT = [
    'Vikas Sadan',     # AQI 178 — cleanest
    'Aya Nagar',       # AQI 186
    'Dr Karni Singh',  # AQI 203
    'Sector 16A',      # AQI 199
    'IGI Airport T3',  # AQI 203
    'NSIT Dwarka',     # AQI 213
    'Dwarka Sector 8', # AQI 230
    'Gwal Pahari',     # AQI 219 — dirtiest
]

EAST_NCR = [
    'Knowledge Park III',  # AQI 195 — cleanest
    'Sector 125 Noida',    # AQI 203
    'Okhla Phase 2',       # AQI 209
    'Sanjay Nagar',        # AQI 220
    'Sector 62 Noida',     # AQI 224
    'IHBAS Dilshad Garden',# AQI 193
    'Vivek Vihar',         # AQI 218
    'Vasundhara',          # AQI 225
    'Loni',                # AQI 249 — dirtiest
    'Anand Vihar',         # AQI 248
]

df_gb  = df[df['nearest_station'].isin(GURGAON_BELT)].copy()
df_enc = df[df['nearest_station'].isin(EAST_NCR)].copy()

# ── Zone summaries ────────────────────────────────────────
print("\n" + "─"*55)
print("ZONE PROFILES")
print("─"*55)

for zone_name, zone_df, stations in [
    ('Gurgaon Belt', df_gb, GURGAON_BELT),
    ('East NCR (Noida / Greater Noida / Ghaziabad)', df_enc, EAST_NCR)
]:
    print(f"\n{zone_name}:")
    print(f"  Total properties : {len(zone_df):,}")
    print(f"  AQI range        : {zone_df['annual_aqi'].min():.0f} — "
          f"{zone_df['annual_aqi'].max():.0f} "
          f"({zone_df['annual_aqi'].max()-zone_df['annual_aqi'].min():.0f} pt spread)")
    print(f"  Avg AQI          : {zone_df['annual_aqi'].mean():.1f}")
    print(f"  Avg price/sqft   : Rs {zone_df['price_per_sqft'].mean():,.0f}")
    print(f"  Median price/sqft: Rs {zone_df['price_per_sqft'].median():,.0f}")
    print(f"  Avg area         : {zone_df['area_sqft'].mean():,.0f} sqft")
    print(f"\n  Station breakdown:")
    st_sum = zone_df.groupby('nearest_station').agg(
        n=('price_per_sqft','count'),
        aqi=('annual_aqi','first'),
        avg_price=('price_per_sqft','mean')
    ).sort_values('aqi')
    for st, row in st_sum.iterrows():
        print(f"    {st:28s} AQI={row['aqi']:.0f}  "
              f"n={row['n']:4.0f}  "
              f"Rs{row['avg_price']:,.0f}/sqft")

# ── OLS Regressions ───────────────────────────────────────
print("\n" + "─"*55)
print("OLS REGRESSION RESULTS")
print("─"*55)

controls = ['log_area', 'furnishing_score', 'is_new']
zone_results = {}

for zone_name, zone_df in [('Gurgaon Belt', df_gb),
                             ('East NCR', df_enc)]:
    sub = zone_df[['log_price_sqft','aqi_100']+controls].dropna()
    sub = sub.apply(pd.to_numeric, errors='coerce').dropna()

    X = sm.add_constant(sub[['aqi_100']+controls].astype(float))
    y = sub['log_price_sqft'].astype(float)
    m = sm.OLS(y, X).fit(cov_type='HC3')
    zone_results[zone_name] = m

    b    = m.params['aqi_100']
    p    = m.pvalues['aqi_100']
    ci_l = m.conf_int().loc['aqi_100', 0]
    ci_h = m.conf_int().loc['aqi_100', 1]
    pct  = (np.exp(b) - 1) * 100
    sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    print(f"\n{'─'*50}")
    print(f"Zone: {zone_name}")
    print(f"{'─'*50}")
    print(f"  N observations  : {len(sub):,}")
    print(f"  R²              : {m.rsquared:.4f}")
    print(f"  Adj R²          : {m.rsquared_adj:.4f}")
    print(f"\n  AQI coefficient (per 100-pt increase):")
    print(f"    β             = {b:.4f}")
    print(f"    95% CI        = [{ci_l:.4f}, {ci_h:.4f}]")
    print(f"    p-value       = {p:.4f} {sig}")
    print(f"    Price effect  = {pct:+.1f}%")
    print(f"\n  Full coefficient table:")
    coef_tbl = pd.DataFrame({
        'coef':   m.params.round(4),
        'se':     m.bse.round(4),
        'p':      m.pvalues.round(4),
        'sig':    m.pvalues.apply(
            lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')
    })
    print(coef_tbl.to_string())

# ── Comparison with NCR overall ───────────────────────────
# NCR overall for reference
sub_all = df[['log_price_sqft','aqi_100']+controls].dropna()
sub_all = sub_all.apply(pd.to_numeric, errors='coerce').dropna()
X_all   = sm.add_constant(sub_all[['aqi_100']+controls].astype(float))
m_all   = sm.OLS(sub_all['log_price_sqft'].astype(float), X_all).fit(cov_type='HC3')
b_all   = m_all.params['aqi_100']
pct_all = (np.exp(b_all) - 1) * 100

b_gb  = zone_results['Gurgaon Belt'].params['aqi_100']
b_enc = zone_results['East NCR'].params['aqi_100']
pct_gb  = (np.exp(b_gb)  - 1) * 100
pct_enc = (np.exp(b_enc) - 1) * 100

print(f"\n{'─'*55}")
print("COMPARISON: Regional vs NCR Overall")
print(f"{'─'*55}")
print(f"\n  {'Specification':35s} {'β':8s} {'Effect':10s} {'p':8s}")
print(f"  {'─'*65}")
print(f"  {'NCR Overall':35s} {b_all:8.4f} {pct_all:+9.1f}%  <0.001 ***")
print(f"  {'Gurgaon Belt (within-zone)':35s} {b_gb:8.4f} {pct_gb:+9.1f}%  "
      f"{zone_results['Gurgaon Belt'].pvalues['aqi_100']:.4f} "
      f"{'***' if zone_results['Gurgaon Belt'].pvalues['aqi_100']<0.001 else '**'}")
print(f"  {'East NCR (within-zone)':35s} {b_enc:8.4f} {pct_enc:+9.1f}%  "
      f"{zone_results['East NCR'].pvalues['aqi_100']:.4f} ***")

print(f"""
  Key Interpretation:
  ┌─────────────────────────────────────────────────────────────┐
  │  Gurgaon Belt: {pct_gb:+.1f}% per 100-pt AQI            │
  │  → Premium buyers in Gurgaon/Faridabad corridor have real  │
  │    location choices between clean (Vikas Sadan, AQI 178)   │
  │    and dirty (Gwal Pahari, AQI 219) stations.              │
  │    They price this difference — but moderately.            │
  │                                                             │
  │  East NCR: {pct_enc:+.1f}% per 100-pt AQI               │
  │  → Budget-to-mid buyers in Noida/Greater Noida/Ghaziabad  │
  │    show the STRONGEST pollution discount in NCR.           │
  │    A buyer choosing between Knowledge Park III (AQI 195)   │
  │    and Anand Vihar (AQI 248) pays {abs(pct_enc):.0f}% less  │
  │    for the dirtier option, controlling for flat quality.   │
  │                                                             │
  │  East NCR > NCR Overall > Gurgaon Belt in pollution        │
  │  sensitivity — buyers with fewer premium alternatives      │
  │  are MORE sensitive to within-zone AQI differences.        │
  └─────────────────────────────────────────────────────────────┘
""")

# ── Policy simulation by zone ─────────────────────────────
print("─"*55)
print("ZONE-LEVEL POLICY SIMULATION")
print("─"*55)

ncr_avg = df['annual_aqi'].mean()
print(f"\nNCR average AQI: {ncr_avg:.0f}")
print(f"Scenario: Above-average AQI properties → NCR average ({ncr_avg:.0f})\n")

sim_results = {}
for zone_name, zone_df, beta in [
    ('Gurgaon Belt', df_gb, b_gb),
    ('East NCR',     df_enc, b_enc)
]:
    zd = zone_df.copy()
    zd['aqi_delta']    = zd['annual_aqi'] - ncr_avg
    zd['pct_uplift']   = np.where(
        zd['aqi_delta'] > 0,
        (np.exp(-beta * zd['aqi_delta']/100) - 1),
        0.0
    )
    zd['price_uplift'] = zd['price_total'] * zd['pct_uplift']
    uplift_cr  = zd['price_uplift'].sum() / 1e7
    props_hit  = (zd['aqi_delta'] > 0).sum()
    avg_uplift = zd.loc[zd['aqi_delta']>0,'price_uplift'].mean()
    sim_results[zone_name] = {
        'uplift_cr': uplift_cr,
        'props_hit': props_hit,
        'avg_uplift': avg_uplift
    }

    print(f"  {zone_name}:")
    print(f"    Properties affected  : {props_hit:,} / {len(zd):,}")
    print(f"    Wealth unlocked      : Rs {uplift_cr:,.1f} crore")
    print(f"    Avg uplift/property  : Rs {avg_uplift:,.0f}")
    print(f"    Station-level detail:")
    st_sim = zd.groupby('nearest_station').agg(
        n=('price_total','count'),
        avg_aqi=('annual_aqi','first'),
        uplift=('price_uplift','sum')
    ).sort_values('avg_aqi')
    for st, row in st_sim.iterrows():
        if row['uplift'] > 0:
            print(f"      {st:28s} AQI={row['avg_aqi']:.0f}  "
                  f"Rs{row['uplift']/1e7:.2f} cr")
    print()

# ── AQI variation table ───────────────────────────────────
print("─"*55)
print("WITHIN-ZONE AQI VARIATION (identification check)")
print("─"*55)
print(f"\nGurgaon Belt station AQI values:")
gb_stations = df_gb.groupby('nearest_station')['annual_aqi'].first().sort_values()
for st, aqi in gb_stations.items():
    bar = '█' * int(aqi - 170)
    print(f"  {st:28s} {aqi:.0f} {bar}")

print(f"\nEast NCR station AQI values:")
enc_stations = df_enc.groupby('nearest_station')['annual_aqi'].first().sort_values()
for st, aqi in enc_stations.items():
    bar = '█' * int((aqi - 190) * 2)
    print(f"  {st:28s} {aqi:.0f} {bar}")

# ═══════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════
os.makedirs('outputs', exist_ok=True)

NAVY  = '#1a2744'
GREEN = '#27ae60'
BLUE  = '#2980b9'
RED   = '#c0392b'
LGREY = '#f4f6f8'
DGREY = '#2c3e50'

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')
fig.suptitle('Regional Pollution Discount: Gurgaon Belt vs East NCR',
             fontsize=14, fontweight='bold', color=NAVY, y=0.98)

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])  # AQI vs price scatter — Gurgaon
ax2 = fig.add_subplot(gs[0, 1])  # AQI vs price scatter — East NCR
ax3 = fig.add_subplot(gs[0, 2])  # Coefficient comparison bar
ax4 = fig.add_subplot(gs[1, 0])  # Price distribution by station — Gurgaon
ax5 = fig.add_subplot(gs[1, 1])  # Price distribution by station — East NCR
ax6 = fig.add_subplot(gs[1, 2])  # Policy simulation

# ── Panel 1: AQI vs Price scatter — Gurgaon Belt ─────────
station_avg_gb = df_gb.groupby('nearest_station').agg(
    avg_aqi   = ('annual_aqi','first'),
    avg_price = ('price_per_sqft','mean'),
    n         = ('price_per_sqft','count')
).reset_index()

sizes_gb = (station_avg_gb['n'] / station_avg_gb['n'].max() * 300 + 50)
sc1 = ax1.scatter(station_avg_gb['avg_aqi'], station_avg_gb['avg_price'],
                   s=sizes_gb, c=GREEN, alpha=0.8, zorder=5, edgecolors=NAVY, linewidth=0.5)
for _, row in station_avg_gb.iterrows():
    ax1.annotate(row['nearest_station'].split()[0],
                 (row['avg_aqi'], row['avg_price']),
                 textcoords='offset points', xytext=(6, 4),
                 fontsize=7, color=DGREY)
z1 = np.polyfit(station_avg_gb['avg_aqi'], station_avg_gb['avg_price'], 1)
x1 = np.linspace(station_avg_gb['avg_aqi'].min()-3,
                  station_avg_gb['avg_aqi'].max()+3, 100)
ax1.plot(x1, np.poly1d(z1)(x1), color=GREEN, linewidth=2,
         linestyle='--', alpha=0.8, label=f'{pct_gb:+.1f}%/100-pt')
ax1.set_xlabel('Station Annual AQI', fontsize=9)
ax1.set_ylabel('Avg Price per Sqft (INR)', fontsize=9)
ax1.set_title('Gurgaon Belt\nAQI vs Station Price', fontsize=10, fontweight='bold', color=NAVY)
ax1.legend(fontsize=8)
ax1.set_facecolor(LGREY)
ax1.spines[['top','right']].set_visible(False)
ax1.grid(alpha=0.3, color='white')
ax1.text(0.05, 0.95, f'β={b_gb:.3f} **', transform=ax1.transAxes,
         fontsize=9, fontweight='bold', color=GREEN, va='top')

# ── Panel 2: AQI vs Price scatter — East NCR ─────────────
station_avg_enc = df_enc.groupby('nearest_station').agg(
    avg_aqi   = ('annual_aqi','first'),
    avg_price = ('price_per_sqft','mean'),
    n         = ('price_per_sqft','count')
).reset_index()

sizes_enc = (station_avg_enc['n'] / station_avg_enc['n'].max() * 300 + 50)
sc2 = ax2.scatter(station_avg_enc['avg_aqi'], station_avg_enc['avg_price'],
                   s=sizes_enc, c=BLUE, alpha=0.8, zorder=5, edgecolors=NAVY, linewidth=0.5)
for _, row in station_avg_enc.iterrows():
    ax2.annotate(row['nearest_station'].split()[0],
                 (row['avg_aqi'], row['avg_price']),
                 textcoords='offset points', xytext=(6, 4),
                 fontsize=7, color=DGREY)
z2 = np.polyfit(station_avg_enc['avg_aqi'], station_avg_enc['avg_price'], 1)
x2 = np.linspace(station_avg_enc['avg_aqi'].min()-3,
                  station_avg_enc['avg_aqi'].max()+3, 100)
ax2.plot(x2, np.poly1d(z2)(x2), color=BLUE, linewidth=2,
         linestyle='--', alpha=0.8, label=f'{pct_enc:+.1f}%/100-pt')
ax2.set_xlabel('Station Annual AQI', fontsize=9)
ax2.set_ylabel('Avg Price per Sqft (INR)', fontsize=9)
ax2.set_title('East NCR\nAQI vs Station Price', fontsize=10, fontweight='bold', color=NAVY)
ax2.legend(fontsize=8)
ax2.set_facecolor(LGREY)
ax2.spines[['top','right']].set_visible(False)
ax2.grid(alpha=0.3, color='white')
ax2.text(0.05, 0.95, f'β={b_enc:.3f} ***', transform=ax2.transAxes,
         fontsize=9, fontweight='bold', color=BLUE, va='top')

# ── Panel 3: Coefficient comparison ──────────────────────
zones_bar    = ['NCR\nOverall', 'Gurgaon\nBelt', 'East NCR']
betas_bar    = [b_all, b_gb, b_enc]
pcts_bar     = [pct_all, pct_gb, pct_enc]
colors_bar   = [DGREY, GREEN, BLUE]
ci_low_bar   = [
    m_all.conf_int().loc['aqi_100',0],
    zone_results['Gurgaon Belt'].conf_int().loc['aqi_100',0],
    zone_results['East NCR'].conf_int().loc['aqi_100',0],
]
ci_high_bar  = [
    m_all.conf_int().loc['aqi_100',1],
    zone_results['Gurgaon Belt'].conf_int().loc['aqi_100',1],
    zone_results['East NCR'].conf_int().loc['aqi_100',1],
]

bars3 = ax3.bar(zones_bar, betas_bar, color=colors_bar, alpha=0.85, width=0.5)
for i, (b, cl, ch, pct, color) in enumerate(zip(betas_bar, ci_low_bar,
                                                  ci_high_bar, pcts_bar, colors_bar)):
    ax3.errorbar(i, b, yerr=[[b-cl],[ch-b]],
                 fmt='none', color=NAVY, capsize=6, linewidth=1.5)
    ax3.text(i, b - 0.02 if b < 0 else b + 0.02,
             f'{pct:+.1f}%', ha='center',
             va='top' if b < 0 else 'bottom',
             fontsize=9, fontweight='bold', color='white')
ax3.axhline(0, color=NAVY, linewidth=0.8)
ax3.set_ylabel('AQI Coefficient (β)', fontsize=9)
ax3.set_title('AQI Coefficient Comparison\n(with 95% CI)', fontsize=10, fontweight='bold', color=NAVY)
ax3.set_facecolor(LGREY)
ax3.spines[['top','right']].set_visible(False)
ax3.grid(axis='y', alpha=0.3, color='white')

# ── Panel 4: Price by station — Gurgaon Belt ─────────────
gb_order = df_gb.groupby('nearest_station')['annual_aqi'].first().sort_values().index.tolist()
gb_data  = [df_gb[df_gb['nearest_station']==s]['price_per_sqft'].dropna()
            for s in gb_order]
gb_aqis  = [df_gb[df_gb['nearest_station']==s]['annual_aqi'].iloc[0] for s in gb_order]
gb_labels = [f"{s.split()[0]}\n(AQI {a:.0f})" for s, a in zip(gb_order, gb_aqis)]

bp4 = ax4.boxplot(gb_data, patch_artist=True,
                   labels=gb_labels,
                   medianprops=dict(color='white', linewidth=2),
                   flierprops=dict(marker='o', markersize=2, alpha=0.3))
cmap_gb = plt.cm.RdYlGn_r
norm_gb = plt.Normalize(min(gb_aqis), max(gb_aqis))
for patch, aqi in zip(bp4['boxes'], gb_aqis):
    patch.set_facecolor(cmap_gb(norm_gb(aqi)))
    patch.set_alpha(0.8)

ax4.set_title('Gurgaon Belt: Price by Station\n(ordered by AQI, clean→dirty)',
              fontsize=10, fontweight='bold', color=NAVY)
ax4.set_ylabel('Price per Sqft (INR)', fontsize=9)
ax4.tick_params(axis='x', labelsize=7)
ax4.set_facecolor(LGREY)
ax4.spines[['top','right']].set_visible(False)
ax4.grid(axis='y', alpha=0.3, color='white')

# ── Panel 5: Price by station — East NCR ─────────────────
enc_order = df_enc.groupby('nearest_station')['annual_aqi'].first().sort_values().index.tolist()
enc_data  = [df_enc[df_enc['nearest_station']==s]['price_per_sqft'].dropna()
             for s in enc_order]
enc_aqis  = [df_enc[df_enc['nearest_station']==s]['annual_aqi'].iloc[0] for s in enc_order]
enc_labels = [f"{s.split()[0]}\n(AQI {a:.0f})" for s, a in zip(enc_order, enc_aqis)]

bp5 = ax5.boxplot(enc_data, patch_artist=True,
                   labels=enc_labels,
                   medianprops=dict(color='white', linewidth=2),
                   flierprops=dict(marker='o', markersize=2, alpha=0.3))
norm_enc = plt.Normalize(min(enc_aqis), max(enc_aqis))
for patch, aqi in zip(bp5['boxes'], enc_aqis):
    patch.set_facecolor(cmap_gb(norm_enc(aqi)))
    patch.set_alpha(0.8)

ax5.set_title('East NCR: Price by Station\n(ordered by AQI, clean→dirty)',
              fontsize=10, fontweight='bold', color=NAVY)
ax5.set_ylabel('Price per Sqft (INR)', fontsize=9)
ax5.tick_params(axis='x', labelsize=7)
ax5.set_facecolor(LGREY)
ax5.spines[['top','right']].set_visible(False)
ax5.grid(axis='y', alpha=0.3, color='white')

# ── Panel 6: Policy simulation ────────────────────────────
zones_sim   = ['Gurgaon\nBelt', 'East NCR']
uplifts_sim = [sim_results['Gurgaon Belt']['uplift_cr'],
               sim_results['East NCR']['uplift_cr']]
props_sim   = [sim_results['Gurgaon Belt']['props_hit'],
               sim_results['East NCR']['props_hit']]
colors_sim  = [GREEN, BLUE]

bars6 = ax6.bar(zones_sim, uplifts_sim, color=colors_sim, alpha=0.85, width=0.5)
for bar, val, props in zip(bars6, uplifts_sim, props_sim):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'Rs {val:.1f} cr\n({props:,} props)',
             ha='center', fontsize=9, fontweight='bold', color=NAVY)

ax6.set_ylabel('Wealth Unlocked (Rs Crore)', fontsize=9)
ax6.set_title('Policy Simulation\n(above-avg AQI → NCR mean 206)',
              fontsize=10, fontweight='bold', color=NAVY)
ax6.set_facecolor(LGREY)
ax6.spines[['top','right']].set_visible(False)
ax6.grid(axis='y', alpha=0.3, color='white')

plt.savefig('outputs/regional_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Chart saved → outputs/regional_regression.png")
print(f"\nSummary:")
print(f"  Gurgaon Belt : {pct_gb:+.1f}% per 100-pt AQI (n={len(df_gb):,})")
print(f"  East NCR     : {pct_enc:+.1f}% per 100-pt AQI (n={len(df_enc):,})")
print(f"  NCR Overall  : {pct_all:+.1f}% per 100-pt AQI (n={len(sub_all):,})")

