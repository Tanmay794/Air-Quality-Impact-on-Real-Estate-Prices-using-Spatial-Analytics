import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
except ImportError:
    print("Run: pip install statsmodels")
    exit()

try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("libpysal not found — will install or use manual Moran's I")

print("=" * 55)
print("STAGE 4 — Spatial Autocorrelation Check")
print("=" * 55)

# ── Load data and refit Model 2 to get residuals ──────────
df = pd.read_csv('data/property/ncr_aqi_matched_merged.csv')

# Rebuild features (same as Stage 3)
df['log_price_sqft']  = np.log(df['price_per_sqft'])
df['aqi_100']         = df['annual_aqi'] / 100
df['log_area']        = np.log(df['area_sqft'].clip(lower=1))
df['furnishing_score'] = df['furnishing'].map({
    'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}).fillna(1)
df['is_new'] = (df['transaction'].str.lower().str.strip() == 'new property').astype(int)

features = ['aqi_100', 'log_area', 'furnishing_score', 'is_new']
cols     = ['log_price_sqft'] + features
sub      = df[cols + ['latitude','longitude','nearest_station']].dropna()
sub      = sub.apply(lambda c: pd.to_numeric(c, errors='coerce')
                     if c.name not in ['nearest_station'] else c).dropna()

X = sm.add_constant(sub[features].astype(float))
y = sub['log_price_sqft'].astype(float)
model = sm.OLS(y, X).fit(cov_type='HC3')

sub = sub.copy()
sub['residual'] = model.resid

print(f"\nModel refitted: {len(sub):,} observations")
print(f"Residual stats:")
print(f"  Mean:  {sub['residual'].mean():.6f}  (should be ~0)")
print(f"  Std:   {sub['residual'].std():.4f}")
print(f"  Min:   {sub['residual'].min():.4f}")
print(f"  Max:   {sub['residual'].max():.4f}")

# ── Approach: aggregate residuals to station level ────────
# 62% of properties share lat/lon — point-level Moran's I
# is unreliable with duplicate coordinates.
# Solution: compute mean residual per station (47 unique points)
# and test spatial autocorrelation at station level.
print("\n" + "=" * 55)
print("Station-level residual aggregation")
print("=" * 55)

station_resids = sub.groupby('nearest_station').agg(
    mean_resid = ('residual', 'mean'),
    n_props    = ('residual', 'count'),
    lat        = ('latitude', 'mean'),
    lon        = ('longitude', 'mean')
).reset_index()

# Use station coordinates from Stage 2 output for precision
aqi_matched = pd.read_csv('data/property/ncr_aqi_matched.csv')
station_coords = aqi_matched.groupby('nearest_station').agg(
    lat = ('station_lat', 'first'),
    lon = ('station_lon', 'first')
).reset_index()

station_resids = station_resids.drop(columns=['lat','lon']).merge(
    station_coords, on='nearest_station', how='left')

print(f"\nStation-level residuals ({len(station_resids)} stations):")
print(station_resids[['nearest_station','n_props','mean_resid','lat','lon']]
      .sort_values('mean_resid').to_string(index=False))

# ── Manual Moran's I (works without libpysal) ─────────────
print("\n" + "=" * 55)
print("Moran's I — Spatial Autocorrelation Test")
print("=" * 55)

def compute_morans_i(values, lats, lons, k=5):
    """
    Compute Moran's I using k-nearest-neighbour weights.
    Uses inverse distance weighting among k neighbours.
    """
    from math import radians, sin, cos, sqrt, atan2

    n = len(values)
    coords = list(zip(lats, lons))

    def haversine(a, b):
        R = 6371
        la1, lo1 = map(radians, a)
        la2, lo2 = map(radians, b)
        dlat, dlon = la2-la1, lo2-lo1
        h = sin(dlat/2)**2 + cos(la1)*cos(la2)*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(h), sqrt(1-h))

    # Build k-NN weight matrix
    W = np.zeros((n, n))
    for i in range(n):
        dists = [(haversine(coords[i], coords[j]), j)
                 for j in range(n) if j != i]
        dists.sort()
        neighbours = dists[:k]
        for dist, j in neighbours:
            W[i, j] = 1.0 / max(dist, 0.001)  # inverse distance

    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # Moran's I formula
    z = np.array(values) - np.mean(values)
    S0 = W.sum()
    numerator   = n * (z @ W @ z)
    denominator = S0 * (z @ z)
    I = numerator / denominator

    # Expected value and variance under normality assumption
    E_I  = -1 / (n - 1)
    S1   = 0.5 * np.sum((W + W.T)**2)
    S2   = np.sum((W.sum(axis=1) + W.sum(axis=0))**2)
    n2   = n * n
    var_I = (n2*(n+1)*S1 - n*(S1+3*S0**2) + 3*S0**2) / \
            ((n-1)*(n+1)*(n-1)*S0**2) - E_I**2

    z_score = (I - E_I) / np.sqrt(var_I)
    p_value = 2 * (1 - _norm_cdf(abs(z_score)))

    return I, E_I, z_score, p_value

def _norm_cdf(x):
    """Standard normal CDF approximation"""
    return 0.5 * (1 + np.sign(x) * (
        1 - np.exp(-2*x*x/np.pi)**0.56))

# Try with libpysal first, fall back to manual
vals = station_resids['mean_resid'].values
lats = station_resids['lat'].values
lons = station_resids['lon'].values
coords_arr = np.column_stack([lats, lons])

if PYSAL_AVAILABLE:
    try:
        w = KNN.from_array(coords_arr, k=5)
        w.transform = 'r'
        mi = Moran(vals, w)
        I, p_val, z_score = mi.I, mi.p_norm, mi.z_norm
        E_I = mi.EI
        print("(Using libpysal KNN weights)")
    except Exception as e:
        print(f"libpysal failed ({e}), using manual calculation")
        PYSAL_AVAILABLE = False

if not PYSAL_AVAILABLE:
    print("(Using manual Moran's I with inverse-distance KNN weights)")
    I, E_I, z_score, p_val = compute_morans_i(vals, lats, lons, k=5)

print(f"\nMoran's I Results (station-level residuals, k=5 neighbours):")
print(f"  Moran's I  = {I:.4f}")
print(f"  Expected I = {E_I:.4f}  (under spatial randomness)")
print(f"  Z-score    = {z_score:.4f}")
print(f"  P-value    = {p_val:.4f}")
print()

if p_val < 0.05:
    if I > E_I:
        print("  RESULT: Significant POSITIVE spatial autocorrelation (p < 0.05)")
        print("  Interpretation: Properties near each other have similar residuals")
        print("  → OLS standard errors may be underestimated")
        print("  → Spatial lag model recommended as robustness check")
    else:
        print("  RESULT: Significant NEGATIVE spatial autocorrelation (p < 0.05)")
        print("  Interpretation: Neighbouring residuals are dissimilar")
else:
    print("  RESULT: No significant spatial autocorrelation (p >= 0.05)")
    print("  → OLS residuals are spatially random")
    print("  → Standard errors are reliable, no spatial model needed")

# ── Residual pattern by city ──────────────────────────────
print("\n" + "=" * 55)
print("Residual pattern by city")
print("=" * 55)
city_resids = sub.merge(
    df[['latitude','longitude','city']],
    on=['latitude','longitude'], how='left')

if 'city' in city_resids.columns:
    cr = city_resids.groupby('city')['residual'].agg(['mean','std','count'])
    cr.columns = ['mean_resid', 'std_resid', 'n']
    cr = cr.round(4)
    print(cr.to_string())
    print()
    print("Interpretation:")
    print("  Positive mean residual = OLS underpredicts prices in that city")
    print("  Negative mean residual = OLS overpredicts prices in that city")
    print("  Large std = high unexplained variation within city")

# ── Save residuals ────────────────────────────────────────
df_final = pd.read_csv('data/property/ncr_final_merged.csv')
# Merge residuals back on index
sub_out = sub[['residual']].copy()
sub_out.index = sub.index
df_final['ols_residual'] = np.nan
df_final.loc[sub_out.index, 'ols_residual'] = sub_out['residual'].values
df_final.to_csv('data/property/ncr_final_merged.csv', index=False)

print(f"\n✓ Residuals saved → data/property/ncr_final_merged.csv")
print(f"  New column: ols_residual")

