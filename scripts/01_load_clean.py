import pandas as pd
import numpy as np
import os

print("=" * 55)
print("STAGE 1 — Load and clean NCR property data")
print("=" * 55)

df = pd.read_csv('data/property/Delhi_v2.csv', index_col=0)
print(f"\nRaw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Drop sparse/noisy columns ──────────────────────────────
drop_cols = ['parking', 'Lift', 'Landmarks', 'desc']
df = df.drop(columns=drop_cols)

# ── Extract city ───────────────────────────────────────────
def extract_city(address):
    if pd.isna(address): return 'Unknown'
    if 'New Delhi' in address: return 'New Delhi'
    if 'Gurugram' in address: return 'Gurgaon'
    parts = [p.strip() for p in address.split(',')]
    return parts[-2] if len(parts) >= 2 else parts[0]

df['city'] = df['Address'].apply(extract_city)

# ── Merge Gurgaon sub-labels ───────────────────────────────
df['city'] = df['city'].replace({
    'Gurgaon - North': 'Gurgaon',
    'Gurgaon - South': 'Gurgaon'
})

# ── Fill missing values BEFORE renaming ───────────────────
df['Furnished_status'] = df['Furnished_status'].fillna(
    df['Furnished_status'].mode()[0])
df['Status'] = df['Status'].fillna(df['Status'].mode()[0])
df['Balcony'] = df['Balcony'].fillna(df['Balcony'].median())

# ── Rename ─────────────────────────────────────────────────
df = df.rename(columns={
    'area':             'area_sqft',
    'Bedrooms':         'bhk',
    'Bathrooms':        'bathrooms',
    'Balcony':          'balcony',
    'Furnished_status': 'furnishing',
    'Status':           'status',
    'neworold':         'transaction',
    'type_of_building': 'property_type',
    'Price_sqft':       'price_per_sqft',
    'price':            'price_total',
    'Address':          'address',
})

# ── Remove outliers ────────────────────────────────────────
q_low  = df['price_per_sqft'].quantile(0.01)
q_high = df['price_per_sqft'].quantile(0.99)
before = len(df)
df = df[(df['price_per_sqft'] >= q_low) &
        (df['price_per_sqft'] <= q_high)].copy()
print(f"\nOutlier removal: {before} → {len(df)} rows")
print(f"price_per_sqft range: ₹{q_low:,.0f} — ₹{q_high:,.0f}")

# ── Verify ─────────────────────────────────────────────────
print(f"\nMissing values: {df.isnull().sum().sum()} total")
print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFinal columns:")
for col in df.columns:
    missing = df[col].isna().sum()
    print(f"  {col:20s}  ({df[col].dtype})" +
          (f"  ← {missing} missing" if missing > 0 else ""))

# ── Save ───────────────────────────────────────────────────
os.makedirs('data/property', exist_ok=True)
df.to_csv('data/property/ncr_clean.csv', index=False)
print(f"\n✓ Saved → data/property/ncr_clean.csv")
