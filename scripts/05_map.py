import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    import branca.colormap as cm
except ImportError:
    print("Run: /Users/tanmaychowdhary/anaconda3/bin/pip install folium branca")
    exit()

print("=" * 55)
print("STAGE 5 — Interactive NCR Map")
print("=" * 55)

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv('data/property/ncr_final_merged.csv')
print(f"\nLoaded: {len(df):,} properties")

# ── Station-level summary ─────────────────────────────────
stations = df.groupby('nearest_station').agg(
    lat         = ('station_lat', 'first'),
    lon         = ('station_lon', 'first'),
    annual_aqi  = ('annual_aqi', 'first'),
    n_props     = ('price_per_sqft', 'count'),
    avg_price   = ('price_per_sqft', 'mean'),
    city        = ('city', 'first')
).reset_index()

print(f"Stations: {len(stations)}")

# ── Colour scales ─────────────────────────────────────────
# AQI: green (clean) → red (polluted)
aqi_cmap = cm.LinearColormap(
    colors=['#2ecc71', '#f39c12', '#e74c3c'],
    vmin=stations['annual_aqi'].min(),
    vmax=stations['annual_aqi'].max(),
    caption='Annual AQI 2021 (higher = more polluted)'
)

# Price: light → dark blue
price_cmap = cm.LinearColormap(
    colors=['#d6eaf8', '#2980b9', '#1a252f'],
    vmin=df['price_per_sqft'].quantile(0.05),
    vmax=df['price_per_sqft'].quantile(0.95),
    caption='Price per sqft (₹)'
)

# ── Base map ──────────────────────────────────────────────
ncr_center = [28.58, 77.20]
m = folium.Map(
    location=ncr_center,
    zoom_start=11,
    tiles='CartoDB positron',
    control_scale=True
)

# ── Layer 1: Property heatmap (price intensity) ───────────
heat_data = df[['latitude', 'longitude', 'price_per_sqft']].dropna()
heat_data = heat_data[
    (heat_data['latitude'].between(28.2, 28.9)) &
    (heat_data['longitude'].between(76.8, 77.6))
]
heat_list = heat_data[['latitude', 'longitude', 'price_per_sqft']].values.tolist()

heatmap_layer = folium.FeatureGroup(name='Property Price Heatmap', show=True)
HeatMap(
    heat_list,
    min_opacity=0.3,
    max_zoom=13,
    radius=15,
    blur=20,
    gradient={'0.2': '#d6eaf8', '0.5': '#2980b9', '0.8': '#1a252f'}
).add_to(heatmap_layer)
heatmap_layer.add_to(m)

# ── Layer 2: AQI station circles ─────────────────────────
aqi_layer = folium.FeatureGroup(name='AQI Stations (circle = pollution level)', show=True)

for _, row in stations.iterrows():
    aqi_val   = row['annual_aqi']
    color     = aqi_cmap(aqi_val)
    radius    = 400 + (aqi_val - 150) * 8  # bigger circle = more polluted

    # Outer circle: AQI level
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=18,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        weight=2,
        popup=folium.Popup(
            f"""
            <div style="font-family:Arial; width:220px">
              <b style="font-size:14px">{row['nearest_station']}</b><br>
              <hr style="margin:4px 0">
              <b>City:</b> {row['city']}<br>
              <b>Annual AQI 2021:</b> {aqi_val:.0f}<br>
              <b>Avg Price/sqft:</b> ₹{row['avg_price']:,.0f}<br>
              <b>Properties matched:</b> {row['n_props']:,}<br>
              <hr style="margin:4px 0">
              <small style="color:#888">
              AQI &lt;100: Good | 100-200: Moderate<br>
              200-300: Poor | &gt;300: Very Poor
              </small>
            </div>
            """,
            max_width=250
        ),
        tooltip=f"{row['nearest_station']} — AQI: {aqi_val:.0f}"
    ).add_to(aqi_layer)

    # Station label
    folium.Marker(
        location=[row['lat'], row['lon']],
        icon=folium.DivIcon(
            html=f'<div style="font-size:8px;font-weight:bold;color:#2c3e50;'
                 f'white-space:nowrap;margin-top:-8px;margin-left:20px">'
                 f'{row["nearest_station"].split()[0]}</div>',
            icon_size=(100, 20)
        )
    ).add_to(aqi_layer)

aqi_layer.add_to(m)

# ── Layer 3: Individual properties (sampled) ──────────────
# Sample max 800 for performance
prop_layer = folium.FeatureGroup(name='Individual Properties (sample)', show=False)
sample = df.sample(min(800, len(df)), random_state=42).dropna(
    subset=['latitude','longitude','price_per_sqft','annual_aqi'])

for _, row in sample.iterrows():
    price = row['price_per_sqft']
    aqi   = row['annual_aqi']
    color = price_cmap(np.clip(price,
                               df['price_per_sqft'].quantile(0.05),
                               df['price_per_sqft'].quantile(0.95)))
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        weight=0,
        popup=folium.Popup(
            f"""
            <div style="font-family:Arial; width:200px">
              <b>{row.get('address','')[:40]}...</b><br>
              <hr style="margin:4px 0">
              <b>Price/sqft:</b> ₹{price:,.0f}<br>
              <b>Area:</b> {row['area_sqft']:.0f} sqft<br>
              <b>BHK:</b> {row['bhk']:.0f}<br>
              <b>Nearest station:</b> {row['nearest_station']}<br>
              <b>Station AQI:</b> {aqi:.0f}<br>
              <b>City:</b> {row['city']}
            </div>
            """,
            max_width=220
        )
    ).add_to(prop_layer)

prop_layer.add_to(m)

# ── Layer 4: City boundary labels ────────────────────────
city_centers = {
    'New Delhi':    [28.6139, 77.2090],
    'Gurgaon':      [28.4595, 77.0266],
    'Noida':        [28.5355, 77.3910],
    'Ghaziabad':    [28.6692, 77.4538],
    'Greater Noida':[28.4744, 77.5040],
    'Faridabad':    [28.4089, 77.3178],
}
city_layer = folium.FeatureGroup(name='City Labels', show=True)
city_aqi   = df.groupby('city')['annual_aqi'].mean().round(0)
city_price = df.groupby('city')['price_per_sqft'].mean().round(0)

for city, coords in city_centers.items():
    aqi_c   = city_aqi.get(city, 0)
    price_c = city_price.get(city, 0)
    folium.Marker(
        location=coords,
        icon=folium.DivIcon(
            html=f'''
            <div style="
              background:rgba(255,255,255,0.92);
              border:1px solid #bdc3c7;
              border-radius:6px;
              padding:4px 8px;
              font-family:Arial;
              font-size:11px;
              font-weight:bold;
              color:#2c3e50;
              white-space:nowrap;
              box-shadow:1px 1px 3px rgba(0,0,0,0.2)
            ">
              {city}<br>
              <span style="font-weight:normal;color:#7f8c8d;font-size:10px">
                AQI: {aqi_c:.0f} | ₹{price_c:,.0f}/sqft
              </span>
            </div>''',
            icon_size=(160, 40),
            icon_anchor=(80, 20)
        )
    ).add_to(city_layer)

city_layer.add_to(m)

# ── Add colourmap legends ─────────────────────────────────
aqi_cmap.add_to(m)

# ── Layer control ─────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)

# ── Title box ─────────────────────────────────────────────
title_html = '''
<div style="
  position: fixed;
  top: 10px; left: 50%; transform: translateX(-50%);
  z-index: 1000;
  background: rgba(255,255,255,0.95);
  border: 1px solid #bdc3c7;
  border-radius: 8px;
  padding: 10px 20px;
  font-family: Arial;
  text-align: center;
  box-shadow: 2px 2px 6px rgba(0,0,0,0.15)
">
  <b style="font-size:15px; color:#2c3e50">
    NCR Air Quality vs Property Prices — 2021
  </b><br>
  <span style="font-size:11px; color:#7f8c8d">
    47 CPCB stations · 11,499 properties · Hedonic pricing regression
  </span>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# ── Research finding annotation ───────────────────────────
finding_html = '''
<div style="
  position: fixed;
  bottom: 30px; left: 10px;
  z-index: 1000;
  background: rgba(255,255,255,0.95);
  border-left: 4px solid #e74c3c;
  border-radius: 4px;
  padding: 8px 12px;
  font-family: Arial;
  font-size: 11px;
  max-width: 280px;
  box-shadow: 2px 2px 6px rgba(0,0,0,0.15)
">
  <b style="color:#e74c3c">Key Finding</b><br>
   A 100-point increase in annual AQI is associated with a
  <b>31.2% reduction</b> in property price per sqft
  (p &lt; 0.001, N = 11,499)<br><br>
  <b>Policy implication:</b> Reducing above-average AQI zones
  to NCR average would unlock <b>&#8377;975 crore</b> in property wealth,
  with Ghaziabad accounting for &#8377;22 crore.
</div>
'''
m.get_root().html.add_child(folium.Element(finding_html))

# ── Save ──────────────────────────────────────────────────
import os
os.makedirs('outputs', exist_ok=True)
output_path = 'outputs/ncr_aqi_property_map_merged.html'
m.save(output_path)

print(f"\n✓ Map saved → {output_path}")
print(f"\nMap layers:")
print(f"  1. Property Price Heatmap — price intensity across NCR")
print(f"  2. AQI Stations — circles coloured by pollution level")
print(f"     (green=clean, orange=moderate, red=polluted)")
print(f"  3. Individual Properties — 800 sampled, click for details")
print(f"  4. City Labels — AQI and avg price per city")
print(f"\nOpen in browser: open {output_path}")

