import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

print("=" * 55)
print("STAGE 2B — Monthly AQI Extraction & Matching")
print("=" * 55)

MONTHS = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']
MONTH_NUM = {m: i+1 for i, m in enumerate(MONTHS)}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def parse_monthly_aqi(filepath):
    """
    Parse CPCB Excel pivot table.
    Returns dict: {1: jan_avg, 2: feb_avg, ..., 12: dec_avg}
    Missing months return NaN.
    """
    try:
        df = pd.read_excel(filepath, header=0)
    except Exception as e:
        print(f"  Cannot read {os.path.basename(filepath)}: {e}")
        return {}

    df = df.rename(columns={df.columns[0]: 'day'})
    df = df[df['day'].astype(str).str.strip().str.match(
        r'^0?[1-9]$|^[12][0-9]$|^3[01]$')]

    monthly = {}
    for month_name in MONTHS:
        if month_name not in df.columns:
            monthly[MONTH_NUM[month_name]] = np.nan
            continue
        vals = pd.to_numeric(df[month_name], errors='coerce')
        vals = vals[(vals >= 0) & (vals <= 500)]
        monthly[MONTH_NUM[month_name]] = round(vals.mean(), 2) if len(vals) >= 10 else np.nan

    return monthly

# ── Station list (same as 02_aqi_process.py) ──────────────
STATIONS = [
    ('AQI_daily_2021_Alipur_Delhi_DPCC_2021.xlsx',            'Alipur',               'New Delhi', 28.7989, 77.1530),
    ('AQI_daily_2021_Anand_Vihar_Delhi_DPCC_2021.xlsx',       'Anand Vihar',          'New Delhi', 28.6469, 77.3152),
    ('AQI_daily_2021_Ashok_Vihar_Delhi_DPCC_2021.xlsx',       'Ashok Vihar',          'New Delhi', 28.6928, 77.1833),
    ('AQI_daily_2021_Aya_Nagar_Delhi_IMD_2021.xlsx',          'Aya Nagar',            'New Delhi', 28.4725, 77.1097),
    ('AQI_daily_2021_Bawana_Delhi_DPCC_2021.xlsx',            'Bawana',               'New Delhi', 28.7784, 77.0405),
    ('AQI_daily_2021_Chandni_Chowk_Delhi_IITM_2021.xlsx',     'Chandni Chowk',        'New Delhi', 28.6562, 77.2300),
    ('AQI_daily_2021_CRRI_Mathura_Road_Delhi_IMD_2021.xlsx',  'CRRI Mathura Road',    'New Delhi', 28.5514, 77.2739),
    ('AQI_daily_2021_Dr._Karni_Singh_Shooting_Range_Delhi_DPCC_2021.xlsx', 'Dr Karni Singh', 'New Delhi', 28.4985, 77.3023),
    ('AQI_daily_2021_DTU_Delhi_CPCB_2021.xlsx',               'DTU',                  'New Delhi', 28.7500, 77.1189),
    ('AQI_daily_2021_Dwarka-Sector_8_Delhi_DPCC__2021.xlsx',  'Dwarka Sector 8',      'New Delhi', 28.5825, 77.0460),
    ('AQI_daily_2021_IGI_Airport_(T3)_Delhi_IMD_2021.xlsx',   'IGI Airport T3',       'New Delhi', 28.5562, 77.1000),
    ('AQI_daily_2021_IHBAS_Dilshad_Garden_Delhi_CPCB_2021.xlsx','IHBAS Dilshad Garden','New Delhi', 28.6830, 77.3120),
    ('AQI_daily_2021_ITO_Delhi_CPCB_2021.xlsx',               'ITO',                  'New Delhi', 28.6289, 77.2400),
    ('AQI_daily_2021_Jahangirpuri_Delhi_DPCC_2021.xlsx',      'Jahangirpuri',         'New Delhi', 28.7299, 77.1664),
    ('AQI_daily_2021_Jawaharlal_Nehru_Stadium_Delhi_DPCC_2021.xlsx','JLN Stadium',    'New Delhi', 28.5833, 77.2333),
    ('AQI_daily_2021_Lodhi_Road_Delhi_IITM_2021.xlsx',        'Lodhi Road IITM',      'New Delhi', 28.5933, 77.2271),
    ('AQI_daily_2021_Lodhi_Road_Delhi_IMD_2021.xlsx',         'Lodhi Road IMD',       'New Delhi', 28.5928, 77.2274),
    ('AQI_daily_2021_Major_Dhyan_Chand_National_Stadium_Delhi_DPCC_2021.xlsx','Major Dhyan Chand','New Delhi', 28.6111, 77.2372),
    ('AQI_daily_2021_Mandir_Marg_Delhi_DPCC_2021.xlsx',       'Mandir Marg',          'New Delhi', 28.6378, 77.2019),
    ('AQI_daily_2021_Mundka_Delhi_DPCC_2021.xlsx',            'Mundka',               'New Delhi', 28.6943, 77.0528),
    ('AQI_daily_2021_Najafgarh_Delhi_DPCC_2021.xlsx',         'Najafgarh',            'New Delhi', 28.6092, 76.9794),
    ('AQI_daily_2021_Narela_Delhi_DPCC_2021.xlsx',            'Narela',               'New Delhi', 28.8520, 77.0930),
    ('AQI_daily_2021_Nehru_Nagar_Delhi_DPCC_2021.xlsx',       'Nehru Nagar',          'New Delhi', 28.5682, 77.2502),
    ('AQI_daily_2021_North_Campus_DU_Delhi_IMD_2021.xlsx',    'North Campus DU',      'New Delhi', 28.7041, 77.1926),
    ('AQI_daily_2021_NSIT_Dwarka_Delhi_CPCB_2021.xlsx',       'NSIT Dwarka',          'New Delhi', 28.6083, 77.0321),
    ('AQI_daily_2021_Okhla_Phase-2_Delhi_DPCC_2021.xlsx',     'Okhla Phase 2',        'New Delhi', 28.5300, 77.2800),
    ('AQI_daily_2021_Patparganj_Delhi_DPCC_2021.xlsx',        'Patparganj',           'New Delhi', 28.6282, 77.2963),
    ('AQI_daily_2021_Punjabi_Bagh_Delhi_DPCC_2021.xlsx',      'Punjabi Bagh',         'New Delhi', 28.6726, 77.1312),
    ('AQI_daily_2021_Pusa_Delhi_DPCC_2021.xlsx',              'Pusa DPCC',            'New Delhi', 28.6381, 77.1497),
    ('AQI_daily_2021_Pusa_Delhi_IMD_2021.xlsx',               'Pusa IMD',             'New Delhi', 28.6400, 77.1520),
    ('AQI_daily_2021_R_K_Puram_Delhi_DPCC_2021.xlsx',         'RK Puram',             'New Delhi', 28.5665, 77.1764),
    ('AQI_daily_2021_Rohini_Delhi_DPCC_2021.xlsx',            'Rohini',               'New Delhi', 28.7041, 77.1172),
    ('AQI_daily_2021_Shadipur_Delhi_CPCB_2021.xlsx',          'Shadipur',             'New Delhi', 28.6512, 77.1507),
    ('AQI_daily_2021_Sirifort_Delhi_CPCB_2021.xlsx',          'Sirifort',             'New Delhi', 28.5496, 77.2167),
    ('AQI_daily_2021_Sonia_Vihar_Delhi_DPCC_2021.xlsx',       'Sonia Vihar',          'New Delhi', 28.7158, 77.2618),
    ('AQI_daily_2021_Sri_Aurobindo_Marg_Delhi_DPCC_2021.xlsx','Sri Aurobindo Marg',   'New Delhi', 28.5407, 77.2011),
    ('AQI_daily_2021_Vivek_Vihar_Delhi_DPCC_2021.xlsx',       'Vivek Vihar',          'New Delhi', 28.6700, 77.3150),
    ('AQI_daily_2021_Wazirpur_Delhi_DPCC_2021.xlsx',          'Wazirpur',             'New Delhi', 28.6975, 77.1637),
    ('AQI_daily_2021_Sector_62_Noida_IMD_2021.xlsx',          'Sector 62 Noida',      'Noida',        28.6270, 77.3730),
    ('AQI_daily_2021_Sector_125_Noida_UPPCB_2021.xlsx',       'Sector 125 Noida',     'Noida',        28.5448, 77.3290),
    ('AQI_daily_2021_Vikas_Sadan_Gurugram_HSPCB_2021.xlsx',   'Vikas Sadan',          'Gurgaon',      28.4500, 77.0260),
    ('AQI_daily_2021_NISE_Gwal_Pahari_Gurugram_IMD_2021.xlsx','Gwal Pahari',          'Gurgaon',      28.4200, 77.1500),
    ('AQI_daily_2021_Sanjay_Nagar_Ghaziabad_UPPCB_2021.xlsx', 'Sanjay Nagar',         'Ghaziabad',    28.6670, 77.4420),
    ('AQI_daily_2021_Loni_Ghaziabad_UPPCB_2021.xlsx',         'Loni',                 'Ghaziabad',    28.7500, 77.2890),
    ('AQI_daily_2021_Vasundhara_Ghaziabad_UPPCB_2021.xlsx',   'Vasundhara',           'Ghaziabad',    28.6590, 77.3650),
    ('AQI_daily_2021_Knowledge_Park_III_Greater_Noida_UPPCB_2021.xlsx','Knowledge Park III','Greater Noida',28.4744,77.5040),
    ('AQI_daily_2021_Sector-_16A_Faridabad_HSPCB_2021.xlsx',  'Sector 16A',           'Faridabad',    28.3670, 77.3120),
]

# ── Step 1: Extract monthly AQI per station ───────────────
print("\nExtracting monthly AQI per station...")
aqi_dir = 'data/aqi'
station_records = []
missing = []

for filename, station_name, city, lat, lon in STATIONS:
    filepath = os.path.join(aqi_dir, filename)
    if not os.path.exists(filepath):
        missing.append(station_name)
        continue

    monthly = parse_monthly_aqi(filepath)
    if not monthly:
        continue

    # Annual average (for reference)
    vals = [v for v in monthly.values() if not np.isnan(v)]
    annual_avg = round(np.mean(vals), 2) if vals else np.nan

    record = {
        'station_name': station_name,
        'city':         city,
        'latitude':     lat,
        'longitude':    lon,
        'annual_aqi':   annual_avg,
    }
    for m_num, avg in monthly.items():
        record[f'aqi_m{m_num:02d}'] = avg

    station_records.append(record)
    print(f"  ✓ {station_name:28s} annual={annual_avg:.0f}  "
          f"Jan={monthly.get(1,np.nan):.0f}  "
          f"Jun={monthly.get(6,np.nan):.0f}  "
          f"Nov={monthly.get(11,np.nan):.0f}")

stations_df = pd.DataFrame(station_records)
print(f"\nStations loaded: {len(stations_df)}")
if missing:
    print(f"Missing: {missing}")

# Save station monthly reference
os.makedirs('data/aqi', exist_ok=True)
stations_df.to_csv('data/aqi/station_monthly_aqi.csv', index=False)
print(f"✓ Station monthly AQI saved → data/aqi/station_monthly_aqi.csv")

# ── Step 2: Load property dataset WITH dates ───────────────
print("\n" + "=" * 55)
print("Matching properties to monthly AQI...")
print("=" * 55)

props = pd.read_csv('data/property/ncr_clean_with_dates.csv')
print(f"Properties loaded: {len(props)}")

# ── Step 3: Nearest station matching ──────────────────────
def find_nearest_station(prop_lat, prop_lon, sdf):
    dists = sdf.apply(
        lambda r: haversine(prop_lat, prop_lon, r['latitude'], r['longitude']),
        axis=1)
    idx = dists.idxmin()
    return idx, round(dists[idx], 2)

print("Finding nearest station for each property...")
station_indices = []
dist_list = []

for _, row in props.iterrows():
    idx, dist = find_nearest_station(row['latitude'], row['longitude'], stations_df)
    station_indices.append(idx)
    dist_list.append(dist)

props['_station_idx']     = station_indices
props['dist_to_station_km'] = dist_list

# Add station info
props['nearest_station'] = props['_station_idx'].map(stations_df['station_name'])
props['station_lat']     = props['_station_idx'].map(stations_df['latitude'])
props['station_lon']     = props['_station_idx'].map(stations_df['longitude'])
props['annual_aqi']      = props['_station_idx'].map(stations_df['annual_aqi'])

# ── Step 4: Monthly AQI matching ──────────────────────────
# For each property, get the AQI of its listing month
def get_monthly_aqi(station_idx, listing_month):
    col = f'aqi_m{int(listing_month):02d}'
    if col in stations_df.columns:
        val = stations_df.loc[int(station_idx), col]
        # Fall back to annual if monthly is NaN
        if np.isnan(val):
            return stations_df.loc[int(station_idx), 'annual_aqi']
        return val
    return stations_df.loc[int(station_idx), 'annual_aqi']

props['monthly_aqi'] = props.apply(
    lambda r: get_monthly_aqi(r['_station_idx'], r['listing_month']),
    axis=1)

# Drop helper column
props = props.drop(columns=['_station_idx'])

# ── Step 5: Validation ────────────────────────────────────
print("\n" + "=" * 55)
print("Validation — Monthly AQI Distribution")
print("=" * 55)

MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

monthly_stats = props.groupby('listing_month').agg(
    n_props      = ('monthly_aqi', 'count'),
    avg_monthly  = ('monthly_aqi', 'mean'),
    avg_annual   = ('annual_aqi',  'mean')
).round(1)
monthly_stats.index = monthly_stats.index.map(MONTH_NAMES)
print("\nAvg AQI by listing month (monthly vs annual):")
print(monthly_stats.to_string())

print(f"\nMonthly AQI range: {props['monthly_aqi'].min():.0f} — {props['monthly_aqi'].max():.0f}")
print(f"Annual AQI range:  {props['annual_aqi'].min():.0f} — {props['annual_aqi'].max():.0f}")
print(f"\nKey insight: Monthly AQI has MUCH wider range than annual average")
print(f"  This gives the regression more variation to work with")

print(f"\nSeason breakdown:")
season_aqi = props.groupby('listing_season')['monthly_aqi'].agg(['mean','min','max']).round(1)
print(season_aqi.to_string())

# ── Save ──────────────────────────────────────────────────
props.to_csv('data/property/ncr_dated_monthly_aqi.csv', index=False)
print(f"\n✓ Saved → data/property/ncr_dated_monthly_aqi.csv")
print(f"  {props.shape[0]} rows × {props.shape[1]} columns")
print(f"\nKey new columns: monthly_aqi, annual_aqi, nearest_station,")
print(f"  dist_to_station_km, listing_month, listing_season")

