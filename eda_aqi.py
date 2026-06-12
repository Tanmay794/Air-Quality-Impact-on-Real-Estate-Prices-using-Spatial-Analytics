"""
AQI Daily 2021 – NCR Region
Exploratory Data Analysis
"""

# ── 0. Imports ──────────────────────────────────────────────────────────────
import glob
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

OUTPUT = "plots_aqi"
import os; os.makedirs(OUTPUT, exist_ok=True)

# ── 1. Load & combine all station files ─────────────────────────────────────
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

def load_station(path: str) -> pd.DataFrame:
    """Read one station xlsx and return a tidy (date, aqi) series."""
    fname = os.path.basename(path)
    # Extract station name from filename
    m = re.search(r"AQI_daily_2021_(.*?)_\d{4}", fname)
    station = m.group(1).replace("_", " ") if m else fname
    # Detect city
    if "Delhi" in fname:
        city = "Delhi"
    elif "Noida" in fname:
        city = "Noida"
    elif "Ghaziabad" in fname:
        city = "Ghaziabad"
    elif "Gurugram" in fname:
        city = "Gurugram"
    elif "Faridabad" in fname:
        city = "Faridabad"
    elif "Greater_Noida" in fname:
        city = "Greater Noida"
    else:
        city = "Other"

    df = pd.read_excel(path, usecols=["Date"] + MONTHS)
    # Melt wide → long
    df = df.melt(id_vars="Date", var_name="Month", value_name="AQI")
    df = df.dropna(subset=["AQI"])
    df["Month_num"] = pd.to_datetime(df["Month"], format="%B").dt.month
    df["Date"] = df["Date"].astype(int)
    df["date"] = pd.to_datetime(
        dict(year=2021, month=df["Month_num"], day=df["Date"]),
        errors="coerce"
    )
    df = df.dropna(subset=["date"])
    df["station"] = station
    df["city"] = city
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    return df[["date", "station", "city", "AQI"]]


files = glob.glob("aqi/*.xlsx")
# drop duplicate Knowledge Park file
files = [f for f in files if "(1)" not in f]

frames = [load_station(f) for f in files]
df = pd.concat(frames, ignore_index=True)
df = df.sort_values("date").reset_index(drop=True)

# Derived time features
df["month"]   = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")
df["week"]    = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter
df["dayofweek"] = df["date"].dt.day_name()

# AQI category (CPCB scale)
def aqi_category(v):
    if v <= 50:   return "Good"
    if v <= 100:  return "Satisfactory"
    if v <= 200:  return "Moderate"
    if v <= 300:  return "Poor"
    if v <= 400:  return "Very Poor"
    return "Severe"

df["category"] = df["AQI"].apply(aqi_category)
CAT_ORDER = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
CAT_COLORS = ["#2ecc71","#a8d84a","#f39c12","#e74c3c","#9b59b6","#34495e"]

print("=" * 60)
print("AQI DATASET  –  OVERVIEW")
print("=" * 60)
print(f"  Stations       : {df['station'].nunique()}")
print(f"  Cities         : {df['city'].nunique()}  →  {sorted(df['city'].unique())}")
print(f"  Date range     : {df['date'].min().date()}  to  {df['date'].max().date()}")
print(f"  Total records  : {len(df):,}")
print(f"  Missing AQI    : {df['AQI'].isna().sum():,}")
print()
print(df.groupby("city")["AQI"].describe().round(1))

# ── 2. Distribution of AQI ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["AQI"].dropna(), bins=60, color="#3498db", edgecolor="white", linewidth=0.4)
axes[0].set_title("AQI Distribution – All Stations 2021", fontsize=13)
axes[0].set_xlabel("AQI"); axes[0].set_ylabel("Frequency")
axes[0].axvline(df["AQI"].mean(),  color="red",    linestyle="--", label=f"Mean  {df['AQI'].mean():.0f}")
axes[0].axvline(df["AQI"].median(),color="orange", linestyle="--", label=f"Median {df['AQI'].median():.0f}")
axes[0].legend()

stats.probplot(df["AQI"].dropna(), plot=axes[1])
axes[1].set_title("Q-Q Plot (AQI vs Normal)", fontsize=13)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/01_aqi_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] 01_aqi_distribution.png")

# ── 3. AQI Category breakdown ────────────────────────────────────────────────
cat_counts = df["category"].value_counts().reindex(CAT_ORDER).fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(cat_counts.index, cat_counts.values, color=CAT_COLORS, edgecolor="white")
axes[0].set_title("AQI Category Frequency (all stations)", fontsize=13)
axes[0].set_ylabel("Count"); axes[0].tick_params(axis='x', rotation=20)

axes[1].pie(cat_counts.values, labels=cat_counts.index, colors=CAT_COLORS,
            autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9})
axes[1].set_title("AQI Category Share", fontsize=13)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/02_aqi_categories.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 02_aqi_categories.png")

# ── 4. Monthly AQI – box plots ───────────────────────────────────────────────
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df, x="month_name", y="AQI", order=month_order,
            palette="coolwarm", ax=ax, width=0.6, fliersize=2)
ax.set_title("Monthly AQI Distribution – All NCR Stations 2021", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("AQI")
ax.axhline(300, color="red", linestyle="--", linewidth=0.8, label="'Poor' threshold (300)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/03_monthly_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 03_monthly_boxplot.png")

# ── 5. City-level comparison ─────────────────────────────────────────────────
city_agg = df.groupby(["city","month"])["AQI"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

city_means = df.groupby("city")["AQI"].mean().sort_values(ascending=False)
bars = axes[0].barh(city_means.index, city_means.values,
                    color=sns.color_palette("Spectral_r", len(city_means)))
axes[0].set_title("Mean Annual AQI by City", fontsize=13)
axes[0].set_xlabel("Mean AQI")
for bar, val in zip(bars, city_means.values):
    axes[0].text(val + 2, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f}", va="center", fontsize=9)

for city in city_agg["city"].unique():
    sub = city_agg[city_agg["city"] == city].sort_values("month")
    axes[1].plot(sub["month"], sub["AQI"], marker="o", markersize=4, label=city)
axes[1].set_title("Monthly Mean AQI by City", fontsize=13)
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Mean AQI")
axes[1].set_xticks(range(1, 13)); axes[1].set_xticklabels(month_order)
axes[1].legend(fontsize=8); axes[1].axhline(200, color="grey", linestyle="--", linewidth=0.7)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/04_city_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 04_city_comparison.png")

# ── 6. Station-level heatmap (monthly mean AQI) ──────────────────────────────
pivot = df.groupby(["station","month"])["AQI"].mean().unstack()
pivot.columns = month_order

# Limit to top 20 stations by mean AQI for readability
top20 = df.groupby("station")["AQI"].mean().nlargest(20).index
pivot_top = pivot.loc[pivot.index.isin(top20)]

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(pivot_top, cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.3, ax=ax, cbar_kws={"label": "Mean AQI"})
ax.set_title("Monthly Mean AQI – Top 20 Stations (by annual mean)", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/05_station_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 05_station_heatmap.png")

# ── 7. Time-series: daily NCR average ───────────────────────────────────────
daily = df.groupby("date")["AQI"].mean().reset_index()
daily_7 = daily["AQI"].rolling(7, center=True).mean()
daily_30 = daily["AQI"].rolling(30, center=True).mean()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily["date"], daily["AQI"], color="#bdc3c7", linewidth=0.8, label="Daily mean")
ax.plot(daily["date"], daily_7,  color="#3498db", linewidth=1.5, label="7-day rolling avg")
ax.plot(daily["date"], daily_30, color="#e74c3c", linewidth=2,   label="30-day rolling avg")
ax.fill_between(daily["date"], 0, 200, alpha=0.04, color="green")
ax.fill_between(daily["date"], 200, 400, alpha=0.04, color="orange")
ax.fill_between(daily["date"], 400, daily["AQI"].max()+50, alpha=0.04, color="red")
ax.set_title("Daily AQI – NCR Average 2021 (with rolling averages)", fontsize=13)
ax.set_xlabel("Date"); ax.set_ylabel("AQI"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/06_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 06_timeseries.png")

# ── 8. Day-of-week effect ────────────────────────────────────────────────────
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="dayofweek", y="AQI", order=dow_order,
            palette="Blues_d", ax=ax, fliersize=2)
ax.set_title("AQI by Day of Week – Any Weekend Effect?", fontsize=13)
ax.set_xlabel(""); ax.set_ylabel("AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/07_dayofweek.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 07_dayofweek.png")

# ── 9. Top & bottom 10 stations ──────────────────────────────────────────────
station_means = df.groupby("station")["AQI"].mean().sort_values(ascending=False)
top10    = station_means.head(10)
bottom10 = station_means.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(top10.index[::-1], top10.values[::-1], color="#e74c3c")
axes[0].set_title("Top 10 Most Polluted Stations (Mean AQI)", fontsize=12)
axes[0].set_xlabel("Mean AQI")

axes[1].barh(bottom10.index[::-1], bottom10.values[::-1], color="#2ecc71")
axes[1].set_title("Top 10 Cleanest Stations (Mean AQI)", fontsize=12)
axes[1].set_xlabel("Mean AQI")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/08_top_bottom_stations.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 08_top_bottom_stations.png")

# ── 10. AQI seasonality: winter vs summer ────────────────────────────────────
df["season"] = df["month"].map(
    {12:"Winter",1:"Winter",2:"Winter",
     3:"Spring",4:"Spring",5:"Spring",
     6:"Summer",7:"Summer",8:"Summer",
     9:"Monsoon",10:"Autumn",11:"Autumn"})

fig, ax = plt.subplots(figsize=(10, 5))
season_order = ["Winter","Spring","Summer","Monsoon","Autumn"]
sns.violinplot(data=df, x="season", y="AQI", order=season_order,
               palette=["#3498db","#f1c40f","#e74c3c","#1abc9c","#e67e22"],
               inner="quartile", ax=ax)
ax.set_title("AQI Violin Plot by Season – NCR 2021", fontsize=13)
ax.set_xlabel(""); ax.set_ylabel("AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/09_seasonal_violin.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 09_seasonal_violin.png")

# ── 11. Summary statistics table ─────────────────────────────────────────────
summary = (df.groupby("city")["AQI"]
             .agg(["count","mean","median","std","min","max"])
             .round(1)
             .rename(columns={"count":"N","mean":"Mean","median":"Median",
                               "std":"Std","min":"Min","max":"Max"})
             .sort_values("Mean", ascending=False))

print("\n" + "=" * 60)
print("CITY-LEVEL AQI SUMMARY")
print("=" * 60)
print(summary.to_string())

print("\n✅  AQI EDA complete – plots saved to:", OUTPUT)
