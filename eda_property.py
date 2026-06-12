"""
NCR Property Listings 2021
Exploratory Data Analysis
"""

# ── 0. Imports ──────────────────────────────────────────────────────────────
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

OUTPUT = "plots_property"
import os; os.makedirs(OUTPUT, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("property/ncr_clean.csv")

# Price in lakhs for readability
df["price_lakh"] = df["price_total"] / 1e5

print("=" * 60)
print("PROPERTY DATASET  –  OVERVIEW")
print("=" * 60)
print(f"  Records   : {len(df):,}")
print(f"  Columns   : {df.columns.tolist()}")
print(f"\n  Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\n  Dtypes:")
print(df.dtypes)

# ── 2. Basic descriptive stats ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)
num_cols = ["price_total","area_sqft","bhk","bathrooms","balcony","price_per_sqft"]
print(df[num_cols].describe().round(2).T.to_string())

# ── 3. Price distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["price_lakh"], bins=60, color="#3498db", edgecolor="white")
axes[0].set_title("Property Price Distribution (₹ Lakh)", fontsize=13)
axes[0].set_xlabel("Price (₹ Lakh)"); axes[0].set_ylabel("Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}L"))
axes[0].axvline(df["price_lakh"].median(), color="orange", linestyle="--",
                label=f"Median ₹{df['price_lakh'].median():.0f}L")
axes[0].axvline(df["price_lakh"].mean(), color="red", linestyle="--",
                label=f"Mean ₹{df['price_lakh'].mean():.0f}L")
axes[0].legend()

axes[1].hist(np.log1p(df["price_total"]), bins=60, color="#9b59b6", edgecolor="white")
axes[1].set_title("Log-Price Distribution (natural log)", fontsize=13)
axes[1].set_xlabel("ln(Price)"); axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/01_price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] 01_price_distribution.png")

# ── 4. Price per sqft distribution ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["price_per_sqft"], bins=60, color="#e67e22", edgecolor="white")
axes[0].set_title("Price per Sq.Ft Distribution", fontsize=13)
axes[0].set_xlabel("₹ / sq.ft"); axes[0].set_ylabel("Count")
axes[0].axvline(df["price_per_sqft"].median(), color="red", linestyle="--",
                label=f"Median ₹{df['price_per_sqft'].median():.0f}")
axes[0].legend()

stats.probplot(df["price_per_sqft"], plot=axes[1])
axes[1].set_title("Q-Q Plot: Price/sqft vs Normal", fontsize=13)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/02_price_sqft_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 02_price_sqft_dist.png")

# ── 5. City-level analysis ───────────────────────────────────────────────────
city_stats = (df.groupby("city")["price_lakh"]
                .agg(["count","mean","median"])
                .round(1)
                .sort_values("median", ascending=False))
print("\nCity-level price summary (₹ Lakh):")
print(city_stats.to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=df, x="city", y="price_lakh",
            order=city_stats.index, palette="Set2",
            ax=axes[0], fliersize=2)
axes[0].set_title("Price by City (₹ Lakh)", fontsize=13)
axes[0].set_xlabel(""); axes[0].set_ylabel("Price (₹ Lakh)")
axes[0].tick_params(axis='x', rotation=25)

sns.boxplot(data=df, x="city", y="price_per_sqft",
            order=city_stats.index, palette="Set2",
            ax=axes[1], fliersize=2)
axes[1].set_title("Price/Sq.Ft by City", fontsize=13)
axes[1].set_xlabel(""); axes[1].set_ylabel("₹ / sq.ft")
axes[1].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/03_city_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 03_city_price.png")

# ── 6. BHK analysis ─────────────────────────────────────────────────────────
bhk_counts = df["bhk"].value_counts().sort_index()
bhk_price  = df.groupby("bhk")["price_lakh"].median().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(bhk_counts.index.astype(str), bhk_counts.values,
            color=sns.color_palette("Blues_d", len(bhk_counts)))
axes[0].set_title("Listing Count by BHK", fontsize=13)
axes[0].set_xlabel("BHK"); axes[0].set_ylabel("Count")
for i, (x, y) in enumerate(zip(bhk_counts.index, bhk_counts.values)):
    axes[0].text(i, y + 20, str(y), ha="center", fontsize=9)

axes[1].bar(bhk_price.index.astype(str), bhk_price.values,
            color=sns.color_palette("Oranges_d", len(bhk_price)))
axes[1].set_title("Median Price by BHK (₹ Lakh)", fontsize=13)
axes[1].set_xlabel("BHK"); axes[1].set_ylabel("Median Price (₹ Lakh)")
for i, (x, y) in enumerate(zip(bhk_price.index, bhk_price.values)):
    axes[1].text(i, y + 1, f"₹{y:.0f}L", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/04_bhk_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 04_bhk_analysis.png")

# ── 7. Categorical features ──────────────────────────────────────────────────
cat_cols = ["status","transaction","furnishing","property_type"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, col in zip(axes.flat, cat_cols):
    vc = df[col].value_counts()
    ax.bar(vc.index, vc.values, color=sns.color_palette("Paired", len(vc)))
    ax.set_title(f"Distribution: {col}", fontsize=12)
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=20)

plt.suptitle("Categorical Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/05_categorical_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 05_categorical_features.png")

# ── 8. Price by categorical features ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for ax, col in zip(axes.flat, cat_cols):
    order = df.groupby(col)["price_lakh"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=col, y="price_lakh", order=order,
                palette="Set3", ax=ax, fliersize=2)
    ax.set_title(f"Price (₹L) by {col}", fontsize=12)
    ax.set_xlabel(""); ax.set_ylabel("Price (₹ Lakh)")
    ax.tick_params(axis='x', rotation=20)

plt.suptitle("Price Variation by Categorical Features", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/06_price_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 06_price_by_category.png")

# ── 9. Area vs Price scatter ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# raw
axes[0].scatter(df["area_sqft"], df["price_lakh"],
                alpha=0.25, s=8, color="#3498db")
m, b, r, p, _ = stats.linregress(df["area_sqft"], df["price_lakh"])
xline = np.linspace(df["area_sqft"].min(), df["area_sqft"].max(), 200)
axes[0].plot(xline, m*xline + b, color="red", linewidth=1.5,
             label=f"OLS  R²={r**2:.2f}")
axes[0].set_title("Area vs Price", fontsize=13)
axes[0].set_xlabel("Area (sq.ft)"); axes[0].set_ylabel("Price (₹ Lakh)")
axes[0].legend()

# log-log
axes[1].scatter(np.log(df["area_sqft"]), np.log(df["price_total"]),
                alpha=0.25, s=8, color="#9b59b6")
la = np.log(df["area_sqft"]); lp = np.log(df["price_total"])
m2, b2, r2, _, _ = stats.linregress(la, lp)
xl = np.linspace(la.min(), la.max(), 200)
axes[1].plot(xl, m2*xl + b2, color="red", linewidth=1.5,
             label=f"Log-log OLS  R²={r2**2:.2f}")
axes[1].set_title("log(Area) vs log(Price)", fontsize=13)
axes[1].set_xlabel("ln(Area)"); axes[1].set_ylabel("ln(Price)")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT}/07_area_price_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 07_area_price_scatter.png")

# ── 10. Correlation heatmap ──────────────────────────────────────────────────
num_df = df[["price_total","area_sqft","bhk","bathrooms","balcony","price_per_sqft"]].copy()
num_df["log_price"] = np.log(df["price_total"])
num_df["log_area"]  = np.log(df["area_sqft"])

corr = num_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap – Property Features", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 08_correlation_heatmap.png")

# ── 11. Price by city × furnishing ──────────────────────────────────────────
pivot = df.pivot_table(index="city", columns="furnishing",
                       values="price_per_sqft", aggfunc="median")

fig, ax = plt.subplots(figsize=(12, 6))
pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_title("Median Price/Sq.Ft by City & Furnishing Status", fontsize=13)
ax.set_xlabel(""); ax.set_ylabel("₹ / sq.ft")
ax.tick_params(axis='x', rotation=25)
ax.legend(title="Furnishing", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(f"{OUTPUT}/09_city_furnishing_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 09_city_furnishing_price.png")

# ── 12. Outlier detection – IQR ─────────────────────────────────────────────
Q1 = df["price_per_sqft"].quantile(0.25)
Q3 = df["price_per_sqft"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["price_per_sqft"] < Q1 - 1.5*IQR) |
              (df["price_per_sqft"] > Q3 + 1.5*IQR)]

print(f"\n  Outliers in price_per_sqft (IQR method): {len(outliers):,} / {len(df):,}  "
      f"({100*len(outliers)/len(df):.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.boxplot(column="price_per_sqft", by="city", ax=axes[0], flierprops=dict(marker='.', markersize=3))
axes[0].set_title("Price/Sqft Outliers by City"); axes[0].set_xlabel(""); plt.sca(axes[0])
plt.xticks(rotation=30)

df.boxplot(column="price_lakh", by="bhk", ax=axes[1], flierprops=dict(marker='.', markersize=3))
axes[1].set_title("Price (₹L) Outliers by BHK"); axes[1].set_xlabel("BHK")

plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/10_outlier_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 10_outlier_boxplots.png")

# ── 13. Geographic scatter (lat/lon coloured by price) ───────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(df["longitude"], df["latitude"],
                c=df["price_per_sqft"], cmap="YlOrRd",
                s=5, alpha=0.5)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("₹ / sq.ft")
ax.set_title("Geographic Distribution of Properties – Coloured by Price/Sq.Ft", fontsize=13)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/11_geo_price_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] 11_geo_price_map.png")

# ── 14. Final summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print(f"  Total listings      : {len(df):,}")
print(f"  Median price        : ₹{df['price_lakh'].median():.1f} Lakh")
print(f"  Median area         : {df['area_sqft'].median():.0f} sq.ft")
print(f"  Most common BHK     : {int(df['bhk'].mode()[0])} BHK")
print(f"  Most common city    : {df['city'].mode()[0]}")
print(f"  Highest avg price   : {df.groupby('city')['price_lakh'].mean().idxmax()}")
print(f"  Furnished premium   : ₹{df[df['furnishing']=='Furnished']['price_per_sqft'].median() - df[df['furnishing']=='Unfurnished']['price_per_sqft'].median():.0f} / sq.ft")
print(f"\n✅  Property EDA complete – plots saved to: {OUTPUT}")
