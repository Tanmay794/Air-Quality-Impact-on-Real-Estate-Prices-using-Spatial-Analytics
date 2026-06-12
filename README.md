# 🏠 Air Quality Impact on NCR Real Estate Prices

### Geospatial Econometrics & Spatial Analytics Study of Housing Markets in India's National Capital Region

This project investigates how air pollution influences residential property prices across the National Capital Region (NCR) of India. By integrating property transaction data with CPCB air quality measurements, the study quantifies the economic cost of pollution using geospatial analytics, hedonic pricing models, and spatial econometric techniques.

Using a merged dataset of **11,499 residential properties** and **47 CPCB monitoring stations**, the study finds that a **100-point increase in annual AQI is associated with a 31.2% reduction in property prices per square foot**, even after accounting for structural and spatial factors.

---

## 📌 Key Findings

- Quantified a **31.2% decline in property value per sqft** for every 100-point increase in AQI.
- Demonstrated that homebuyers price **long-term pollution reputation** rather than real-time pollution levels.
- Validated results using **Spatial Lag Models** and spatial autocorrelation diagnostics.
- Identified a **pollution justice gap**, where budget homebuyers face higher pollution exposure while receiving lower price compensation.
- Estimated up to **₹4,290 Cr** in potential housing value gains from pollution reduction.
- Revealed significant regional heterogeneity across Delhi, Gurgaon, Noida, Ghaziabad, and Greater Noida.

---

## 🎯 Problem Statement

Air pollution is one of the largest environmental challenges facing NCR. While its health impacts are widely studied, its effect on residential property prices remains less understood.

This project addresses three key questions:

1. Does air pollution significantly discount housing prices?
2. Do buyers react to current pollution levels or long-term pollution reputation?
3. Are pollution costs distributed equally across different income groups and regions?

---

## 📊 Dataset Overview

### Property Data
- 11,499 residential properties
- Delhi NCR region
- Features include:
  - Property price
  - Price per sqft
  - Area
  - Bedrooms
  - Bathrooms
  - Furnishing status
  - Transaction type
  - Geographic coordinates

### Air Quality Data
- CPCB Daily AQI Records (2021)
- 47 monitoring stations across NCR
- Annual and monthly AQI metrics

### Final Dataset
- Properties spatially matched to nearest monitoring station using Haversine distance
- Final merged dataset size: **11,499 observations**

---

## 🛠 Methodology

### 1. Data Engineering
- Property dataset cleaning and standardization
- Missing value imputation
- Outlier removal
- Schema harmonization across sources
- Feature engineering

### 2. Geospatial Matching
- Implemented Haversine nearest-neighbor matching
- Assigned each property to its nearest CPCB monitoring station
- Generated annual and monthly pollution exposure measures

### 3. Exploratory Data Analysis

#### Air Quality Analysis
- AQI distribution analysis
- Seasonal pollution trends
- Station-level comparisons
- City-level pollution patterns
- Time-series analysis
- AQI category breakdown

#### Property Market Analysis
- Price distribution analysis
- City-wise market comparison
- BHK analysis
- Furnishing premium analysis
- Correlation analysis
- Geographic price mapping

### 4. Hedonic Pricing Model

Estimated the impact of:

- Air Quality Index (AQI)
- Property size
- Furnishing status
- Transaction type
- Structural characteristics

using:

- Log-linear regression
- Robust HC3 standard errors
- Variance Inflation Factor (VIF) diagnostics

### 5. Spatial Econometrics

#### Spatial Diagnostics
- Moran's I
- Spatial autocorrelation testing
- K-nearest-neighbor weight matrices

#### Spatial Lag Model
- Maximum Likelihood Spatial Lag estimation
- Spatial spillover analysis
- OLS vs Spatial Model comparison

---

## 🔬 Additional Analyses

### Pollution Justice Analysis
Investigated whether lower-income households face higher pollution exposure and whether property discounts adequately compensate for environmental disadvantages.

### Temporal Analysis
Compared:

- Annual AQI
- Monthly AQI
- Seasonal pollution effects

to determine whether buyers respond to real-time pollution or long-term pollution reputation.

### Regional Analysis
Compared:

- Gurgaon Belt
- East NCR (Noida, Greater Noida, Ghaziabad)

to identify location-specific pollution pricing effects.

### Station-Level Comparison
Performed within-city comparisons between cleaner and more polluted monitoring zones to isolate local pollution effects.

---

## 📈 Visualizations

The project includes:

- AQI seasonality dashboards
- Pollution heatmaps
- Property price heatmaps
- Interactive GIS maps
- Temporal trend analysis
- Regional comparison dashboards
- Pollution justice visualizations

---

## 🧰 Tech Stack

### Languages
- Python

### Data Processing
- Pandas
- NumPy

### Statistics & Econometrics
- Statsmodels
- SciPy

### Spatial Analytics
- PySAL
- Moran's I
- Spatial Lag Models

### Visualization
- Matplotlib
- Seaborn
- Folium

### GIS
- Haversine Distance Matching
- Spatial Weight Matrices
- Geographic Mapping

---

## 📋 Results Summary

| Analysis | Key Result |
|-----------|------------|
| Hedonic Regression | -31.2% price effect per +100 AQI |
| Spatial Lag Model | AQI effect remains significant |
| Temporal Analysis | Annual AQI significant; Monthly AQI insignificant |
| Pollution Justice | Budget buyers face higher pollution exposure |
| Regional Analysis | Strong effect in East NCR; confounded in Gurgaon |
| Policy Simulation | ₹4,290 Cr potential value unlocked |

---

## 🚀 Future Improvements

- Geographically Weighted Regression (GWR)
- Spatial Error Models (SEM)
- Satellite PM2.5 integration
- Multi-year panel data analysis
- XGBoost-based property valuation
- Causal inference using policy interventions

---

## 👨‍💻 Author

**Tanmay Chowdhary**  
Data Science | Machine Learning | Analytics | Spatial Analytics

---

## ⭐ Highlights

- 11,499 property transactions analyzed
- 47 CPCB monitoring stations integrated
- Geospatial matching using Haversine distance
- Hedonic Pricing Regression
- Spatial Econometric Modeling
- Pollution Justice Analysis
- Interactive GIS Visualizations
- Policy Impact Quantification

---

If you found this project interesting, consider ⭐ starring the repository.
