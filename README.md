# CRUDE OIL CHARACTERISATION & PRODUCTS PROPERTIES ESTIMATION

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Tensorflow](https://img.shields.io/badge/TensorFlow-2.0+-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0+-red?style=for-the-badge&logo=keras)](https://keras.io/)

## Project Overview
This project develops **Machine Learning and ANN surrogate models** to predict critical crude oil quality parameters directly from readily available physical measurements, bypassing time-consuming laboratory assays. Using a dataset of 114 crude oil blend assays, the models learn complex non-linear relationships between physical properties and:
- **Set 1 — Hydrocarbon Composition**: Aromatics, Naphthenes, and Paraffins (% by weight)
- **Set 2 — Kinematic Viscosity**: cSt at 37.78°C (100°F) and 98.89°C (210°F)
- **Set 3 — Secondary Quality Specifications**: Cetane Number, Bromine Number, Aniline Point, Freeze Point, Pour Point, Cloud Point, Total Acid Number, and C:H Ratio

Traditional laboratory methods (ASTM assays) require 4–8 hours per sample, with large sample volumes and hazardous reagents. The surrogate models developed here provide near-instant predictions, enabling real-time Crude Distillation Unit (CDU) optimization and reduced operational risk.

> **Context**: This project was completed during **FOSSEE Semester Long Internship (Autumn 2025), IIT Bombay**

## Literature Foundation
- **Crude Oil Characterisation & Products Properties Estimation using Artificial Neural Network** *(Thesis)* - By Jhuma Sadhukhan (1997)
- **Application of artificial neural network for prediction of 10 crude oil properties** *(Research Paper)* - Alizadeh et al. (2023)
  
## Dataset Information
- **Source**: Custom python script used to extract data from 114 Crude Oil Assays (csv)
- **Size**: 114 records × 26 features
- **Inputs**: 13 standardized features

### 1. Input Features (Independent Variables):

| Category | Feature | Unit | Description |
|----------|---------|------|-------------|
| **Bulk Properties** | `StdLiquidDensity (kg/m3)` | kg/m³ | Standard Liquid Density |
| | `SulphurByWt (%)` | wt% | Total Sulfur Content |
| | `ConradsonCarbonByWt (%)` | wt% | Carbon residue after evaporation/pyrolysis |
| | `NitrogenByWt (%)` | wt% | Total Nitrogen Content |
| **Distillation Profile (Boiling Points)** | `Distillation Mass @ X Pct (C)@ 1 (%) - TBP` | °C | True boiling point at 1% cut |
| | `Distillation Mass @ X Pct (C)@ 5 (%) - TBP` | °C | True boiling point at 5% cut |
| | `Distillation Mass @ X Pct (C)@ 10 (%) - TBP` | °C | True boiling point at 10% cut |
| | `Distillation Mass @ X Pct (C)@ 30 (%) - TBP` | °C | True boiling point at 30% cut |
| | `Distillation Mass @ X Pct (C)@ 50 (%) - TBP` | °C | True boiling point at 50% cut |
| | `Distillation Mass @ X Pct (C)@ 70 (%) - TBP` | °C | True boiling point at 70% cut |
| | `Distillation Mass @ X Pct (C)@ 90 (%) - TBP` | °C | True boiling point at 90% cut |
| | `Distillation Mass @ X Pct (C)@ 95 (%) - TBP` | °C | True boiling point at 95% cut |
| | `Distillation Mass @ X Pct (C)@ 99 (%) - TBP` | °C | True boiling point at 99% cut |

### 2. Output Variables (Target Sets):

#### Set 1: Hydrocarbon Composition - PNA Analysis
| Variable | Description | Unit |
|----------|-------------|------|
| `AromByWt` | Aromatics content | % by weight |
| `NaphthenesByWt` | Naphthenes content | % by weight |
| `ParaffinsByWt` | Paraffins content | % by weight |

#### Set 2: Kinematic Viscosity
| Variable | Description | Unit |
|----------|-----------|------|
| `KinematicViscosity (cSt)@ 37.78 (C)` | Kinematic Viscosity at 37.78°C | cSt |
| `KinematicViscosity (cSt)@ 98.89 (C)` | Kinematic Viscosity at 98.89°C | cSt |

#### Set 3: Secondary Quality Specifications
| Variable | Description | Unit |
|----------|-------------|------|
| `CetaneNumber` | Ignition quality indicator for diesel fractions | Unitless |
| `BromineNumber` | Ignition quality indicator for diesel fractions | g Br2/100g |
| `AnilinePoint` | Indicator of aromatic content | °C |
| `FreezePoint` | Lowest temp before hydrocarbon crystals form | °C |
| `PourPoint` | Lowest temp at which oil remains fluid | °C |
| `CloudPoint` | Temperature at which wax crystals first appear | °C |
| `TotalAcidNumber` | Measure of acidity/corrosivity (naphthenic acid) | mg KOH/g |
| `CtoHRatioByWt` | Carbon-to-Hydrogen weight ratio (energy density) | Ratio |

> **Note**: `BromineNumber` was dropped from the dataset due to 100% missing values.



