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

