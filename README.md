# BCI EEG Classification Competition - 2nd Place Solution

**Team Imitation** | Egypt National AI Competition 2025 | MTC, AIC, Huawei

## Overview

This repository contains our **2nd place solution** (out of 216 teams) for the National Brain-Computer Interface (BCI) EEG Classification Competition. We developed high-performance pipelines for two distinct BCI paradigms:
- **Motor Imagery (MI)** - Binary classification of left/right hand movements
- **Steady-State Visual Evoked Potentials (SSVEP)** - Four-class directional classification

## Key Results

### Motor Imagery Track
- **Method**: Hybrid CSP + 1D-CNN pipeline
- **Performance**: 64% validation accuracy
- **Key Innovation**: Aggressive data curation strategy removing 680 problematic trials

### SSVEP Track  
- **Method**: Multi-modal feature fusion + Ensemble learning
- **Performance**: 82% LOSO F1-score
- **Key Innovation**: Comprehensive feature engineering (FBCCA, TRCA, PLV, SNR, PSD)

## Technical Approach

### Data Quality First
Both tracks revealed severe data quality issues. Our forensic analysis uncovered:
- Systematic artifact contamination
- Subject duplication artifacts
- Non-physiological outliers

**Solution**: Expert-driven data curation proved more effective than automated cleaning.

### Motor Imagery Pipeline
```
Raw EEG → Data Curation → Global Standardization → CSP → 1D-CNN → Prediction
```

### SSVEP Pipeline
```
Raw EEG → CAR → Feature Extraction* → Ensemble Classifier → Prediction
```

## Key Insights

1. **Signal Processing > Deep Learning**: For noisy EEG data, deterministic signal processing outperformed end-to-end deep learning
2. **Data Quality Critical**: Systematic outlier removal improved performance more than model complexity
3. **Ensemble Methods**: Combining diverse classifiers provided robust predictions
4. **Domain Knowledge**: Understanding EEG physiology was crucial for effective feature engineering

## Documentation

Detailed technical documentation available in `docs/`:
- [Motor Imagery System Description](docs/MI_System_Description.pdf)
- [SSVEP System Description](docs/SSVEP_System_Description.pdf)

## Competition Details

- **Organizers**: MTC, AIC, Huawei
- **Participants**: 216 teams
- **Our Rank**: 2nd place
- **Year**: 2025

## Contact

For questions or collaboration opportunities, please open an issue or contact the team.