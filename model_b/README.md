# Re:Desert — Model B
### Experimental satellite-derived vegetation persistence proxy | Lahbab, Dubai, UAE

**Purpose.** Model B uses earlier satellite and environmental features to classify whether later satellite observations meet a *retained-greenness proxy*. It is a research demonstration, **not** a validated estimate of native-tree survival, restoration suitability, water savings, or irrigation-free establishment.

This folder is a streamlined, self-contained **training and evaluation package**. It consolidates the original research report and selected outputs, while preserving the model's limitations. It does **not** reproduce the upstream Google Earth Engine data export.

## Quick start

From this folder, with Python 3 installed:

```bash
python -m pip install -r requirements.txt
python train_model_b.py
```

The script automatically reads `training_data.csv` from this folder and writes a new `outputs/` directory with trained-model weights, test metrics, confusion matrix, feature importance and test predictions. To change locations:

```bash
python train_model_b.py --data training_data.csv --output outputs
```

**GitHub note:** A public repository should include these seven extracted files, not the ZIP. You can view the HTML demonstration by downloading `Model_B_Heldout_Test_Map.html` and opening it in a browser; GitHub's file preview itself does not run HTML scripts.

## Files (seven)

| File | Purpose |
|---|---|
| `README.md` | This merged methodology, results, credits and limitations report |
| `train_model_b.py` | Portable end-to-end ML training, holdout and grouped cross-validation |
| `training_data.csv` | Original Earth Engine-derived feature table, 360 sample locations |
| `requirements.txt` | Python libraries |
| `metrics.json` | Preserved/reproduced detailed evaluation results |
| `model_B_rf.joblib` | Holdout-trained full Random Forest model for technical inspection or trusted reuse |
| `Model_B_Heldout_Test_Map.html` | Offline interactive *held-out sample* map, not a land-suitability raster |

The removed PNGs and CSV evaluation charts are **not lost analytically**: `train_model_b.py` regenerates `confusion_matrix.csv/.png`, `feature_importance.csv/.png`, `test_predictions.csv`, and `metrics.json`. The extra original sample-point PNG was illustrative and not needed to reproduce scores. We deliberately excluded the second all-sample demo `.joblib` to avoid confusing it with the holdout-validated model.

## Data and target definition

- **Region:** approximately 10 km × 10 km in Lahbab, Dubai.
- **Records:** 360 geographically distinct, intentionally balanced samples: 180 positive / 180 negative. These are selected research points, **not** whole-landscape class prevalence.
- **Earlier predictor window:** April–October 2019–2021.
- **Later outcome window:** April–October 2023–2025.
- **Proxy definition:** a positive label means later HLS median NDVI was at least **0.15**, and at least **75% of earlier NDVI**. These are *algorithmic satellite thresholds*, not measured plant survival.
- **Eight predictors:** `ndvi_early`, `msavi_early`, `ndvi_variability`, `swir_early`, `rain_mean_month_mm`, `lst_c`, `elevation_m`, `slope_deg`.
- GPS coordinates are used for geographic grouping but **not** given to the classifier. Later observations, proxy target and post-window information are not predictor features.

The supplied CSV is the model-ready export; the original Earth Engine preprocessing script/asset chain is not part of these fourteen uploaded files. This package can reproduce the ML experiment **from the CSV**, not independently reconstruct the original satellite feature table from raw imagery.

## Model and validation

- **Algorithm:** Random Forest classifier with 150 trees, minimum leaf size 3, balanced subsampling, seed 42.
- **Geographic holdout:** points are grouped into approximately 0.015-degree geographic blocks; no block appears in both train and test.
- **Training/test:** 247 / 113 points, drawn from 28 / 10 blocks. Holdout seed 0.
- **Comparators:** majority-class baseline; NDVI-only Random Forest; and a one-split NDVI decision tree.
- **Sensitivity:** 5-fold geographic GroupKFold cross-validation.

### Actual held-out results (113 samples)

| Model | Accuracy | Balanced accuracy | Macro F1 | ROC AUC |
|---|---:|---:|---:|---:|
| Most-frequent baseline | 0.4425 | 0.5000 | 0.3067 | 0.5000 |
| NDVI-only Random Forest | 0.9204 | 0.9224 | 0.9197 | 0.9452 |
| Eight-feature Random Forest | **0.9204** | 0.9203 | 0.9194 | 0.9876 |
| One-split NDVI tree | 0.9204 | — | 0.9194 | — |

**Eight-feature holdout confusion matrix** (actual rows × predicted columns):

| | Predicted not persistent | Predicted persistent |
|---|---:|---:|
| Actual not persistent | 46 | 4 |
| Actual persistent | 5 | 58 |

**Geographically grouped five-fold mean macro F1:** NDVI-only RF **0.8838**, full RF **0.9118**, simple NDVI decision tree **0.9148**. Fold performance varies; these figures do **not** justify a statistical-superiority claim.

The one-split decision tree found an NDVI threshold of approximately **0.13359** and reached the same holdout accuracy as the full model. This means the additional climate/terrain features have **not demonstrated a clear accuracy improvement**, despite a higher full-model holdout ROC AUC.

### Feature importance (Random Forest impurity; not causal)

| Feature | Importance |
|---|---:|
| NDVI early | 0.3246 |
| MSAVI early | 0.3039 |
| NDVI variability | 0.1890 |
| SWIR early | 0.0912 |
| Land surface temperature | 0.0344 |
| Elevation | 0.0306 |
| Slope | 0.0211 |
| Monthly mean rainfall | 0.0051 |

## Crucial limitations and responsible use

1. **Label–feature coupling:** The label is defined from future NDVI thresholds and earlier NDVI has a strong relationship to it. Apparent skill might largely reflect a simple spectral discriminator.
2. **Sampling bias:** 360 points were balanced across labels and selected from limited earlier-NDVI conditions (0.12–0.55). Model scores and prediction values must not be treated as calibrated regional restoration likelihoods.
3. **Resolution:** Rainfall is coarse (only four unique rainfall values across the sample); MODIS land-surface temperature is much coarser than HLS. These features are not plot-level soil measurements.
4. **Spatial dependence:** Blocks are disjoint but adjacent blocks can still be correlated. No independent region or temporally independent validation was done.
5. **Unverified ecology:** Satellite greenness cannot confirm that vegetation is native, naturally regenerated, non-irrigated, or able to establish without intervention. No ground-truth survival data, infiltration studies or groundwater verification are present.
6. **No planting prescriptions:** This is not a calibrated site-selection, soil-quality, water-demand, or species-recommendation tool. The HTML map displays only held-out observations on a schematic background.

## Datasets and software credits

Environmental features and labels were provided as a user-exported Google Earth Engine CSV. The accompanying source research report identifies:

- [NASA Harmonized Landsat and Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/): vegetation/spectral indicators and later proxy labels.
- [NASA GPM](https://gpm.nasa.gov/data): rainfall information.
- [NASA MODIS](https://modis.gsfc.nasa.gov/): land-surface temperature products.
- [ESA WorldCover](https://esa-worldcover.org/): supporting land-cover exclusions as described in the original report.
- [Google Earth Engine](https://earthengine.google.com/): upstream geospatial processing/export.
- Python libraries: NumPy, pandas, scikit-learn, matplotlib and joblib; see `requirements.txt`.

**AI assistance disclosure:** The reorganisation, portable training script and consolidated README were assisted by OpenAI ChatGPT. All supplied model results were checked against the original metrics, and the training script was rerun. Review the provenance of any other AI-assisted development before external submission.

## Provenance / competition eligibility

This package consolidates **pre-existing Model B work**. Reformatting, moving to GitHub or rerunning training does **not** change the original development date. If a hackathon prohibits extensions of earlier projects, **do not submit this as newly developed work or sign an originality declaration unless the organiser explicitly confirms eligibility.**

---

**Re:Desert — Right Land. Right Plants. Lasting Growth.**  
*Satellite Intelligence and AI for Self-Sustaining Desert Restoration*
