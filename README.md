# Re:Desert — Model A (Lahbab surface-class classification)

## What this package contains
- `ReDesert_Model_A_Training_Colab.ipynb`: original user-supplied Colab notebook for Model A.
- `lahbab_hls_features_corrected.tif`: 8-band raster with B2, B3, B4, B8A, B11, B12, NDVI, MSAVI; WGS84, 417 × 353 pixels.
- `redesert_modelA_verified_labels.csv`: user-supplied label CSV with 290 samples in classes 0=bare/sparse, 1=vegetated, 2=built/disturbed.
- `results/`: output files created by rerunning the training function on the included raster and labels.

## Reproduce in Google Colab
1. Open the `.ipynb` in Google Colab.
2. Choose Runtime → Restart session (if necessary).
3. Run the upload cell and select BOTH the included `.tif` and `.csv` files. The file names must match the names hard-coded into the training cell. Use a clean session so Colab does not add `(1)` suffixes.
4. Run the package-install cell, then model definition/training cell, then results cell.
5. Download the model outputs using the final cell.

## Verification performed
Ran the original training function with both supplied input files in a Python environment containing rasterio, scikit-learn, pandas, numpy and Pillow. The model completed successfully with the following held-out geographically grouped test results:
- Train samples: 194; test samples: 96
- Accuracy: 0.68 (from classification report)
- Macro F1: 0.6914213717615559
- Balanced accuracy: 0.673015873015873
- Baseline macro F1: 0.2127659574468085

These results describe **land-cover classifications**, NOT tree survival, soil quality, groundwater availability, water savings, or the viability of a restoration site. The labels have not been independently ground-truthed by this verification, and the trained model has not been validated on a second location. The 25 built/disturbed samples are a small class.

## Competition originality and credits
This package reproduces previously supplied code and datasets. Packaging and rerunning do **not** establish that the project was originally developed within any specified competition window. Before submitting to Veritas AI, verify all work complied with its dates and originality rules and list all actual data sources, AI tools, training materials and contributors. Don't claim this is a new project developed during the competition if it extends prior work.

## How to share code
For Veritas, upload this notebook and its required files to an accessible GitHub repository, or publish the Colab notebook with inputs available to reviewers. A ZIP download is **not** a substitute for the required publicly viewable code link.
