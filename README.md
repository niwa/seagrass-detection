# seagrass-detection
A repository focused on the prediction of estuary land cover with a focus on seagrass from satellite imagery using models trained on UAV classified images or field surveys.

This repository was used to generate the results shared in the [Estuary Habitat Detection application](https://estuaries.niwa.co.nz/). The work was funded by the Envirolink Tools Programme. The application results use Version 1 of the habitat classification models and training datasets; the application version identifies the software release and may differ from future model or dataset versions.

# Repository structure
Three folders are expected: the versioned `scripts` and `notebooks` folders, and the unversioned `data` folder. The `scripts` folder contains routines used by the Jupyter notebooks within the `notebooks` folder.

An API key for ESNZ's tide model is required to run some of the notebooks. See `https://github.com/niwa/tide-examples` for setup info. Then create a `.env` file in the repository root with the contents `TIDE_API = "YOUR_KEY_HERE"`.

## Data folder
The `notebooks` expect a `classified_orthos` folder, and `ELF24505_ClassificationClasses.txt`, `ELF24505_SurveyDates.csv` and `ELF24505_satellite_day_search_range.csv` files within the `data` folder.

Various subfolders and files are created in the `data` folder when notebooks are run. Specifically:
├── site_polygons folder \
\
├── training folder \
|   ├── satellite_images subfolder \
|   └── training_data subfolder \
\
├── models folder \
├── validation folder \
|  └── predictions subfolder \
\
├── predictions \
|  └── satellite_images subfolder \
|  ── training_data  subfolder \
└── predictions


Containing the files: 
* site_polygons - contains a `<SITE_NAME>_polygon.gpkg` for each site generated from the UAV image in the `classified_orthos` folder.
* training
  * satellite_images - contains a `<SITE_NAME>_sentinel-2.nc` for each site containing low-tide cloud free images near the UAV survey date.
   * training_data - a subfolder for each sampling methodology. Contains a `<SITE_NAME>_training_data.csv`, `<SITE_NAME>_training_data.gpkg` and `<SITE_NAME>_training_data_summary.csv` for each site and an overall `samples_summary.csv`. The site specific datasets are used for training or validation. The `gpkg` allows visualisation of the training data in a GIS package. 
* models - contains a `<MODEL_NAME>.joblib`, `<MODEL_NAME>_random_forest_feature_importance.png`, `<MODEL_NAME>_training_uav_class_IDs.png`, `<MODEL_NAME>_training_satellite_class_IDs.png`, and `<MODEL_NAME>_class_mappings.csv` for each trained model. The class mappings record the UAV class to Satellite class mappings used prior to model training.
* validation
   * predictions - contains a `<TEST_SITE>_prediction_<MODEL_NAME>.nc`, `<TEST_SITE>_prediction_<MODEL_NAME>_confusion_matrix_time_all_dates.png` and `<TEST_SITE>_prediction_<MODEL_NAME>_confusion_matrix_time_<i>.png` for each satellite time for each model to test site validation.
* predictions
   * satellite_images - contains a `<PREDICTION_SITE>_<YEAR>_sentinel-2.nc` of low-tide cloud-free satellite images for each site and year combination that has been run.
   * predictions - contains a `<TEST_SITE>_<YEAR>_prediction_<MODEL_NAME>.nc` for each site and year run.

# Environment setup
The notebooks and scripts require various Python packages to run. The [environment.yml](environment.yml) file specifies the environment and its dependencies.

# Workflow
The notebooks are generally run in the following order:

1. **Prepare UAV imagery:** [Standardise the UAV data](notebooks/uav_imagery_setup/standardise_uav_data.ipynb), and prepare [target areas](notebooks/uav_imagery_setup/record_target_uav_areas_for_website.ipynb) and [polygons](notebooks/uav_imagery_setup/prepare_uav_polygon_for_website.ipynb) for review and website display.
2. **Prepare training imagery:** [Compute the band quality baseline](notebooks/data_download/compute_band_quality_baseline.ipynb), then [download satellite imagery for training](notebooks/data_download/download_satellite_training.ipynb). The baseline notebook downloads the imagery it needs before calculating the baseline.
3. **Extract samples:** Use the [Random Forest](notebooks/training/extract_training_samples.ipynb) or [U-Net](notebooks/training/unet_extract_training_samples.ipynb) notebook.
4. **Train and validate:** Run the [Random Forest](notebooks/training/train_and_validate_percent_holdout.ipynb) or [U-Net](notebooks/training/unet_train_and_validate_percent_holdout_diffuser.ipynb) notebook.
5. **Predict and prepare the website:** [Download prediction imagery](notebooks/data_download/download_satellite_predictions.ipynb), then run the [Random Forest](notebooks/prediction/predict_for_website.ipynb) or [U-Net](notebooks/prediction/unet_predict_for_website.ipynb) prediction and website notebook.
6. **Process council priority areas:** [Compute the council band quality baseline](notebooks/data_download/compute_council_band_quality_baseline.ipynb), [download council prediction imagery](notebooks/data_download/download_satellite_council_predictions.ipynb), then run the [council U-Net prediction and website notebook](notebooks/prediction/unet_predict_council_sites_for_website.ipynb).

# Experiments, previous approaches and technical notes
## Selecting low-tide images
The models are very sensitive to low-tide images. Several approaches were explored to improve the selection of low-tide images. These include:
* Using the NIWA Tide API (currently used)
* Using SCL classes - noticed drift in classifications over time and sensitivity to areas where sediments change over time
* Calculating site-specific MNDWI thresholds - struggled with high suspended sediment
* Image selection - selection of images to use for baseline calculation for anomaly detection. Briefly explored before halting as the tide API with anomaly filtering was working well enough.
The code associated with these approaches was deleted 19/11/2026.

## Model architectures 
Random Forest and a diffusers-based U-Net were used for land-cover classification. A U-Net++ architecture using a pretrained Sentinel-2 model was briefly explored but not pursued further because the diffusers U-Net was working. Pretrained backbones could be revisited in future.

## Validation
Three approaches were explored: holding out a site, holding out a single site and date, and a 10% stratified holdout. The stratified holdout was selected because the limited data and variation in classes present between sites made site-based holdouts difficult to evaluate.

For Random Forest, the split is stratified by pixel class. For U-Net, it is made by tile, using class counts for stratification.

The holdout result doesn’t necessarily measure performance at an unseen site. U-Net tiles can also overlap, so the validation data may not be spatially independent.

