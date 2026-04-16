# seagrass-detection
A repository focused on the prediction of estuary land cover with a focus on seagrass from satellite imagery using models trained on UAV classified images or field surveys.

# Repository structure
Three folders are expected the versioned `scripts` and `notebooks` folders, and the unversioned `data` folder. The `scripts` contains routines used by the jupyter noebooks within the `notebooks` folder. 

An API key for ESNZ's tide model is required to run some of the notebooks. See `https://github.com/niwa/tide-examples` for setup info. Then create a `.env` file in the repository root with the contents `TIDE_API = "YOUR_KEY_HERE"`.

## Data folder
The `notebooks` expect a `classified_orthos` folder, and `ELF24505_ClassificationClasses.txt`, `ELF24505_SurveyDates.csv` and `ELF24505_satellite_day_search_range.csv` files within the `data` folder.

Various subfolders and containing files are created in the `data` folder when `notebooks` are run. Specificaly:
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
├── predictons \
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
* predictons
   * satellite_images - contains a `<PREDICTION_SITE>_<YEAR>_sentinel-2.nc` of low-tide clould free sattelite images for each site and year combination that has been run.
   * predictions - contains a `<TEST_SITE>_<YEAR>_prediction_<MODEL_NAME>.nc` for each site and year run.

# Environment setup
The `notebooks` and `scripts` require various Python packages to run. A pinned YML and general YML file for creating a Python environment with the required packages is in the root of the repository.
