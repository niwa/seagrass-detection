# seagrass-detection
A repository focused on the prediction of estuary land cover with a focus on seagrass from satellite imagery using models trained on UAV classified images or field surveys.

# Repository structure
Three folders are expected the versioned `scripts` and `notebooks` folders, and the unversioned `data` folder. The `scripts` contains routines used by the jupyter noebooks within the `notebooks` folder. 

The `notebooks` expect a `classified_orthos` folder, and `ELF24505_ClassificationClasses.txt`, `ELF24505_SurveyDates.csv` and `ELF24505_satellite_day_search_range.csv` files within the `data` folder.

Finally, an API key for ESNZ's tide model is required to run some of the notebooks. See `https://github.com/niwa/tide-examples` for setup info. Then create a `.env` file in the repository root with the contents `TIDE_API = "YOUR_KEY_HERE"`.

# Environment setup
The `notebooks` and `scripts` require various Python packages to run. A pinned YML and general YML file for creating a Python environment with the required packages is in the root of the repository.
