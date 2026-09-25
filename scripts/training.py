"""Module focused on extracting training data and training pixel classification models."""

import utils
import sentinel2
import sampling
import plotting
import gc
import typing
import geopandas
import xarray
import pandas
import scipy
import sklearn.ensemble
import sklearn.model_selection
import joblib
import numpy
import pathlib
import matplotlib.pyplot
import torch
import segmentation_models_pytorch
import huggingface_hub
import pytorch_lightning
import iterstrat.ml_stratifiers


UNET_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
UNET_PRETRAINED_REPO = "giswqs/s2-water-unetplusplus-efficientnet-b4"


def get_training_data_across_sites(
    training_sites: list,
    samples_path: pathlib.Path,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
):
    """Combine all training datasets then drop the UAV classes to ignore
    and map the UAV classes to the satellite classes."""

    samples_dataframe = load_samples(training_sites=training_sites,
                                     samples_path=samples_path)

    training_dataframe = map_satellite_ids_into_samples(
        samples_dataframe=samples_dataframe,
        uav_labels_file=uav_labels_file,
        uav_classes_to_ignore=uav_classes_to_ignore,
        satellite_classes=satellite_classes,
        satellite_from_uav_classes=satellite_from_uav_classes,
        drop_scl_classes_to_ignore=True)

    return training_dataframe


def get_training_data_across_sites_excluding_one_site_date(
    training_sites: list,
    test_site: str,
    test_date: str,
    samples_path: pathlib.Path,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
):
    """Combine all training datasets then drop the UAV classes to ignore
    and map the UAV classes to the satellite classes. Exclude the test
    site and date."""

    samples_dataframe = load_samples_excluding_test_date(
        training_sites=training_sites,
        samples_path=samples_path,
        test_site=test_site,
        test_date=test_date
    )


    training_dataframe = map_satellite_ids_into_samples(
        samples_dataframe=samples_dataframe,
        uav_labels_file=uav_labels_file,
        uav_classes_to_ignore=uav_classes_to_ignore,
        satellite_classes=satellite_classes,
        satellite_from_uav_classes=satellite_from_uav_classes,
        drop_scl_classes_to_ignore=True)

    return training_dataframe


def randomise_to_test_and_training_across_sites(
    training_sites: list,
    samples_path: pathlib.Path,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
    test_threshold: float,
):
    """Combine all training datasets then drop the UAV classes to ignore
    and map the UAV classes to the satellite classes. Train a random
    forest classifier and return."""

    samples_dataframe = load_samples(training_sites=training_sites,
                                     samples_path=samples_path)

    training_dataframe = map_satellite_ids_into_samples(
        samples_dataframe=samples_dataframe,
        uav_labels_file=uav_labels_file,
        uav_classes_to_ignore=uav_classes_to_ignore,
        satellite_classes=satellite_classes,
        satellite_from_uav_classes=satellite_from_uav_classes,
        drop_scl_classes_to_ignore=True)

    training_dataframe, test_dataframe = sklearn.model_selection.train_test_split(
        training_dataframe,
        test_size=test_threshold,
        stratify=training_dataframe["satellite_class_id"],
    )

    training_dataframe = training_dataframe.reset_index(drop=True)
    test_dataframe = test_dataframe.reset_index(drop=True)

    return training_dataframe, test_dataframe


def train_random_forest_classifier(
    training_dataframe: pandas.DataFrame,
):
    """Train a random forest classifier for a given training dataframe."""

    print("\tTrain a Random Forest Model")
    training_classes = numpy.array(training_dataframe["satellite_class_id"])
    training_observations = numpy.array(
        training_dataframe.drop(columns=["satellite_class_id", "uav_class_id", "time"])
    )
    model_columns = pandas.DataFrame(columns=training_dataframe.drop(columns=["satellite_class_id", "uav_class_id", "time"]).columns)
    classifier = sklearn.ensemble.RandomForestClassifier()
    model = classifier.fit(training_observations, training_classes)

    return model, model_columns


def load_samples_and_tile(
    training_sites: list,
    low_tide_delta_hrs: int,
    low_tide_delta_mins: int,
    max_cloud_cover: int,
    method_2_threshold: float,
    tile_size: int = 64,
    stride: int = None, verbose: bool = False,
) -> tuple:
    """Load the satellite imagery and UAV classification for each site, align the UAV
    classification onto the satellite grid (mode per pixel), then cut both into fixed
    size square tiles for U-Net training. Tiles that are entirely nodata are dropped.
    Returns (tiles, labels) as numpy arrays of shape (N, bands, tile_size, tile_size)
    and (N, tile_size, tile_size). This is a first-pass implementation - band
    selection/normalisation and smarter tile filtering can be refined later."""

    stride = stride or tile_size
    tiles = []
    labels = []
    for training_site in training_sites:
        if verbose:
            print(f"Load and tile site: {training_site}")
        satellite_file = utils.get_satellite_path(
            site_name=training_site, low_tide_delta_hrs=low_tide_delta_hrs, low_tide_delta_mins=low_tide_delta_mins,
            max_cloud_cover=max_cloud_cover
        )
        uav_file = utils.get_training_data_path(
                site_name=training_site, sample_method="sampling_2", method_2_threshold=method_2_threshold,
                low_tide_delta_hrs=low_tide_delta_hrs, low_tide_delta_mins=low_tide_delta_mins,
                max_cloud_cover=max_cloud_cover
            ).with_suffix(".nc")

        if not satellite_file.exists():
            print(f"\tWARNING - no satellite image for site {training_site} (no valid low tide/"
                  "low cloud image was available). Skipping this site.")
            continue
        if not uav_file.exists():
            raise ValueError(f"\tWARNING - no training raster extracted from the UAV classification for site"
                             f" {training_site}. Please run extract_training_samples_raster.ipynb.")
                    

        satellite_data = utils.load_satellite(filename=satellite_file, chunks=None)
        uav_labels = utils.load_classification(filename=uav_file, chunks=None)
        satellite_data = satellite_data.reindex_like(uav_labels, method="nearest",)

        label = numpy.where(
            numpy.isnan(uav_labels.values),
            utils.UAV_NAN_CLASS,
            uav_labels.values,
        ).astype(numpy.int64)
        height, width = label.shape
        if verbose:
            print(f"\tTile images to {int(numpy.ceil(height/stride))*int(numpy.ceil(width/stride))}:"
                  f" ({int(numpy.ceil(height/stride))}, {int(numpy.ceil(width/stride))})")

        for time_index in range(len(satellite_data["time"])):
            # Select a fixed, ordered set of bands so every tile has the same channel count
            image = satellite_data[UNET_BANDS].isel(time=time_index).to_array().values

            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    image_slice = image[:, y:y + tile_size, x:x + tile_size]
                    label_slice = label[y:y + tile_size, x:x + tile_size]
                    valid_labels = label_slice != utils.UAV_NAN_CLASS
                    if numpy.isnan(image_slice).all() or not valid_labels.any():
                        if verbose:
                            print(f"\t\tSkipping tile at position ({y}, {x}) due to NaN values.")
                        continue

                    image_tile = numpy.full(
                        (image.shape[0], tile_size, tile_size),
                        numpy.nan,
                        dtype=numpy.float32,
                    )
                    label_tile = numpy.full(
                        (tile_size, tile_size),
                        utils.UAV_NAN_CLASS,
                        dtype=numpy.int64,
                    )
                    slice_height, slice_width = label_slice.shape
                    image_tile[:, :slice_height, :slice_width] = image_slice
                    label_tile[:slice_height, :slice_width] = label_slice
                    tiles.append(image_tile)
                    labels.append(label_tile)

    tiles = numpy.stack(tiles).astype(numpy.float32)
    labels = numpy.stack(labels).astype(numpy.int64)

    return tiles, labels


def load_samples_and_tile_excluding_one_site_date(
    training_sites: list,
    test_site: str,
    test_date: str,
    low_tide_delta_hrs: int,
    low_tide_delta_mins: int,
    max_cloud_cover: int,
    method_2_threshold: float,
    tile_size: int = 64,
    stride: int = None,
    verbose: bool = False,
) -> tuple:
    """Same as load_samples_and_tile, but exclude the satellite image for test_site at
    test_date (e.g. so it can be held out for testing). Mirrors how
    load_samples_and_tile relates to
    get_training_data_across_sites_excluding_one_site_date vs
    get_training_data_across_sites."""

    stride = stride or tile_size
    tiles = []
    labels = []
    for training_site in training_sites:
        if verbose:
            print(f"Load and tile site: {training_site}")
        satellite_file = utils.get_satellite_path(
            site_name=training_site, low_tide_delta_hrs=low_tide_delta_hrs, low_tide_delta_mins=low_tide_delta_mins,
            max_cloud_cover=max_cloud_cover
        )
        uav_file = utils.get_training_data_path(
                site_name=training_site, sample_method="sampling_2", method_2_threshold=method_2_threshold,
                low_tide_delta_hrs=low_tide_delta_hrs, low_tide_delta_mins=low_tide_delta_mins,
                max_cloud_cover=max_cloud_cover
            ).with_suffix(".nc")

        if not satellite_file.exists():
            print(f"\tWARNING - no satellite image for site {training_site} (no valid low tide/"
                  "low cloud image was available). Skipping this site.")
            continue

        satellite_data = utils.load_satellite(filename=satellite_file, chunks=None)
        uav_labels = utils.load_classification(filename=uav_file, chunks=None)
        satellite_data = satellite_data.reindex_like(uav_labels, method="nearest",)

        # Identify (if present) the time index to exclude for the held out test site/date
        excluded_time_index = None
        if training_site == test_site:
            dates = satellite_data["time"].dt.strftime("%Y-%m-%d").values
            test_date_normalized = pandas.to_datetime(test_date).strftime("%Y-%m-%d")
            matching_indices = numpy.flatnonzero(dates == test_date_normalized)
            if len(matching_indices) == 0:
                print(
                    f"Warning: test date {test_date} is not present in "
                    f"the satellite images for site {test_site}."
                )
            else:
                excluded_time_index = int(matching_indices[0])

        label = numpy.where(
            numpy.isnan(uav_labels.values),
            utils.UAV_NAN_CLASS,
            uav_labels.values,
        ).astype(numpy.int64)
        height, width = label.shape
        if verbose:
            print(f"\tTile images to {int(numpy.ceil(height/stride))*int(numpy.ceil(width/stride))}:"
                  f" ({int(numpy.ceil(height/stride))}, {int(numpy.ceil(width/stride))})")

        for time_index in range(len(satellite_data["time"])):
            if time_index == excluded_time_index:
                if verbose:
                    print(f"\tSkipping held out date at time index {time_index}")
                continue

            # Select a fixed, ordered set of bands so every tile has the same channel count
            image = satellite_data[UNET_BANDS].isel(time=time_index).to_array().values

            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    image_slice = image[:, y:y + tile_size, x:x + tile_size]
                    label_slice = label[y:y + tile_size, x:x + tile_size]
                    valid_labels = label_slice != utils.UAV_NAN_CLASS
                    if numpy.isnan(image_slice).all() or not valid_labels.any():
                        if verbose:
                            print(f"\t\tSkipping tile at position ({y}, {x}) due to NaN values.")
                        continue

                    image_tile = numpy.full(
                        (image.shape[0], tile_size, tile_size),
                        numpy.nan,
                        dtype=numpy.float32,
                    )
                    label_tile = numpy.full(
                        (tile_size, tile_size),
                        utils.UAV_NAN_CLASS,
                        dtype=numpy.int64,
                    )
                    slice_height, slice_width = label_slice.shape
                    image_tile[:, :slice_height, :slice_width] = image_slice
                    label_tile[:slice_height, :slice_width] = label_slice
                    tiles.append(image_tile)
                    labels.append(label_tile)

    if not tiles:
        raise ValueError(
            "No tiles were generated - none of the training sites had satellite "
            "image data available."
        )
    tiles = numpy.stack(tiles).astype(numpy.float32)
    labels = numpy.stack(labels).astype(numpy.int64)

    return tiles, labels


def class_counts_per_tile(labels: numpy.ndarray) -> numpy.ndarray:
    """Return a (n_tiles, n_classes) matrix of per-class pixel counts within each
    label tile, ignoring utils.UAV_NAN_CLASS. Used to stratify tile train/test splits
    so the overall per-class pixel distribution (not just a single dominant class)
    is preserved between the two sets."""

    valid_classes = numpy.unique(labels[labels != utils.UAV_NAN_CLASS])
    class_counts = numpy.zeros((labels.shape[0], len(valid_classes)), dtype=numpy.int64)
    for index, label_tile in enumerate(labels):
        tile_classes, tile_counts = numpy.unique(label_tile, return_counts=True)
        for tile_class, tile_count in zip(tile_classes, tile_counts):
            if tile_class == utils.UAV_NAN_CLASS:
                continue
            class_counts[index, numpy.searchsorted(valid_classes, tile_class)] = tile_count
    return class_counts


def randomise_tiles_to_test_and_training(
    tiles: numpy.ndarray,
    labels: numpy.ndarray,
    test_threshold: float,
    random_state: int = None,
) -> tuple:
    """Split tiles/labels (e.g. from load_samples_and_tile or
    load_samples_and_tile_excluding_one_site_date) into train and test sets. Mirrors
    randomise_to_test_and_training_across_sites, but for tiled U-Net data: uses
    iterative multilabel stratification (iterstrat) over the full per-class pixel
    count matrix for each tile, so the overall per-class pixel distribution is
    balanced between the two sets rather than just a single dominant class per tile.
    Returns (train_tiles, train_labels, test_tiles, test_labels)."""

    class_counts = class_counts_per_tile(labels)

    splitter = iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_threshold, random_state=random_state
    )
    train_indices, test_indices = next(splitter.split(tiles, class_counts))

    return tiles[train_indices], labels[train_indices], tiles[test_indices], labels[test_indices]


class UNetClassifier(pytorch_lightning.LightningModule):
    """Pixel-wise classifier using a pretrained Sentinel-2 UNet++ model."""

    def __init__(
        self,
        in_channels: int,
        tile_size: int,
        num_classes: int,
        band_mean: list,
        band_std: list,
        learning_rate: float = 1e-4,
        monitor_class_ids: dict = None,
        pretrained: bool = True,
        pretrained_model_repo: str = UNET_PRETRAINED_REPO,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer(
            "band_mean",
            torch.as_tensor(band_mean, dtype=torch.float32).reshape(1, -1, 1, 1),
        )
        self.register_buffer(
            "band_std",
            torch.as_tensor(band_std, dtype=torch.float32).reshape(1, -1, 1, 1),
        )
        self.unet = segmentation_models_pytorch.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        if pretrained:
            self._load_pretrained_weights(pretrained_model_repo)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=int(utils.UAV_NAN_CLASS))

    def _load_pretrained_weights(self, repo_id: str):
        checkpoint_file = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename="model.pth",
        )
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format from {repo_id}")

        # The Hugging Face checkpoint has a two-class segmentation head. Keep the
        # pretrained encoder/decoder and leave this project's class-specific head
        # randomly initialized.
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if not key.removeprefix("model.").startswith("segmentation_head.")
        }
        missing_keys, unexpected_keys = self.unet.load_state_dict(
            state_dict, strict=False
        )
        unexpected_keys = [
            key for key in unexpected_keys if not key.startswith("segmentation_head.")
        ]
        if unexpected_keys:
            raise ValueError(
                f"Unexpected keys in pretrained checkpoint from {repo_id}: "
                f"{unexpected_keys[:5]}"
            )
        if not any(key.startswith("encoder.") for key in state_dict):
            raise ValueError(
                f"No UNet++ encoder weights were found in the checkpoint from {repo_id}."
            )
        if not any(key.startswith("decoder.") for key in state_dict):
            raise ValueError(
                f"No UNet++ decoder weights were found in the checkpoint from {repo_id}."
            )
        if not missing_keys:
            raise ValueError(
                "The pretrained checkpoint unexpectedly contains the project's "
                "segmentation head."
            )

    def forward(self, tiles):
        tiles = torch.where(torch.isfinite(tiles), tiles, self.band_mean)
        tiles = (tiles - self.band_mean) / self.band_std
        return self.unet(tiles)

    def training_step(self, batch, batch_index):
        batch_tiles, batch_labels = batch
        loss = self.loss_function(self(batch_tiles), batch_labels)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_tiles.shape[0])
        return loss

    def _per_class_iou(self, predicted_labels, true_labels):
        """Per-batch IoU for each class in monitor_class_ids, ignoring UAV_NAN_CLASS
        pixels. Averaged across batches/epochs via self.log - an approximation of
        the true epoch-wide IoU (which would need accumulated intersection/union),
        but sufficient to track whether the classes of interest are improving."""
        valid = true_labels != int(utils.UAV_NAN_CLASS)
        ious = {}
        for class_name, class_id in self.hparams.monitor_class_ids.items():
            predicted_class = (predicted_labels == class_id) & valid
            true_class = (true_labels == class_id) & valid
            union = (predicted_class | true_class).sum()
            if union == 0:
                continue
            ious[class_name] = (predicted_class & true_class).sum().float() / union.float()
        return ious

    def validation_step(self, batch, batch_index):
        batch_tiles, batch_labels = batch
        logits = self(batch_tiles)
        loss = self.loss_function(logits, batch_labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_tiles.shape[0])

        if self.hparams.monitor_class_ids:
            predicted_labels = logits.argmax(dim=1)
            for class_name, iou in self._per_class_iou(predicted_labels, batch_labels).items():
                self.log(
                    f"val_iou_{class_name.replace(' ', '_')}", iou,
                    on_epoch=True, on_step=False, batch_size=batch_tiles.shape[0]
                )
        return loss

    def configure_optimizers(self):
        # Can hve a learning rate decay - later - more complexity
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def train_unet_classifier(
    tiles: numpy.ndarray,
    labels: numpy.ndarray,
    models_path: pathlib.Path,
    num_classes: int,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    val_tiles: numpy.ndarray = None,
    val_labels: numpy.ndarray = None,
    monitor_class_ids: dict = None,
    early_stopping_patience: int = 15,
    pretrained: bool = True,
    pretrained_model_repo: str = UNET_PRETRAINED_REPO,
):
    """Train a U-Net pixel classifier (see UNetClassifier) with PyTorch Lightning.
    `tiles` is (N, bands, tile_size, tile_size) and `labels` is (N, tile_size,
    tile_size) of integer class ids, e.g. from load_samples_and_tile.

    If val_tiles/val_labels are given, validation loss (and per-class IoU for any
    classes in monitor_class_ids, e.g. {"Seagrass": 1, "Ulva": 5}) is tracked each
    epoch, early stopping is enabled (monitor="val_loss", patience=
    early_stopping_patience), and the best checkpoint is selected on val_loss
    instead of train_loss. epochs then acts as a ceiling - early stopping decides
    the actual number of epochs run. Without a validation set, behaviour is
    unchanged from before (fixed epochs, checkpoint on train_loss)."""

    print("\tTrain a U-Net Model") # consider caching to disk and loading
    band_mean = numpy.nanmean(tiles, axis=(0, 2, 3))
    band_std = numpy.nanstd(tiles, axis=(0, 2, 3))
    band_mean = numpy.where(numpy.isfinite(band_mean), band_mean, 0.0)
    band_std = numpy.where(numpy.isfinite(band_std) & (band_std > 0), band_std, 1.0)

    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(tiles, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_data_loader = None
    if val_tiles is not None and val_labels is not None:
        val_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(val_tiles, dtype=torch.float32),
            torch.as_tensor(val_labels, dtype=torch.long),
        )
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNetClassifier(
        in_channels=tiles.shape[1],
        tile_size=tiles.shape[-1],
        num_classes=num_classes,
        band_mean=band_mean.tolist(),
        band_std=band_std.tolist(),
        learning_rate=learning_rate,
        monitor_class_ids=monitor_class_ids,
        pretrained=pretrained,
        pretrained_model_repo=pretrained_model_repo,
    )

    monitor_metric = "val_loss" if val_data_loader is not None else "train_loss"
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        dirpath=models_path,
        filename="unet-{epoch:02d}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpoint_callback]
    if val_data_loader is not None:
        callbacks.append(pytorch_lightning.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=early_stopping_patience
        ))

    trainer = pytorch_lightning.Trainer(max_epochs=epochs, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=data_loader, val_dataloaders=val_data_loader) # fit method can pass a checkpoint
    best_path = checkpoint_callback.best_model_path
    return model, best_path


def load_samples(
    training_sites: list,
    samples_path: pathlib.Path,
) -> pandas.DataFrame:
    """Combine all training datasets and return."""

    print(f"\tLoad in sites: {training_sites}")
    samples_dataframe = []
    for training_site in training_sites:
        samples_file = samples_path / f"{training_site}_training_data.csv"
        if not samples_file.exists():
            print(f"\tWARNING - no training data for site {training_site}. Skipping this site."
                  "Rerun `extract_training_data` for this site if data is expected.")
            continue
        samples_dataframe.append(pandas.read_csv(samples_file))
    samples_dataframe = pandas.concat(samples_dataframe, ignore_index=True)

    return samples_dataframe


def load_samples_excluding_test_date(
    training_sites: list,
    samples_path: pathlib.Path,
    test_site: str,
    test_date: str,
) -> pandas.DataFrame:
    """Combine all training datasets and return, excluding the test site at the given date."""

    print(f"\tLoad in sites: {training_sites}")
    samples_dataframe = []
    for training_site in training_sites:
        samples_file = samples_path / f"{training_site}_training_data.csv"
        sample_dataframe = pandas.read_csv(samples_file)
        if training_site == test_site:
            sample_dates = pandas.to_datetime(sample_dataframe["time"]).dt.normalize()
            test_date_normalized = pandas.to_datetime(test_date).normalize()
            if not sample_dates.eq(test_date_normalized).any():
                print(
                    f"Warning: test date {test_date} is not present in "
                    f"the samples for site {test_site}."
                )
            sample_dataframe = sample_dataframe[
                sample_dates != test_date_normalized
            ]

        samples_dataframe.append(sample_dataframe)
    samples_dataframe = pandas.concat(samples_dataframe, ignore_index=True)

    return samples_dataframe


def map_satellite_ids_into_samples(
    samples_dataframe: pandas.DataFrame,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
    drop_scl_classes_to_ignore: bool,
):
    """For a given samples dataframe then drop the UAV classes to ignore
    and map the UAV classes to the satellite classes. Option to drop SCL
    cloud classes."""

    print("\tMap UAV training ids to the specified satellite training ids")
    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )

    # Drop UAV image specific classes - e.g. Shadow, Glare
    class_ids_to_ignore = [uav_training_labels[key] for key in uav_classes_to_ignore]
    samples_dataframe = samples_dataframe[
        ~samples_dataframe["uav_class_id"].isin(class_ids_to_ignore)
    ]

    # Map the UAV classes to those used in the satellite imagery
    samples_dataframe["satellite_class_id"] = samples_dataframe["uav_class_id"]
    for key in satellite_from_uav_classes.keys():
        class_ids_to_map = [
            uav_training_labels[key] for key in satellite_from_uav_classes[key]
        ]
        samples_dataframe.loc[
            samples_dataframe["uav_class_id"].isin(class_ids_to_map),
            "satellite_class_id",
        ] = satellite_classes[key]

    # Drop SCL classes to ignore
    if drop_scl_classes_to_ignore:
        samples_dataframe = samples_dataframe[
            ~samples_dataframe["SCL"].isin(sentinel2.SCL_TO_IGNORE)
        ]
    return samples_dataframe


def map_satellite_ids_into_labels_array(
    labels_array: numpy.ndarray,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
) -> numpy.ndarray:
    """Map UAV class IDs in an array to satellite class IDs.

    Ignored UAV classes are set to utils.UAV_NAN_CLASS. The input array is not modified.
    """

    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )
    source_labels = labels_array.copy()
    satellite_labels = source_labels.astype(numpy.int64, copy=True)

    class_ids_to_ignore = [
        uav_training_labels[key] for key in uav_classes_to_ignore
    ]
    satellite_labels[numpy.isin(source_labels, class_ids_to_ignore)] = (
        utils.UAV_NAN_CLASS
    )

    for satellite_class_name, uav_class_names in satellite_from_uav_classes.items():
        class_ids_to_map = [
            uav_training_labels[class_name] for class_name in uav_class_names
        ]
        satellite_labels[numpy.isin(source_labels, class_ids_to_map)] = (
            satellite_classes[satellite_class_name]
        )

    return satellite_labels


def time_index_for_date(data, date: str) -> int:
    """Return the index of the time step matching the given date (e.g. "2025-04-27")."""

    # Match on the calendar day as times include a time of day. Use the xarray dt
    # accessor as times may be cftime rather than datetime64.
    dates = data["time"].dt.strftime("%Y-%m-%d").values
    matching_indices = numpy.flatnonzero(
        dates == pandas.to_datetime(date).strftime("%Y-%m-%d")
    )
    if len(matching_indices) == 0:
        raise ValueError(f"No image for date {date}. Available dates: {list(dates)}.")

    return int(matching_indices[0])


def predict_site_for_date(
    test_satellite_file: pathlib.Path,
    polygon_file: pathlib.Path,
    model_file: pathlib.Path,
    model_feature_names_file: pathlib.Path,
    date: str,
):
    """Predict classes for the satellite image on the given date (e.g. "2025-04-27").
    Load feature names to ensure the same order."""

    satellite_data = utils.load_satellite(filename=test_satellite_file)
    uav_polygon = geopandas.read_file(polygon_file)
    model = joblib.load(model_file)
    model_columns = pandas.read_csv(model_feature_names_file)

    time_index = time_index_for_date(data=satellite_data, date=date)

    print(f"\tPredict satellite image for date {date}")
    observations_to_predict = (
        satellite_data.isel(time=time_index)[model_columns.columns]
        .to_array()
        .stack(dims=["y", "x"])
        .transpose()
    )

    predictions = model.predict(observations_to_predict)
    # probabilities = model.predict_proba(observations_to_predict)
    predictions = predictions.reshape(
        len(satellite_data.y), len(satellite_data.x)
    ).astype(utils.CLASSIFICATION_DTYPE)

    predictions = xarray.DataArray(
        [predictions],
        coords={
            "time": numpy.atleast_1d(satellite_data["time"][time_index]),
            "y": satellite_data.y,
            "x": satellite_data.x,
        },
        dims=["time", "y", "x"],
    )

    predictions.rio.write_crs(input_crs=utils.CRS_NZTM, inplace=True)
    predictions = predictions.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )

    return predictions


def predict_site(
    test_satellite_file: pathlib.Path,
    polygon_file: pathlib.Path,
    model_file: pathlib.Path,
    model_feature_names_file: pathlib.Path
):
    """Predict classes for satellite images across all time steps."""

    satellite_data = utils.load_satellite(filename=test_satellite_file)
    predictions = []
    satellite_dates = satellite_data["time"].dt.strftime("%Y-%m-%d").values
    print(f"\tPredict for {len(satellite_dates)} satellite images")
    for date in satellite_dates:
        predictions.append(
            predict_site_for_date(
                test_satellite_file=test_satellite_file,
                polygon_file=polygon_file,
                model_file=model_file,
                model_feature_names_file=model_feature_names_file,
                date=date,
            )
        )

    print("\tCombine predictions and write conventions")
    predictions = xarray.concat(predictions, dim="time")
    utils.write_netcdf_conventions_in_place(predictions)

    return predictions, satellite_data


def predict_tile_probabilities_unet(
    model,
    image: numpy.ndarray,
    tile_size: int,
    stride: int,
    device,
) -> numpy.ndarray:
    """Predict per-class probabilities across a single (bands, height, width) image
    using overlapping tiles (stride < tile_size), averaging softmax probabilities in
    overlapping regions to reduce blockiness at tile boundaries. Returns a
    (num_classes, height, width) array of mean probabilities."""

    num_classes = model.hparams.num_classes
    height, width = image.shape[1], image.shape[2]

    probability_sums = numpy.zeros((num_classes, height, width), dtype=numpy.float32)
    overlap_counts = numpy.zeros((height, width), dtype=numpy.float32)

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Extract and pad tile
            image_slice = image[:, y:y + tile_size, x:x + tile_size]

            image_tile = numpy.full(
                (image.shape[0], tile_size, tile_size),
                numpy.nan,
                dtype=numpy.float32,
            )
            slice_height, slice_width = image_slice.shape[1], image_slice.shape[2]
            image_tile[:, :slice_height, :slice_width] = image_slice

            # Convert to tensor and predict
            with torch.no_grad():
                tile_tensor = torch.as_tensor(
                    image_tile[numpy.newaxis, ...], dtype=torch.float32, device=device
                )
                logits = model(tile_tensor)  # (1, num_classes, tile_size, tile_size)
                tile_probabilities = (
                    torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                )

            # Accumulate probabilities for the valid (non-padded) region only
            probability_sums[:, y:y + slice_height, x:x + slice_width] += (
                tile_probabilities[:, :slice_height, :slice_width]
            )
            overlap_counts[y:y + slice_height, x:x + slice_width] += 1

    if (overlap_counts == 0).any():
        print("\t\tWarning: some pixels were not predicted.")

    return probability_sums / numpy.maximum(overlap_counts, 1)


def predict_site_for_date_unet(
    test_satellite_file: pathlib.Path,
    polygon_file: pathlib.Path,
    checkpoint_file: pathlib.Path,
    date: str,
    tile_size: int = 128,
    stride: int = 64,
):
    """Predict classes for the satellite image on the given date (e.g. "2025-04-27")
    using a trained U-Net checkpoint. Mirrors predict_site_for_date, but for the U-Net
    model - see predict_site_unet for the multi-date equivalent."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\tLoading U-Net checkpoint from {checkpoint_file}")
    model = UNetClassifier.load_from_checkpoint(checkpoint_file)
    model = model.to(device)
    model.eval()

    satellite_data = utils.load_satellite(filename=test_satellite_file, chunks=None)
    uav_polygon = geopandas.read_file(polygon_file)

    time_index = time_index_for_date(data=satellite_data, date=date)

    print(f"\tPredict satellite image for date {date}")
    image = satellite_data[UNET_BANDS].isel(time=time_index).to_array().values
    mean_probabilities = predict_tile_probabilities_unet(
        model=model, image=image, tile_size=tile_size, stride=stride, device=device
    )
    predictions = numpy.argmax(mean_probabilities, axis=0).astype(utils.CLASSIFICATION_DTYPE)

    predictions = xarray.DataArray(
        [predictions],
        coords={
            "time": numpy.atleast_1d(satellite_data["time"][time_index]),
            "y": satellite_data.y,
            "x": satellite_data.x,
        },
        dims=["time", "y", "x"],
    )

    predictions.rio.write_crs(input_crs=utils.CRS_NZTM, inplace=True)
    predictions = predictions.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )

    return predictions


def predict_site_unet(
    test_satellite_file: pathlib.Path,
    polygon_file: pathlib.Path,
    checkpoint_file: pathlib.Path,
    tile_size: int = 128,
    stride: int = 64,
):
    """Predict classes for satellite images across all time steps using a trained U-Net
    checkpoint. Tiles the satellite imagery for inference; overlapping tiles (stride <
    tile_size) have their softmax probabilities averaged before taking the final class,
    which reduces blockiness at tile boundaries. Returns predictions and satellite data
    as xarray DataArrays."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\tLoading U-Net checkpoint from {checkpoint_file}")
    model = UNetClassifier.load_from_checkpoint(checkpoint_file)
    model = model.to(device)
    model.eval()

    satellite_data = utils.load_satellite(filename=test_satellite_file, chunks=None)
    uav_polygon = geopandas.read_file(polygon_file)

    predictions_list = []
    satellite_dates = satellite_data["time"].dt.strftime("%Y-%m-%d").values
    print(f"\tPredict for {len(satellite_dates)} satellite images")

    for time_index in range(len(satellite_data["time"])):
        # Select a fixed, ordered set of bands
        image = satellite_data[UNET_BANDS].isel(time=time_index).to_array().values
        mean_probabilities = predict_tile_probabilities_unet(
            model=model, image=image, tile_size=tile_size, stride=stride, device=device
        )
        predictions = numpy.argmax(mean_probabilities, axis=0).astype(utils.CLASSIFICATION_DTYPE)

        # Convert to DataArray and clip to polygon
        predictions_da = xarray.DataArray(
            [predictions],
            coords={
                "time": numpy.atleast_1d(satellite_data["time"][time_index]),
                "y": satellite_data.y,
                "x": satellite_data.x,
            },
            dims=["time", "y", "x"],
        )
        predictions_da.rio.write_crs(input_crs=utils.CRS_NZTM, inplace=True)
        predictions_da = predictions_da.rio.clip(
            uav_polygon.geometry, all_touched=True, drop=True
        )

        predictions_list.append(predictions_da)

    print("\tCombine predictions and write conventions")
    predictions = xarray.concat(predictions_list, dim="time")
    utils.write_netcdf_conventions_in_place(predictions)

    return predictions, satellite_data


def predict_samples(
    test_dataframe: pandas.DataFrame,
    model_file: pathlib.Path,
    model_feature_names_file: pathlib.Path,
):
    """Predict classes for each row of a samples dataframe. Load feature names to
    ensure the same order. Returns a copy with a "predicted_class_id" column."""

    model = joblib.load(model_file)
    model_columns = pandas.read_csv(model_feature_names_file)

    print(f"\tPredict {len(test_dataframe)} samples")
    test_dataframe = test_dataframe.copy()
    test_dataframe["predicted_class_id"] = model.predict(
        test_dataframe[model_columns.columns]
    )

    return test_dataframe


def predict_samples_unet(
    test_tiles: numpy.ndarray,
    checkpoint_file: pathlib.Path,
    batch_size: int = 8,
) -> numpy.ndarray:
    """Predict classes for each tile using a trained U-Net checkpoint (see
    UNetClassifier and train_unet_classifier). `test_tiles` is (N, bands, tile_size,
    tile_size), e.g. from randomise_tiles_to_test_and_training. Returns predicted
    labels as (N, tile_size, tile_size) integer class ids."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\tLoading U-Net checkpoint from {checkpoint_file}")
    model = UNetClassifier.load_from_checkpoint(checkpoint_file)
    model = model.to(device)
    model.eval()

    print(f"\tPredict {len(test_tiles)} tiles")
    predicted_labels = []
    with torch.no_grad():
        for start in range(0, len(test_tiles), batch_size):
            batch_tiles = torch.as_tensor(
                test_tiles[start:start + batch_size], dtype=torch.float32, device=device
            )
            logits = model(batch_tiles)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predicted_labels.append(batch_predictions)

    predicted_labels = numpy.concatenate(predicted_labels).astype(utils.CLASSIFICATION_DTYPE)

    return predicted_labels


def load_truth_and_predictions(
    test_uav_file,
    uav_labels_file,
    prediction_file,
    satellite_classes,
    satellite_from_uav_classes,
    uav_classes_to_ignore,
    polygon_file,
    match_satellite_resolution: bool,
):
    """Load the UAV classification and satellite predictions, map the UAV classes to
    the satellite classes, clip both to the polygon and align them onto a common grid.
    If match_satellite_resolution the UAV data is coarsened to the satellite resolution
    taking the mode, otherwise the predictions are reindexed onto the UAV grid."""

    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )

    # Load in images
    # UAV classification is int8 (~1.1 GB) so load without dask chunks: all
    # subsequent .where() and rio.clip() operations then run eagerly in the
    # main process rather than building a lazy graph that dask workers must
    # compute simultaneously (which can multiply peak memory 4–8x).
    uav_training_data = utils.load_classification(filename=test_uav_file, chunks=None, masked=False)
    sat_prediction_data = utils.load_classification(filename=prediction_file, chunks=True, masked=False)
    uav_polygon = geopandas.read_file(polygon_file)

    # drop classes to ignore
    uav_training_data_reclassed = uav_training_data.where(
        ~uav_training_data.isin(
            [uav_training_labels[key] for key in uav_classes_to_ignore]
        ),
        utils.UAV_NAN_CLASS,
    )

    # convert UAV to satellite classifications
    for key in satellite_from_uav_classes.keys():
        class_ids_to_map = [
            uav_training_labels[key] for key in satellite_from_uav_classes[key]
        ]
        uav_training_data_reclassed = uav_training_data_reclassed.where(
            ~uav_training_data.isin(class_ids_to_map), satellite_classes[key]
        )

    # Force free memory
    del uav_training_data
    gc.collect()

    # Clip both to polygon
    uav_training_data_reclassed = uav_training_data_reclassed.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )
    sat_prediction_data = sat_prediction_data.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )

    if match_satellite_resolution:
        # Align UAV to satellite then coarsen taking the mode
        uav_training_data_reclassed, upsample_rate = sampling.align_fine_grid_to_coarse_grid(
            fine_grid=uav_training_data_reclassed, coarse_grid=sat_prediction_data)

        def get_mode(array, axis):
            """returns the mode values only; ignore counts"""
            return scipy.stats.mode(array, axis=axis).mode
        uav_training_data_reclassed = uav_training_data_reclassed.coarsen(
                x=upsample_rate, y=upsample_rate, boundary="trim"
                ).reduce(get_mode)

    # reindex to match the UAV data - also ensures even x & y spacing
    sat_prediction_data = sat_prediction_data.reindex_like(
        uav_training_data_reclassed, method="nearest"
    )

    return uav_training_data_reclassed, sat_prediction_data


def extract_truth_and_predictions(
    uav_training_data_reclassed,
    sat_prediction_data,
    time_index: int,
):
    """Return the matched UAV truth and satellite prediction values for one time index."""

    # Compute mask to a numpy array before using it so it is not
    # recomputed twice (once for truth, once for predictions) and so
    # that it does not trigger a concurrent dask computation alongside
    # uav_training_data_reclassed.values.
    mask = sat_prediction_data.isel(time=time_index).notnull().values
    truth = uav_training_data_reclassed.values[mask]
    predictions = sat_prediction_data.isel(time=time_index).values[mask]
    del mask
    gc.collect()

    # drop any NaN interpolated from clipped areas in the prediction
    mask = (
        ~numpy.isnan(predictions)
        & ~numpy.isnan(truth)
        & (truth != utils.UAV_NAN_CLASS)
    )

    return truth[mask], predictions[mask]


def confusion_matrix_of_site_for_date(
    test_uav_file,
    uav_labels_file,
    prediction_file,
    satellite_classes,
    satellite_from_uav_classes,
    uav_classes_to_ignore,
    polygon_file,
    method_2_threshold,
    date,
    match_satellite_resolution=False,
):
    """Confusion matrix for the predictions on a single date (e.g. "2025-04-27")."""

    resolution_label = (
        f"_{sentinel2.S2_RESOLUTION}_resolution" if match_satellite_resolution else ""
    )
    plot_filename = prediction_file.with_name(
        f"{prediction_file.stem}_confusion_matrix{resolution_label}_date_{date}.png"
    )
    if plot_filename.exists():
        print(f"{plot_filename.name} already exists."
              "Skipping. Delete plots if you want them regenerated.")
        return

    uav_training_data_reclassed, sat_prediction_data = load_truth_and_predictions(
        test_uav_file=test_uav_file,
        uav_labels_file=uav_labels_file,
        prediction_file=prediction_file,
        satellite_classes=satellite_classes,
        satellite_from_uav_classes=satellite_from_uav_classes,
        uav_classes_to_ignore=uav_classes_to_ignore,
        polygon_file=polygon_file,
        match_satellite_resolution=match_satellite_resolution,
    )

    print(f"\tExtract truth and predictions for date: {date}")
    truth, predictions = extract_truth_and_predictions(
        uav_training_data_reclassed=uav_training_data_reclassed,
        sat_prediction_data=sat_prediction_data,
        time_index=time_index_for_date(data=sat_prediction_data, date=date),
    )

    # Force free memory
    del uav_training_data_reclassed
    del sat_prediction_data
    gc.collect()

    plotting.plot_confusion_matrix(truth=truth,
                                   predictions=predictions,
                                   class_names=satellite_classes,
                                   plot_filename=plot_filename,
                                   title=f"{int(method_2_threshold*100)}% Sampling Purity; {date}",
    )

    return truth, predictions


def confusion_matrix_of_site(
    test_uav_file,
    uav_labels_file,
    prediction_file,
    satellite_classes,
    satellite_from_uav_classes,
    uav_classes_to_ignore,
    polygon_file,
    method_2_threshold,
    match_satellite_resolution=False,
):

    debug=False

    resolution_label = (
            f"_{sentinel2.S2_RESOLUTION}_resolution" if match_satellite_resolution else ""
        )

    # Exit if the plots have already been created.
    overall_plot_filename = prediction_file.with_name(
            f"{prediction_file.stem}_confusion_matrix{resolution_label}_all_dates.png"
        )
    if overall_plot_filename.exists():
        print(f"{overall_plot_filename.name} already exists."
              "Skipping. Delete plots if you want them regenerated.")
        return

    print("Match satellite resolution to UAV then compare predictions to UAV")

    uav_training_data_reclassed, sat_prediction_data = load_truth_and_predictions(
        test_uav_file=test_uav_file,
        uav_labels_file=uav_labels_file,
        prediction_file=prediction_file,
        satellite_classes=satellite_classes,
        satellite_from_uav_classes=satellite_from_uav_classes,
        uav_classes_to_ignore=uav_classes_to_ignore,
        polygon_file=polygon_file,
        match_satellite_resolution=match_satellite_resolution,
    )

    # Pull out the predictions vs ground truth
    all_truth = []
    all_predictions = []
    for time_index in range(len(sat_prediction_data["time"])):

        plot_filename = prediction_file.with_name(
            f"{prediction_file.stem}_confusion_matrix{resolution_label}_date_{time_index}.png"
        )

        print(f"\tExtract truth and predictions time index: {time_index}")
        truth, predictions = extract_truth_and_predictions(
            uav_training_data_reclassed=uav_training_data_reclassed,
            sat_prediction_data=sat_prediction_data,
            time_index=time_index,
        )
        all_truth.append(truth)
        all_predictions.append(predictions)

        if debug:
            if plot_filename.exists():
                print(f"\t\t{plot_filename.name} exists. Skipping. Delete if you want regenerated.")
                continue
            print(f"\t\tConstruct confusion matrix")
            plotting.plot_confusion_matrix(truth=truth,
                                           predictions=predictions,
                                           class_names=satellite_classes,
                                           plot_filename=plot_filename,
                                           title= f"{int(method_2_threshold*100)}% Sampling Purity; Time index {time_index}",
            )
            matplotlib.pyplot.close()

    # Force free memory
    del uav_training_data_reclassed
    del sat_prediction_data
    gc.collect()
    print("Overall confusion matrix across prediction dates")
    all_truth = numpy.concatenate(all_truth)
    all_predictions = numpy.concatenate(all_predictions)
    plotting.plot_confusion_matrix(truth=all_truth,
                                   predictions=all_predictions,
                                   class_names=satellite_classes,
                                   plot_filename=overall_plot_filename,
                                   title= f"{int(method_2_threshold*100)}% Sampling Purity",
    )

    return all_truth, all_predictions


def confusion_matrix_of_pixels(
    predictions: typing.Union[pandas.Series, numpy.ndarray],
    truth: typing.Union[pandas.Series, numpy.ndarray],
    satellite_classes: dict,
    plot_filename: pathlib.Path,
    plot_title: str,
):
    """Calculate the normalized confusion matrix for pixel classifications."""

    # drop any NaN class values from the truth prior to calculating the confusion matrix
    mask = truth != utils.UAV_NAN_CLASS

    plotting.plot_confusion_matrix(
        truth=truth[mask],
        predictions=predictions[mask],
        class_names=satellite_classes,
        plot_filename=plot_filename,
        title=plot_title,
        )
