import itertools
from collections import deque
from pathlib import Path
from typing import Iterator, Tuple

import tensorflow as tf

from src.pipeline.classes.dataset import MNISTAudioDataset
from src.pipeline.classes.extractor import MelSpectrogramExtractor
from src.utils.io_utils import AudioFileLoader, list_subdir, list_files
from src.pipeline.classes.model import CNNModel, InputShape, ConvLayer
from src.pipeline.classes.preprocessing import AudioProcessor


def dataset_dir_structure(cfg: dict) -> Path:
    dp = (Path(cfg.get("working_dir")) / Path(cfg.get("dataset_dir"))).resolve()
    print(f"dataset path: {dp}")
    Path.mkdir(dp, parents=True, exist_ok=True)
    # create class dirs
    deque(
        map(
            lambda x: Path.mkdir((dp / f"class-{x}").resolve(), exist_ok=True),
            range(cfg.get("n_classes"))
        )
    )

    return dp


def get_audio_paths(path: Path) -> Iterator[Path]:
    # Lazy loading, i.e. file paths
    return AudioFileLoader(path).load()


def process_audio(cfg: dict, out_path: Path) -> None:
    """Process audio data into mel-spectrograms
    and save result in specified out path.

    :param cfg: configuration for processor
    :param out_path: out path for storage
    :return: None
    """
    AudioProcessor(
        extractor=MelSpectrogramExtractor(),
        in_path=get_audio_paths(Path(cfg.get("input_dir"))),
        out_path=out_path,
        sr=cfg.get("sample_rate"),
        dur=cfg.get("keep_duration")
    ).process()


def get_in_data_paths(dp: Path):
    """Compute an iterator of paths for input data
    to be used in building the dataset.

    :param dp: dataset path
    :return: iterator of all paths for dataset
    """
    sub_dirs = list_subdir(dp)
    files_in_sub = map(lambda x: list(list_files(x)), sub_dirs)

    return itertools.chain([f for fs in files_in_sub for f in fs])


def create_mnist_audio_dataset(dp: Path) -> tf.data.Dataset:
    in_data_paths = get_in_data_paths(dp)
    return MNISTAudioDataset(in_data_paths).populate()


def build_mnist_audio_model(model_cfg: dict):
    cls: dict = model_cfg.get("conv_layer_specs")
    model = CNNModel(
        in_shape=InputShape(128, 11, 1),
        n_class=model_cfg.get("n_classes"),
        n_layers=model_cfg.get("n_layers"),
        conv_layer_spec=ConvLayer(
            cls.get("filters"),
            cls.get("kernel_size"),
            cls.get("padding"),
            cls.get("activation")
        )).compile()
    model.summary()

    return model


def split_dataset(dataset: tf.data.Dataset, validation_split: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Split dataset into train and validation sets
    given validation split.

    :param dataset: tensorflow dataset
    :param validation_split: validation/full dataset ratio
    :return: train and validation tensorflow dataset
    """
    train_set_size: int = int(50 * (1 - validation_split))
    train_set = dataset.take(train_set_size)
    validation_set = dataset.skip(train_set_size).take(50 - train_set_size)

    return train_set, validation_set


def train_and_evaluate(
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        train_cfg: dict) -> Tuple[tf.keras.Model, dict]:
    epochs: int = train_cfg.get("epochs")
    batch_size: int = train_cfg.get("batch_size")
    history = model.fit(train_dataset.batch(batch_size), epochs=epochs)
    print(history.history)

    evaluation: dict = model.evaluate(validation_dataset.batch(batch_size), verbose=1, return_dict=True)
    print(evaluation)

    return model, evaluation
