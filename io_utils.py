import warnings
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Iterator, Union

import librosa
import numpy as np
import yaml
from yaml.scanner import ScannerError


def list_subdir(main_dir: Path) -> Iterator[Path]:
    return filter(lambda x: x.is_dir(), main_dir.iterdir())


def list_files(main_dir: Path) -> Iterator[Path]:
    return filter(lambda x: x.is_file(), main_dir.iterdir())


def select_wav_files(fs: Iterator[Path]) -> Iterator[Path]:
    return filter(lambda x: x.suffix == ".wav", fs)


def path_stem(path: Path) -> str:
    return path.stem


def load_audio(path: Path) -> np.ndarray:
    y, sr = librosa.load(path, sr=8000)
    return y


def save_to_npy(a: np.ndarray, p: Path) -> None:
    Path.mkdir(p.parent) if not p.parent.is_dir() else None
    np.save(str(p), a)


def load_config(filename: Union[str, Path]) -> dict:
    with open(filename, "r") as f:
        try:
            return yaml.safe_load(f)
        except ScannerError:
            warnings.warn("Incorrect yaml file, check syntax")
            raise


class Loader(ABC):
    @abstractmethod
    def load(self):
        raise NotImplemented


class AudioFileLoader(Loader):

    __slots__ = "folder"

    def __init__(self, folder: Path):
        self.folder = folder
        print(f"AudioFileLoader input folder: {self.folder}")

    def load(self) -> Iterator[Path]:
        # obs: sort of lazy loading
        files = list_files(Path(self.folder))
        return select_wav_files(files)


