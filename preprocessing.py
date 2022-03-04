from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Union

from extractor import Extractor
from io_utils import path_stem, load_audio, save_to_npy
from utils import pad


class Processor(ABC):

    @abstractmethod
    def process(self):
        raise NotImplemented


class AudioProcessor(Processor):
    __slots__ = ("extractor", "sr", "dur", "in_path", "out_path")

    def __init__(
            self,
            extractor: Extractor,
            in_path: Iterator[Path],
            out_path: Path,
            sr: Optional[int] = None,
            dur: Optional[Union[int, float]] = None
    ):
        self.extractor = extractor
        self.in_path = in_path
        self.out_path = out_path
        self.sr = sr
        self.dur = dur

    def process(self):
        # Compute and save mel-spectrograms
        for f in self.in_path:
            stem = path_stem(f)
            digit = stem[0]
            y = load_audio(f)
            y = pad(y, int(self.sr * self.dur) - y.shape[0])
            a = self.extractor.extract(y)
            save_to_npy(a, Path("".join([
                str(self.out_path / f"class-{digit}" / stem), ".npy"]))
                        )
