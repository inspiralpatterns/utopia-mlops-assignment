from abc import abstractmethod, ABC

import librosa.feature
import numpy as np


class Extractor(ABC):
    @abstractmethod
    def extract(self, data):
        raise NotImplementedError


class MFCCExtractor(Extractor):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def extract(self, data: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(data, **self.__dict__)


class MelSpectrogramExtractor(Extractor):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def extract(self, data: np.ndarray) -> np.ndarray:
        return librosa.feature.melspectrogram(data, **self.__dict__)
