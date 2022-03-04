import itertools
from pathlib import Path
from typing import Iterator, List
from functools import reduce

import numpy as np
import tensorflow as tf


class MNISTAudioDataset:

    __slots__ = ("file_list", "shape", "initializer", "itx", "ity")

    def __init__(self, fl: Iterator[Path]):
        self.file_list = fl
        self.initializer = tf.zeros([0, 128, 11])
        self.itx, self.ity = itertools.tee(self.file_list)

    def populate(self) -> tf.data.Dataset:
        features = tf.data.Dataset.\
            from_tensor_slices(reduce(self._create_features, self.itx, self.initializer))\
            .map(lambda x: tf.expand_dims(x, axis=-1))
        labels = tf.data.Dataset.from_tensor_slices(tf.constant(self._create_labels(self.ity)))
        return tf.data.Dataset.zip((features, labels))

    @staticmethod
    def _create_features(t: tf.Tensor, f: Path):
        a = np.load(f)
        tx = tf.expand_dims(tf.convert_to_tensor(a, dtype=tf.float32), axis=0)
        t = tf.concat([t, tx], axis=0)

        return t

    @staticmethod
    def _create_labels(f: Iterator[Path]) -> List[int]:
        return list(map(lambda x: int(x.parent.stem.split('-').pop()), f))
