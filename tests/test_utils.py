import pytest

import extractor
import normalizer
from src.utils.utils import config_to_cls


def test_config_to_cls_extractor():
    d = {"type": "MFCCExtractor"}
    cls = config_to_cls(d)
    assert cls.__class__ == extractor.MFCCExtractor
    assert cls.__dict__ == {}


def test_config_to_cls_normalizer():
    d = {"type": "StandardScalingNormalizer"}
    cls = config_to_cls(d)
    assert cls.__class__ == normalizer.StandardScalingNormalizer
    assert cls.__dict__ == {}


def test_config_to_cls_extractor_w_params():
    d = {"type": "MFCCExtractor", "params": {"n_mfcc": 13, "hop_length": 1024}}
    cls = config_to_cls(d)
    assert cls.__class__ == extractor.MFCCExtractor
    assert cls.__dict__ == {'n_mfcc': 13, 'hop_length': 1024}


@pytest.mark.xfail(raises=KeyError)
def test_config_to_cls_fake_extractor():
    d = {"type": "FakeExtractor"}
    config_to_cls(d)


@pytest.mark.xfail(raises=KeyError)
def test_config_to_cls_fake_normalizer():
    d = {"type": "FakeNormalizer"}
    config_to_cls(d)


@pytest.mark.xfail(raises=AttributeError)
def test_config_to_cls_incorrect_input():
    d = "Wrong input"
    config_to_cls(d)


@pytest.mark.xfail(raises=TypeError)
def test_config_to_cls_no_input():
    config_to_cls()
