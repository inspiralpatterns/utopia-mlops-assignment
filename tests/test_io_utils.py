from pathlib import Path

import pytest
from yaml.scanner import ScannerError

from io_utils import set_dataset_dir, list_subdir, list_files, load_config


def test_set_dataset_dir():
    d = "test/dir"
    assert set_dataset_dir(d) == Path("".join([f"{(Path().resolve())}/", "test/dir"]))


@pytest.mark.xfail(raises=TypeError)
def test_set_dataset_dir_wrong_in():
    set_dataset_dir(42)


def test_list_subdir():
    p = Path().resolve() / "tests" / "test_dir"
    s = list_subdir(p)
    assert next(s) == p / "test_subdir"


def test_list_files():
    p = Path().resolve() / "tests" / "test_dir" / "test_subdir"
    fs = list_files(p)
    assert next(fs) == p / "file"


@pytest.mark.xfail(raises=ScannerError)
def test_load_config_no_yaml():
    load_config(Path().resolve() / "fake_config.yaml")

