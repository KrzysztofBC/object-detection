import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def samples_dir():
    return f"{str(Path(__file__).absolute().parent.parent)}\\samples\\"


@pytest.fixture()
def sample_image(samples_dir):
    return samples_dir + 'vid_5_26720.jpg'


@pytest.fixture()
def coco_labels(samples_dir):
    return samples_dir + 'labels.csv'
