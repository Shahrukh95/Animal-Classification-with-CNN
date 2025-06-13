import os
import sys
import pathlib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils import data_processor


def create_images(directory):
    gray = (np.random.rand(128, 128) * 255).astype('uint8')
    Image.fromarray(gray, mode='L').save(os.path.join(directory, 'gray.bmp'))

    rgb = (np.random.rand(128, 128, 3) * 255).astype('uint8')
    Image.fromarray(rgb, mode='RGB').save(os.path.join(directory, 'rgb.bmp'))

    rgba = (np.random.rand(128, 128, 4) * 255).astype('uint8')
    Image.fromarray(rgba, mode='RGBA').save(os.path.join(directory, 'rgba.png'))


def test_data_processor(tmp_path):
    create_images(tmp_path)
    X, y = data_processor(str(tmp_path) + os.sep, 0)

    n_images = 3
    assert X.shape == (n_images, 49152)
    for row in X.to_numpy():
        assert row.reshape(128, 128, 3).shape == (128, 128, 3)
    assert isinstance(y, pd.Series)
    assert len(y) == n_images
