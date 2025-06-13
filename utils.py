import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize


def data_processor(path_to_animal: str, label: int):
    """Process images in a directory and return flattened arrays.

    Parameters
    ----------
    path_to_animal : str
        Directory containing images.
    label : int
        Numeric label to assign to each image.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        Flattened image data X and labels y.
    """
    dataframe = []
    label_array = []

    for i in os.listdir(path_to_animal):
        if i[-3:] != 'txt':
            img = mpimg.imread(os.path.join(path_to_animal, i))
            img = resize(img, (128, 128), anti_aliasing=True)
            if img.shape == (128, 128):
                single_channel = img.reshape(128, 128, 1)
                img = np.concatenate([single_channel] * 3, axis=-1)
            elif img.shape == (128, 128, 4):
                img = img[:, :, :3]
            tensor = img.reshape(49152)
            dataframe.append(tensor)
    for _ in range(len(dataframe)):
        label_array.append(label)
    dataframe = pd.DataFrame(dataframe)
    label_array = pd.DataFrame({'label': label_array})
    data = pd.concat([label_array, dataframe], axis=1)
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y
