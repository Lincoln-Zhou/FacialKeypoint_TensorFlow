import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import datetime

from config import IMAGE_SHAPE, NUM_KEYPOINTS


def read_data_file(img_path: str, kp_path: str):
    """
    Retrieve usable data from provided files. Dataset entries containing NaN values would be dropped.

    :param img_path: Path to .npz file containing face images
    :param kp_path: Path to .csv file containing facial keypoints
    :return: Processed images & keypoints in numpy array format
    """

    images = np.load(img_path)['face_images']
    keypoints = pd.read_csv(kp_path)

    nan_mask = keypoints.isnull().any(axis=1)

    images = images[:, :, ~nan_mask]    # Drop the images whose keypoints are unusable as well
    images = np.expand_dims(images, axis=0)
    images = images.swapaxes(0, -1)     # Reshape the sequential image data into (N_IMG, 96, 96, 1)

    keypoints = keypoints.dropna().values

    # keypoints = np.expand_dims(np.expand_dims(keypoints, 1), 1)

    return images, keypoints


def visualize_keypoint(img: np.ndarray, kps: np.ndarray):
    try:
        img = img.reshape(IMAGE_SHAPE)
        kps = kps.reshape((-1, 2))
    except ValueError:
        raise RuntimeError('Incorrect input shape')

    plt.imshow(img[:, :, 0], cmap='gray', vmin=0, vmax=255)
    plt.scatter(kps[:, 0].flatten(), kps[:, 1].flatten())
    plt.axis('off')

    plt.show()


def sample_visualization(n: int = 10):
    """
    Randomly choose a few images and display the corresponding keypoints
    """

    images, keypoints = read_data_file("face_images.npz", "facial_keypoints.csv")

    data_size = keypoints.shape[0]

    index_list = random.sample(range(data_size), k=n)

    for index in index_list:
        visualize_keypoint(images[index], keypoints[index])


def plot_training_process(history):
    plt.title('Train History')

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    plt.savefig(f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


if __name__ == '__main__':
    sample_visualization()
