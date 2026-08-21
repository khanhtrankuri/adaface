"""WebDataset input pipeline for sharded face-recognition datasets."""

import glob
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import webdataset as wds

from .augmenter import Augmenter


class FaceWebDatasetTransform:
    """Apply the same channel order, augmentation, and normalization as MXRecord."""

    def __init__(self, transform, low_res_augmentation_prob,
                 crop_augmentation_prob, photometric_augmentation_prob,
                 swap_color_channel, output_dir):
        self.transform = transform
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir
        self.augmenter = Augmenter(
            crop_augmentation_prob,
            photometric_augmentation_prob,
            low_res_augmentation_prob,
        )

    def __call__(self, image):
        image = image.convert('RGB')

        # AdaFace's MXRecord pipeline trains on BGR tensors by default. The
        # WebDataset JPEGs decode as RGB, so mirror BaseMXDataset exactly.
        image_array = np.asarray(image)
        image = Image.fromarray(image_array[:, :, ::-1].copy())
        if self.swap_color_channel:
            image = Image.fromarray(np.asarray(image)[:, :, ::-1].copy())

        image = self.augmenter.augment(image)
        self._save_training_sample(image)
        return self.transform(image)

    def _save_training_sample(self, image):
        sample_path = os.path.join(
            self.output_dir, 'training_samples', 'sample.jpg')
        if os.path.isfile(sample_path):
            return
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        cv2.imwrite(sample_path, np.asarray(image))


def decode_class_label(label):
    if isinstance(label, bytes):
        label = label.decode('utf-8')
    if isinstance(label, np.generic):
        label = label.item()
    # WebDataset converts Python integers to a platform-native NumPy dtype
    # (int32 on Windows), while NumPy scalar values remain Python lists.
    # Returning a scalar tensor makes its batch collation deterministic.
    return torch.tensor(int(label), dtype=torch.long)


def find_webdataset_shards(data_root, train_data_path, shard_pattern):
    train_root = os.path.join(data_root, train_data_path)
    pattern = shard_pattern
    if not os.path.isabs(pattern):
        pattern = os.path.join(train_root, pattern)

    shards = sorted(glob.glob(pattern))
    shards = [path for path in shards if path.lower().endswith('.tar')]
    if not shards:
        raise FileNotFoundError(
            'no WebDataset TAR shards matched: {}'.format(pattern))
    return shards


def build_face_webdataset(shards, image_transform, shuffle_buffer, seed):
    """Build an infinite, shuffled stream; WebLoader defines epoch length."""
    # urlparse treats the drive letter in ``C:\\...`` as a URL scheme.
    # WebDataset 1.0's cache opener also expects ``file:C:/...`` (without
    # three slashes) on Windows. POSIX paths stay unchanged on the H200 host.
    if os.name == 'nt':
        shards = ['file:' + Path(path).resolve().as_posix()
                  for path in shards]

    dataset = wds.WebDataset(
        shards,
        resampled=True,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        handler=wds.warn_and_continue,
        seed=seed,
    )
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(
            shuffle_buffer, initial=min(1000, shuffle_buffer))
    return (
        dataset
        .decode('pil', handler=wds.warn_and_continue)
        .to_tuple('jpg;jpeg;png', 'cls', handler=wds.warn_and_continue)
        .map_tuple(
            image_transform,
            decode_class_label,
            handler=wds.warn_and_continue,
        )
    )
