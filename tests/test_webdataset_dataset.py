import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from dataset.webdataset_dataset import (
    FaceWebDatasetTransform,
    build_face_webdataset,
    decode_class_label,
    find_webdataset_shards,
)


def add_tar_member(archive, name, payload):
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


class FaceWebDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.shard = self.root / 'glint360k_train-000.tar'

        image = Image.new('RGB', (112, 112), color=(10, 20, 30))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        with tarfile.open(self.shard, mode='w') as archive:
            add_tar_member(archive, 'sample.png', image_bytes.getvalue())
            add_tar_member(archive, 'sample.cls', b'7')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_shards_ignores_idx_files(self):
        (self.root / 'glint360k_train-000.idx').write_bytes(b'index')
        shards = find_webdataset_shards(
            str(self.root), '', 'glint360k_train-*')
        self.assertEqual(shards, [str(self.shard)])

    def test_decode_class_label(self):
        self.assertEqual(decode_class_label(b'360231').item(), 360231)
        self.assertEqual(decode_class_label(np.int64(12)).item(), 12)

    def test_pipeline_decodes_label_and_matches_bgr_preprocessing(self):
        image_transform = FaceWebDatasetTransform(
            transform=transforms.ToTensor(),
            low_res_augmentation_prob=0.0,
            crop_augmentation_prob=0.0,
            photometric_augmentation_prob=0.0,
            swap_color_channel=False,
            output_dir=str(self.root / 'output'),
        )
        dataset = build_face_webdataset(
            shards=[str(self.shard)],
            image_transform=image_transform,
            shuffle_buffer=0,
            seed=42,
        )

        iterator = iter(dataset.batched(2, partial=False))
        try:
            images, labels = next(iterator)
        finally:
            iterator.close()

        self.assertEqual(labels.tolist(), [7, 7])
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(tuple(images.shape), (2, 3, 112, 112))
        # JPEG/PNG decodes RGB (10, 20, 30); AdaFace default expects BGR.
        np.testing.assert_allclose(
            images[0, :, 0, 0].numpy(),
            np.array([30, 20, 10], dtype=np.float32) / 255.0,
            rtol=0,
            atol=1e-6,
        )


if __name__ == '__main__':
    unittest.main()
