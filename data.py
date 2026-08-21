import os
import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import webdataset as wds
import numpy as np
import pandas as pd
import evaluate_utils
import utils
from dataset.image_folder_dataset import CustomImageFolderDataset
from dataset.five_validation_dataset import FiveValidationDataset
from dataset.webdataset_dataset import (
    FaceWebDatasetTransform,
    build_face_webdataset,
    find_webdataset_shards,
)


class DataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.output_dir = kwargs['output_dir']
        self.data_root = kwargs['data_root']
        self.train_data_path = kwargs['train_data_path']
        self.val_data_path = kwargs['val_data_path']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.train_data_subset = kwargs['train_data_subset']

        self.low_res_augmentation_prob = kwargs['low_res_augmentation_prob']
        self.crop_augmentation_prob = kwargs['crop_augmentation_prob']
        self.photometric_augmentation_prob = kwargs['photometric_augmentation_prob']
        self.swap_color_channel = kwargs['swap_color_channel']
        self.use_mxrecord = kwargs['use_mxrecord']
        self.use_webdataset = kwargs.get('use_webdataset', False)
        self.webdataset_pattern = kwargs.get(
            'webdataset_pattern', 'glint360k_train-*.tar')
        self.train_num_samples = kwargs.get('train_num_samples', -1)
        self.webdataset_shuffle_buffer = kwargs.get(
            'webdataset_shuffle_buffer', 20000)
        self.seed = kwargs.get('seed', 42)

        if self.use_mxrecord and self.use_webdataset:
            raise ValueError('--use_mxrecord and --use_webdataset are mutually exclusive')
        if self.use_webdataset and self.train_num_samples <= 0:
            raise ValueError(
                '--train_num_samples must be positive when using WebDataset')

        concat_mem_file_name = os.path.join(self.data_root, self.val_data_path, 'concat_validation_memfile')
        self.concat_mem_file_name = concat_mem_file_name


    def prepare_data(self):
        # call this once to convert val_data to memfile for saving memory
        if not os.path.isdir(os.path.join(self.data_root, self.val_data_path, 'agedb_30', 'memfile')):
            print('making validation data memfile')
            evaluate_utils.get_val_data(os.path.join(self.data_root, self.val_data_path))

        if not os.path.isfile(self.concat_mem_file_name):
            # create a concat memfile
            concat = []
            for key in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']:
                np_array, issame = evaluate_utils.get_val_pair(path=os.path.join(self.data_root, self.val_data_path),
                                                               name=key,
                                                               use_memfile=False)
                concat.append(np_array)
            concat = np.concatenate(concat)
            evaluate_utils.make_memmap(self.concat_mem_file_name, concat)


    def setup(self, stage=None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            print('creating train dataset')
            self.train_dataset = train_dataset(self.data_root,
                                               self.train_data_path,
                                               self.low_res_augmentation_prob,
                                               self.crop_augmentation_prob,
                                               self.photometric_augmentation_prob,
                                               self.swap_color_channel,
                                               self.use_mxrecord,
                                               self.output_dir,
                                               use_webdataset=self.use_webdataset,
                                               webdataset_pattern=self.webdataset_pattern,
                                               webdataset_shuffle_buffer=self.webdataset_shuffle_buffer,
                                               seed=self.seed,
                                               )

            if 'faces_emore' in self.train_data_path and self.train_data_subset:
                # subset ms1mv2 dataset for reproducing the same setup in AdaFace ablation experiments.
                with open('assets/ms1mv2_train_subset_index.txt', 'r') as f:
                    subset_index = [int(i) for i in f.read().split(',')]
                    self.subset_ms1mv2_dataset(subset_index)

            print('creating val dataset')
            self.val_dataset = val_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.test_dataset = test_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

    def train_dataloader(self):
        if self.use_webdataset:
            world_size = utils.get_world_size()
            global_batch_size = self.batch_size * world_size
            batches_per_rank = int(math.ceil(
                self.train_num_samples / float(global_batch_size)))
            dataset = self.train_dataset.batched(
                self.batch_size, partial=False)
            loader = wds.WebLoader(
                dataset,
                batch_size=None,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
            loader = loader.with_epoch(batches_per_rank).with_length(
                batches_per_rank)
            print('WebDataset batches per rank:', batches_per_rank)
            return loader

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def subset_ms1mv2_dataset(self, subset_index):
        # remove too few example identites
        self.train_dataset.samples = [self.train_dataset.samples[idx] for idx in subset_index]
        self.train_dataset.targets = [self.train_dataset.targets[idx] for idx in subset_index]
        value_counts = pd.Series(self.train_dataset.targets).value_counts()
        to_erase_label = value_counts[value_counts<5].index
        e_idx = [i in to_erase_label for i in self.train_dataset.targets]
        self.train_dataset.samples = [i for i, erase in zip(self.train_dataset.samples, e_idx) if not erase]
        self.train_dataset.targets = [i for i, erase in zip(self.train_dataset.targets, e_idx) if not erase]

        # label adjust
        max_label = np.max(self.train_dataset.targets)
        adjuster = {}
        new = 0
        for orig in range(max_label+1):
            if orig in to_erase_label:
                continue
            adjuster[orig] = new
            new += 1

        # readjust class_to_idx
        self.train_dataset.targets = [adjuster[orig] for orig in self.train_dataset.targets]
        self.train_dataset.samples = [(sample[0], adjuster[sample[1]]) for sample in self.train_dataset.samples]
        new_class_to_idx = {}
        for label_str, label_int in self.train_dataset.class_to_idx.items():
            if label_int in to_erase_label:
                continue
            else:
                new_class_to_idx[label_str] = adjuster[label_int]
        self.train_dataset.class_to_idx = new_class_to_idx


def train_dataset(data_root, train_data_path,
                  low_res_augmentation_prob,
                  crop_augmentation_prob,
                  photometric_augmentation_prob,
                  swap_color_channel,
                  use_mxrecord,
                  output_dir,
                  use_webdataset=False,
                  webdataset_pattern='glint360k_train-*.tar',
                  webdataset_shuffle_buffer=20000,
                  seed=42):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if use_webdataset:
        shards = find_webdataset_shards(
            data_root, train_data_path, webdataset_pattern)
        print('WebDataset shard count:', len(shards))
        image_transform = FaceWebDatasetTransform(
            transform=train_transform,
            low_res_augmentation_prob=low_res_augmentation_prob,
            crop_augmentation_prob=crop_augmentation_prob,
            photometric_augmentation_prob=photometric_augmentation_prob,
            swap_color_channel=swap_color_channel,
            output_dir=output_dir,
        )
        train_dataset = build_face_webdataset(
            shards=shards,
            image_transform=image_transform,
            shuffle_buffer=webdataset_shuffle_buffer,
            seed=seed,
        )
    elif use_mxrecord:
        # MXNet is only needed for legacy RecordIO datasets. Import it lazily
        # so Glint360K WebDataset training does not depend on MXNet 1.x.
        from dataset.record_dataset import AugmentRecordDataset

        train_dir = os.path.join(data_root, train_data_path)
        train_dataset = AugmentRecordDataset(root_dir=train_dir,
                                             transform=train_transform,
                                             low_res_augmentation_prob=low_res_augmentation_prob,
                                             crop_augmentation_prob=crop_augmentation_prob,
                                             photometric_augmentation_prob=photometric_augmentation_prob,
                                             swap_color_channel=swap_color_channel,
                                             output_dir=output_dir)
    else:
        train_dir = os.path.join(data_root, train_data_path, 'imgs')
        train_dataset = CustomImageFolderDataset(root=train_dir,
                                                 transform=train_transform,
                                                 low_res_augmentation_prob=low_res_augmentation_prob,
                                                 crop_augmentation_prob=crop_augmentation_prob,
                                                 photometric_augmentation_prob=photometric_augmentation_prob,
                                                 swap_color_channel=swap_color_channel,
                                                 output_dir=output_dir
                                                 )

    return train_dataset


def val_dataset(data_root, val_data_path, concat_mem_file_name):
    val_data = evaluate_utils.get_val_data(os.path.join(data_root, val_data_path))
    # theses datasets are already normalized with mean 0.5, std 0.5
    age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
    val_data_dict = {
        'agedb_30': (age_30, age_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)

    return val_dataset


def test_dataset(data_root, val_data_path, concat_mem_file_name):
    val_data = evaluate_utils.get_val_data(os.path.join(data_root, val_data_path))
    # theses datasets are already normalized with mean 0.5, std 0.5
    age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
    val_data_dict = {
        'agedb_30': (age_30, age_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)
    return val_dataset


