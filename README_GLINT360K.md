# Training AdaFace DLA on Glint360K

The Hugging Face snapshot is already in WebDataset format. Keep the 139
`glint360k_train-*.tar` shards intact; the matching `.idx` files are not needed
for sequential training, and `convert.py` must not be run on this directory.

Glint360K contains 17,091,657 training images and 360,232 identities. This repo
reads the local TAR files directly with `--use_webdataset`.

You do not need to call Hugging Face `load_dataset()` for training. The loader
in `dataset/webdataset_dataset.py` streams JPEG and `cls` members directly from
the local shards, and splits the stream by DDP rank and DataLoader worker.

## H200 environment

Use Python 3.10, then install the pinned CUDA training environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

If the H200 system image already provides PyTorch, keep the site-compatible
PyTorch/CUDA build and install the remaining packages instead of replacing a
working cluster build.

## Validation data

Glint360K's TAR snapshot contains training data only. Set `--val_data_path` to a
separate directory containing `agedb_30.bin`, `cfp_fp.bin`, `lfw.bin`,
`cplfw.bin`, and `calfw.bin`, or their previously generated memfiles.

## Smoke test on one H200

The examples below assume the snapshot was downloaded to
`/datasets/Glint360k_local`. Adjust `/datasets` to the parent of your actual
`local_dir`.

```bash
python main.py \
  --data_root /datasets \
  --train_data_path Glint360k_local \
  --val_data_path face_validation \
  --use_webdataset \
  --webdataset_pattern 'glint360k_train-*.tar' \
  --train_num_samples 17091657 \
  --arch ir_101_dla \
  --prefix smoke_ir101_glint360k_dla \
  --custom_num_class 360232 \
  --gpus 1 \
  --precision bf16 \
  --batch_size 256 \
  --num_workers 16 \
  --epochs 1 \
  --lr 0.1 \
  --lr_milestones 1 \
  --head adaface \
  --m 0.4 \
  --h 0.333 \
  --fast_dev_run
```

## Full training on one H200

```bash
python main.py \
  --data_root /datasets \
  --train_data_path Glint360k_local \
  --val_data_path face_validation \
  --use_webdataset \
  --webdataset_pattern 'glint360k_train-*.tar' \
  --train_num_samples 17091657 \
  --webdataset_shuffle_buffer 20000 \
  --arch ir_101_dla \
  --prefix ir101_glint360k_dla \
  --custom_num_class 360232 \
  --gpus 1 \
  --precision bf16 \
  --batch_size 512 \
  --num_workers 16 \
  --epochs 20 \
  --lr 0.1 \
  --lr_milestones 8,14,18 \
  --head adaface \
  --m 0.4 \
  --h 0.333 \
  --low_res_augmentation_prob 0.2 \
  --crop_augmentation_prob 0.2 \
  --photometric_augmentation_prob 0.2
```

`batch_size` is the global batch size. If memory is insufficient, use 256 and
set `--accumulate_grad_batches 2` to retain an effective batch size of 512.

## Eight H200 GPUs

Lightning launches DDP processes, so invoke `python main.py` once rather than
wrapping it in `torchrun`:

```bash
python main.py \
  --data_root /datasets \
  --train_data_path Glint360k_local \
  --val_data_path face_validation \
  --use_webdataset \
  --webdataset_pattern 'glint360k_train-*.tar' \
  --train_num_samples 17091657 \
  --arch ir_101_dla \
  --prefix ir101_glint360k_dla_8xh200 \
  --custom_num_class 360232 \
  --gpus 8 \
  --distributed_backend ddp \
  --precision bf16 \
  --batch_size 2048 \
  --num_workers 8 \
  --epochs 20 \
  --lr 0.4 \
  --lr_milestones 8,14,18 \
  --head adaface \
  --m 0.4 \
  --h 0.333
```

The current AdaFace head is a full 360,232-class classifier replicated on each
GPU. H200 has enough memory, but classifier gradient synchronization can limit
multi-GPU scaling. PartialFC is a separate optimization and is not enabled by
these commands.
