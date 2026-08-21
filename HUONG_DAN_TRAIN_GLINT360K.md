# Hướng dẫn train AdaFace với Glint360K trên NVIDIA H200

Tài liệu này hướng dẫn chạy kiến trúc AdaFace `ir_101_dla` với dữ liệu
Glint360K đã tải từ Hugging Face. Pipeline đọc trực tiếp các TAR shard bằng
WebDataset, hỗ trợ BF16 và DDP trên một hoặc nhiều GPU H200 trong cùng máy.

## 1. Tổng quan cấu hình

- Dataset: [yayoimizuha/Glint360k](https://huggingface.co/datasets/yayoimizuha/Glint360k)
- Số ảnh dùng cho một epoch: `17,091,657`
- Số identity/class: `360,232`
- Backbone khuyến nghị: `ir_101_dla`
- Head: AdaFace
- Precision trên H200: BF16
- Input: ảnh khuôn mặt đã căn chỉnh `112x112`

Kiến trúc `*_dla` dùng các cặp `Conv -> BatchNorm` có thể fold khi triển khai
TensorRT/DLA. Trong lúc train, phép chuẩn hóa embedding và cosine được tính với
FP32 để giảm sai số số học.

## 2. Cấu trúc thư mục

Ví dụ dữ liệu được đặt dưới `/datasets`:

```text
/datasets/
├── Glint360k_local/
│   ├── README.md
│   ├── glint360k_train-000.tar
│   ├── glint360k_train-000.idx
│   ├── glint360k_train-001.tar
│   ├── glint360k_train-001.idx
│   ├── ...
│   ├── glint360k_train-138.tar
│   └── glint360k_train-138.idx
│
└── face_validation/
    ├── agedb_30.bin
    ├── cfp_fp.bin
    ├── lfw.bin
    ├── cplfw.bin
    └── calfw.bin
```

Các file `.idx` có thể giữ nguyên nhưng pipeline train không sử dụng chúng.
Glint360K chỉ chứa tập train, vì vậy cần chuẩn bị riêng năm tập validation ở
trên.

> Không chạy `convert.py` với Glint360K và không giải nén TAR shard. Cũng không
> cần gọi `datasets.load_dataset()` trong code train.

## 3. Tải dataset

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yayoimizuha/Glint360k",
    repo_type="dataset",
    local_dir="/datasets/Glint360k_local",
)
```

Kiểm tra đủ 139 TAR shard:

```bash
find /datasets/Glint360k_local -maxdepth 1 \
  -name 'glint360k_train-*.tar' | wc -l
```

Kết quả mong đợi:

```text
139
```

Kiểm tra nhanh một shard không bị lỗi:

```bash
tar -tf /datasets/Glint360k_local/glint360k_train-000.tar | head
```

## 4. Chuẩn bị môi trường H200

Khuyến nghị Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu Docker/cluster đã cung cấp sẵn PyTorch tương thích driver CUDA, nên giữ bản
PyTorch của system image và chỉ cài các dependency còn thiếu. Train TAR không
cần cài MXNet; MXNet chỉ dành cho luồng RecordIO cũ với `--use_mxrecord`.

Kiểm tra GPU và BF16:

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0)); print('bf16:', torch.cuda.is_bf16_supported())"
```

Giá trị `cuda` và `bf16` phải là `True`.

## 5. Smoke test trước khi train

Luôn chạy `fast_dev_run` trước. Lệnh này thực hiện một batch train và một batch
validation để kiểm tra TAR decoder, label, forward, loss, backward và GPU.

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
  --custom_num_class 360232 \
  --head adaface \
  --m 0.4 \
  --h 0.333 \
  --gpus 1 \
  --precision bf16 \
  --batch_size 64 \
  --num_workers 8 \
  --epochs 1 \
  --lr 0.1 \
  --lr_milestones 1 \
  --prefix smoke_glint360k_ir101_dla \
  --fast_dev_run
```

Smoke test thành công khi log có các thông tin tương tự:

```text
WebDataset shard count: 139
Using bfloat16 Automatic Mixed Precision (AMP)
fast_dev_run finished; skipping checkpoint-based test
```

`fast_dev_run` không tạo checkpoint dùng cho train chính thức.

## 6. Train đầy đủ với một H200

Nên bắt đầu với batch size `256`. Nếu H200 còn nhiều bộ nhớ, tăng lên `512`.

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
  --custom_num_class 360232 \
  --head adaface \
  --m 0.4 \
  --h 0.333 \
  --s 64.0 \
  --t_alpha 0.01 \
  --gpus 1 \
  --precision bf16 \
  --batch_size 256 \
  --num_workers 16 \
  --epochs 20 \
  --lr 0.1 \
  --lr_milestones 8,14,18 \
  --lr_gamma 0.1 \
  --low_res_augmentation_prob 0.2 \
  --crop_augmentation_prob 0.2 \
  --photometric_augmentation_prob 0.2 \
  --prefix glint360k_ir101_dla_h200
```

Nếu batch `256` bị thiếu bộ nhớ, dùng batch `128` và gradient accumulation:

```text
--batch_size 128 --accumulate_grad_batches 2
```

Khi đó effective batch size vẫn là `256`.

## 7. Train DDP với 8 H200 trên cùng một máy

`--batch_size` là tổng batch size của toàn bộ GPU. Với `2048` và 8 GPU, mỗi
process nhận batch `256`. Chạy `python main.py` một lần; Lightning tự khởi tạo
các process DDP.

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
  --custom_num_class 360232 \
  --head adaface \
  --m 0.4 \
  --h 0.333 \
  --gpus 8 \
  --distributed_backend ddp \
  --precision bf16 \
  --batch_size 2048 \
  --num_workers 8 \
  --epochs 20 \
  --lr 0.4 \
  --lr_milestones 8,14,18 \
  --lr_gamma 0.1 \
  --prefix glint360k_ir101_dla_8xh200
```

Đây là cấu hình DDP single-node. Mỗi rank và mỗi DataLoader worker nhận một
phần stream riêng. Head hiện tại là full classifier 360,232 class và được
replicate trên từng GPU; PartialFC chưa được bật.

## 8. Output, log và checkpoint

Kết quả được lưu theo `--prefix`:

```text
experiments/
└── glint360k_ir101_dla_h200_MM-DD_N/
    ├── last.ckpt
    ├── epoch=...ckpt
    ├── result/
    │   └── version_0/
    │       └── metrics.csv
    └── training_samples/
        └── sample.jpg
```

- `last.ckpt`: trạng thái mới nhất để resume.
- Checkpoint tốt nhất được chọn theo `val_acc`.
- `metrics.csv`: learning rate, train loss và validation accuracy.
- `sample.jpg`: ảnh mẫu sau bước decode/augmentation để kiểm tra màu và input.

## 9. Resume khi job bị dừng

Dùng lại toàn bộ cấu hình cũ và thêm `--resume_from_checkpoint`:

```bash
python main.py \
  --data_root /datasets \
  --train_data_path Glint360k_local \
  --val_data_path face_validation \
  --use_webdataset \
  --webdataset_pattern 'glint360k_train-*.tar' \
  --train_num_samples 17091657 \
  --arch ir_101_dla \
  --custom_num_class 360232 \
  --head adaface \
  --m 0.4 \
  --h 0.333 \
  --gpus 1 \
  --precision bf16 \
  --batch_size 256 \
  --num_workers 16 \
  --epochs 20 \
  --lr 0.1 \
  --lr_milestones 8,14,18 \
  --resume_from_checkpoint experiments/glint360k_ir101_dla_h200_MM-DD_N/last.ckpt
```

Không thay đổi `arch`, số class hoặc head khi resume.

## 10. Export ONNX sau khi train

Export raw embedding `512-D` cho TensorRT/DLA:

```bash
python extract_onnx.py \
  --checkpoint experiments/glint360k_ir101_dla_h200_MM-DD_N/last.ckpt \
  --arch ir_101_dla \
  --output models/adaface_ir101_glint360k_dla.onnx \
  --batch-size 1 \
  --opset 13 \
  --fp16
```

ONNX chỉ trả về `raw_embedding`. Sau inference, dequantize embedding và tính L2
normalization/cosine bằng FP32 ở CPU hoặc GPU, không đặt cosine INT8 trong DLA.

## 11. Lỗi thường gặp

### `no WebDataset TAR shards matched`

Kiểm tra lại ba tham số:

```text
--data_root /datasets
--train_data_path Glint360k_local
--webdataset_pattern 'glint360k_train-*.tar'
```

Đường dẫn cuối cùng phải match:

```text
/datasets/Glint360k_local/glint360k_train-*.tar
```

### `--train_num_samples must be positive`

Luôn truyền:

```text
--train_num_samples 17091657
```

Tham số này xác định số batch trong một epoch của stream WebDataset.

### Thiếu `agedb_30.bin`, `cfp_fp.bin`, ...

Glint360K không chứa validation. Hãy copy bộ validation vào một thư mục riêng
và sửa `--val_data_path`. Lần chạy đầu tiên có thể mất thời gian để sinh
validation memfile.

### CUDA out of memory

Giảm `--batch_size` theo thứ tự `256 -> 128 -> 64`. Có thể tăng
`--accumulate_grad_batches` để giữ effective batch size.

### GPU chờ dữ liệu

- Đặt TAR shard trên local NVMe thay vì network filesystem chậm.
- Thử `--num_workers 16`, sau đó tăng/giảm theo số CPU core và tốc độ storage.
- Giữ `--webdataset_shuffle_buffer 20000`; nếu thiếu RAM host, giảm còn `5000`.

### DDP chậm khi dùng nhiều H200

Full classifier có kernel và gradient lớn, nên all-reduce có thể là nút thắt.
Kiểm tra NVLink/NVSwitch và NCCL trước khi tăng số GPU. PartialFC là một thay
đổi kiến trúc riêng, chưa nằm trong pipeline hiện tại.

## 12. Checklist trước job dài

- [ ] Có đủ 139 TAR shard.
- [ ] Không giải nén TAR và không chạy `convert.py`.
- [ ] Có đủ năm validation dataset.
- [ ] `torch.cuda.is_bf16_supported()` trả về `True`.
- [ ] Smoke test chạy qua forward, backward và validation.
- [ ] `sample.jpg` có màu và alignment đúng.
- [ ] Storage còn đủ dung lượng cho checkpoint và validation memfile.
- [ ] Đã ghi lại batch size, learning rate, seed và số GPU.

