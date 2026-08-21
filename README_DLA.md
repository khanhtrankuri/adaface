# AdaFace on TensorRT DLA (FP16/INT8)

The published AdaFace IR checkpoints use pre-activation residual blocks:

```text
BatchNorm -> Conv -> BatchNorm -> PReLU -> Conv -> BatchNorm
```

The DLA variants in this repository use foldable post-normalization blocks:

```text
Conv -> BatchNorm -> PReLU -> Conv -> BatchNorm
```

Select one by adding `_dla` to the architecture used for training:

```bash
python main.py --arch ir_50_dla <other training arguments>
```

Available names are `ir_18_dla`, `ir_34_dla`, `ir_50_dla`, `ir_101_dla`,
and `ir_se_50_dla`.

## Important checkpoint note

The published `ir_50`/`ir_101` weights are pre-activation weights. They are not
functionally equivalent to the DLA graph. The matching Conv/BN/PReLU tensors can
be used as an initialization, but the model must then be fine-tuned and calibrated
again:

```python
checkpoint = torch.load('pretrained/adaface_ir50_ms1mv2.ckpt')
legacy_state = {
    key[6:]: value
    for key, value in checkpoint['state_dict'].items()
    if key.startswith('model.')
}

backbone = net.build_model('ir_50_dla')
dla_state = net.convert_legacy_state_dict_for_dla(legacy_state)
backbone.load_state_dict(dla_state, strict=True)
# Fine-tune before export. Do not evaluate this initialization as a final model.
```

## Export raw embeddings

Do not put L2 normalization or cosine similarity in the INT8 DLA subgraph. Export
the raw 512-dimensional embedding instead:

```python
import torch
import net

backbone = net.build_model('ir_50_dla')
# Load a checkpoint trained with --arch ir_50_dla here.
backbone.eval()
dla_model = net.DLAEmbeddingModel(backbone).eval()

example = torch.randn(1, 3, 112, 112)
torch.onnx.export(
    dla_model,
    example,
    'adaface_ir50_dla_raw.onnx',
    input_names=['image'],
    output_names=['raw_embedding'],
    opset_version=13,
)
```

After inference, dequantize the raw embedding and compute normalization/cosine
with FP32 accumulation on the CPU or GPU:

```python
import torch.nn.functional as F

embedding_a = F.normalize(raw_embedding_a.float(), p=2, dim=1, eps=1e-12)
embedding_b = F.normalize(raw_embedding_b.float(), p=2, dim=1, eps=1e-12)
cosine = (embedding_a * embedding_b).sum(dim=1)
```

For INT8, calibrate with representative, aligned face images after applying the
same BGR conversion and `[-1, 1]` normalization as production. Residual-add input
scales must also be consistent when configuring TensorRT/DLA.
