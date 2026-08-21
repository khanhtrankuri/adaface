"""Export a trained AdaFace DLA backbone to an ONNX raw-embedding model.

L2 normalization and cosine similarity are intentionally excluded from the
exported graph. Dequantize the 512-D output and compute both operations with
FP32 accumulation outside TensorRT DLA.
"""

import argparse
import inspect
import os
from pathlib import Path
import tempfile

import torch

import net


DLA_ARCHITECTURES = (
    'ir_18_dla',
    'ir_34_dla',
    'ir_50_dla',
    'ir_101_dla',
    'ir_se_50_dla',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export an AdaFace DLA checkpoint as a raw-embedding ONNX model.')
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Lightning .ckpt or a raw backbone state_dict file')
    parser.add_argument('--output', required=True, type=Path,
                        help='destination .onnx path')
    parser.add_argument('--arch', default='ir_50_dla', choices=DLA_ARCHITECTURES,
                        help='DLA architecture used to train the checkpoint')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='static export batch size (default: 1)')
    parser.add_argument('--dynamic-batch', action='store_true',
                        help='make only the batch axis dynamic; static is safer for DLA')
    parser.add_argument('--opset', default=13, type=int,
                        help='ONNX opset version (default: 13)')
    parser.add_argument('--fp16', action='store_true',
                        help='store model weights and input/output as FP16')
    parser.add_argument('--skip-check', action='store_true',
                        help='skip onnx.checker after export')
    return parser.parse_args()


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError('checkpoint not found: {}'.format(path))

    load_kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        # Lightning checkpoints contain metadata in addition to tensors.
        load_kwargs['weights_only'] = False
    return torch.load(str(path), **load_kwargs)


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint must contain a state_dict mapping')

    for key in ('state_dict', 'model_state_dict'):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate

    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise TypeError('could not find state_dict or model_state_dict in checkpoint')


def select_backbone_state_dict(state_dict, expected_keys):
    """Select and strip the most likely backbone prefix from a checkpoint."""
    prefixes = ('', 'model.', 'module.model.', 'backbone.',
                'module.backbone.', 'module.')
    candidates = []

    for prefix in prefixes:
        if prefix:
            candidate = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        else:
            candidate = dict(state_dict)
        overlap = len(expected_keys.intersection(candidate))
        candidates.append((overlap, prefix, candidate))

    overlap, prefix, candidate = max(candidates, key=lambda item: item[0])
    if overlap == 0:
        preview = ', '.join(list(state_dict)[:5])
        raise RuntimeError(
            'checkpoint has no AdaFace backbone tensors; first keys: {}'.format(preview))

    # Ignore optimizer/head tensors while still requiring every backbone tensor.
    candidate = {key: value for key, value in candidate.items()
                 if key in expected_keys}
    print('checkpoint prefix: {}'.format(prefix or '<none>'))
    return candidate


def load_backbone(architecture, checkpoint_path):
    backbone = net.build_model(architecture)
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = unwrap_state_dict(checkpoint)
    state_dict = select_backbone_state_dict(
        state_dict, set(backbone.state_dict().keys()))

    try:
        backbone.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'checkpoint does not exactly match {}. A legacy ir_* checkpoint '
            'cannot be exported as *_dla without conversion and fine-tuning.\n{}'
            .format(architecture, error))

    return backbone.eval()


def require_onnx():
    try:
        import onnx
    except ImportError:
        raise RuntimeError(
            'ONNX is required. Install it with: python -m pip install onnx')
    return onnx


def export_onnx(args):
    if args.batch_size < 1:
        raise ValueError('--batch-size must be at least 1')
    if args.opset < 13:
        raise ValueError('--opset must be 13 or newer')
    if args.output.suffix.lower() != '.onnx':
        raise ValueError('--output must end with .onnx')

    onnx = require_onnx()
    backbone = load_backbone(args.arch, args.checkpoint)
    model = net.DLAEmbeddingModel(backbone).eval()
    dtype = torch.float16 if args.fp16 else torch.float32
    model = model.to(dtype=dtype)
    example = torch.zeros(args.batch_size, 3, 112, 112, dtype=dtype)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch'},
            'raw_embedding': {0: 'batch'},
        }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=output_path.stem + '.', suffix='.onnx',
        dir=str(output_path.parent), delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    export_kwargs = {
        'input_names': ['image'],
        'output_names': ['raw_embedding'],
        'opset_version': args.opset,
        'do_constant_folding': True,
        'dynamic_axes': dynamic_axes,
    }
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        # The legacy exporter is more predictable for TensorRT/DLA and opset 13.
        export_kwargs['dynamo'] = False

    try:
        with torch.inference_mode():
            torch.onnx.export(model, example, str(temp_path), **export_kwargs)

        if not args.skip_check:
            onnx_model = onnx.load(str(temp_path))
            onnx.checker.check_model(onnx_model)

        os.replace(str(temp_path), str(output_path))
    finally:
        if temp_path.exists():
            temp_path.unlink()

    size_mb = output_path.stat().st_size / (1024 ** 2)
    precision = 'FP16' if args.fp16 else 'FP32'
    batch = 'dynamic' if args.dynamic_batch else str(args.batch_size)
    print('exported: {}'.format(output_path))
    print('architecture: {} | precision: {} | batch: {}'.format(
        args.arch, precision, batch))
    print('output: raw_embedding [N, 512] | size: {:.1f} MB'.format(size_mb))
    print('Compute L2 normalization and cosine similarity in FP32 outside DLA.')


def main():
    args = parse_args()
    try:
        export_onnx(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit('error: {}'.format(error))


if __name__ == '__main__':
    main()
