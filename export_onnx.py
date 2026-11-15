#!/usr/bin/env python3
"""Export the fixed-length CRNN CAPTCHA model to FP32 and browser-friendly INT8 ONNX."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from crnn_model import CRNNFixedCaptcha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='artifacts/captcha_crnn_a100.pt')
    p.add_argument('--fp32', default='artifacts/captcha_crnn_a100.onnx')
    p.add_argument('--int8', default='artifacts/captcha_crnn_a100_int8.onnx')
    args = p.parse_args()

    checkpoint = torch.load(args.model, map_location='cpu')
    model = CRNNFixedCaptcha()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    dummy = torch.zeros(1, 1, 38, 120)
    Path(args.fp32).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.fp32, opset_version=17,
        input_names=['image'], output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        do_constant_folding=True,
    )
    Path(args.int8).parent.mkdir(parents=True, exist_ok=True)
    # WASM supports these quantized operators more reliably than ConvInteger.
    quantize_dynamic(
        args.fp32, args.int8, weight_type=QuantType.QInt8,
        per_channel=True, op_types_to_quantize=['MatMul', 'Gemm'],
    )
    print(f'FP32: {args.fp32}')
    print(f'INT8: {args.int8}')


if __name__ == '__main__':
    main()
