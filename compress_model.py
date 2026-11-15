#!/usr/bin/env python3
"""Export and evaluate compact versions of the trained CRNN checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from crnn_model import CRNNFixedCaptcha
from captcha_model import CaptchaDataset, evaluate

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")

def export_fp16(args):
    ckpt = load_checkpoint(args.source)
    model = CRNNFixedCaptcha(); model.load_state_dict(ckpt["state_dict"]); model.half().eval()
    torch.save({**ckpt, "state_dict": model.state_dict(), "precision": "fp16"}, args.target)
    print(json.dumps({"target": args.target, "precision": "fp16"}))

def export_int8(args):
    ckpt = load_checkpoint(args.source)
    model = CRNNFixedCaptcha()
    model.load_state_dict(ckpt["state_dict"])
    model = torch.ao.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    model.eval()
    torch.save({**ckpt, "state_dict": model.state_dict(), "precision": "int8-dynamic"}, args.target)
    print(json.dumps({"target": args.target, "precision": "int8-dynamic"}))

def test(args):
    ckpt = load_checkpoint(args.model)
    model = CRNNFixedCaptcha()
    if ckpt.get("precision") == "int8-dynamic":
        model = torch.ao.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    metrics = evaluate(model, DataLoader(CaptchaDataset(Path(args.data)), args.batch), torch.device("cpu"))
    print(json.dumps({"model": args.model, "precision": ckpt.get("precision", "fp32"), **metrics}))

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fp16", export_fp16), ("int8", export_int8)):
        q = sub.add_parser(name); q.add_argument("source"); q.add_argument("target"); q.set_defaults(fn=fn)
    q = sub.add_parser("test"); q.add_argument("model"); q.add_argument("data"); q.add_argument("--batch", type=int, default=256); q.set_defaults(fn=test)
    a = p.parse_args(); a.fn(a)
if __name__ == "__main__": main()
