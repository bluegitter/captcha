#!/usr/bin/env python3
"""CNN + BiLSTM fixed-length CAPTCHA recognizer and fair comparison runner."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from captcha_model import CHARS, LENGTH, CaptchaDataset, evaluate

class CRNNFixedCaptcha(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(128 * 9, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.2), nn.Linear(256, len(CHARS))) for _ in range(LENGTH)])

    def forward(self, x):
        feature = self.cnn(x)
        b, c, h, w = feature.shape
        sequence = feature.permute(0, 3, 1, 2).reshape(b, w, c * h)
        sequence, _ = self.rnn(sequence)
        chunks = torch.chunk(sequence, LENGTH, dim=1)
        return torch.stack([head(chunk.mean(dim=1)) for head, chunk in zip(self.heads, chunks)], dim=1)

def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    ds = CaptchaDataset(Path(args.data)); nv = max(1, int(len(ds) * args.val_ratio))
    train_ds, val_ds = random_split(ds, [len(ds) - nv, nv], generator=torch.Generator().manual_seed(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CRNNFixedCaptcha().to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); ce = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, args.batch, shuffle=True); val_loader = DataLoader(val_ds, args.batch)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for x, y in train_loader:
            out = model(x.to(device)); y = y.to(device)
            loss = sum(ce(out[:, i], y[:, i]) for i in range(LENGTH)) / LENGTH
            opt.zero_grad(); loss.backward(); opt.step(); running += loss.item() * len(x)
        metrics = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "train_loss": running / len(train_ds), **metrics}))
        if metrics["captcha_accuracy"] >= best:
            best = metrics["captcha_accuracy"]
            Path(args.model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "chars": CHARS, "length": LENGTH, "size": [120, 38], "val": metrics}, args.model)
    print(json.dumps({"model": args.model, "device": str(device), "best_val_captcha_accuracy": best}))

def test(args):
    ckpt = torch.load(args.model, map_location="cpu"); model = CRNNFixedCaptcha(); model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(json.dumps({"model": args.model, "data": str(args.data), **evaluate(model.to(device), DataLoader(CaptchaDataset(Path(args.data)), args.batch), device)}))

def predict(args):
    from PIL import Image
    ckpt = torch.load(args.model, map_location="cpu")
    model = CRNNFixedCaptcha(); model.load_state_dict(ckpt["state_dict"]); model.eval()
    image = Image.open(args.image).convert("L").resize((120, 38))
    x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).float().view(1, 1, 38, 120) / 255.0
    with torch.no_grad():
        probabilities = model(x).softmax(-1)[0]
    indices = probabilities.argmax(-1)
    print(json.dumps({"image": args.image, "prediction": "".join(CHARS[i] for i in indices.tolist()), "confidence": [round(float(probabilities[i, indices[i]]), 4) for i in range(LENGTH)]}))

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, fn in (("train", train), ("test", test)):
        q = sub.add_parser(cmd); q.add_argument("data", type=Path); q.add_argument("--model", required=True); q.add_argument("--batch", type=int, default=128); q.add_argument("--epochs", type=int, default=20); q.add_argument("--lr", type=float, default=5e-4); q.add_argument("--val-ratio", type=float, default=.2); q.add_argument("--seed", type=int, default=20261114); q.set_defaults(fn=fn)
    q = sub.add_parser("predict"); q.add_argument("model"); q.add_argument("image"); q.set_defaults(fn=predict)
    a = p.parse_args(); a.fn(a)
if __name__ == "__main__": main()
