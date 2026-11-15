#!/usr/bin/env python3
"""Train and evaluate a fixed-length CAPTCHA recognizer.

Labels are read from filenames (e.g. ``a8c5n.png``). ``generate`` creates a
synthetic corpus that mirrors CodeEndpoint's Kaptcha contract: five
alpha-numeric characters, dark text, and crossing noise lines.
"""
from __future__ import annotations
import argparse, json, random, string
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

# production Kaptcha character pool. Keeping the model
# vocabulary identical to the generator avoids wasting classes on characters
# that can never occur in production.
CHARS = "2345678abcdefgmnpwxy"
LENGTH = 5
W, H = 120, 38
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def render(text: str, rng: random.Random, size=(W, H)) -> Image.Image:
    w, h = size
    im = Image.new("L", (w, h), rng.randint(235, 255))
    d = ImageDraw.Draw(im)
    # Kaptcha-like random baseline, rotation, and dark text.
    x = 5
    for ch in text:
        fs = rng.randint(24, 31)
        layer = Image.new("RGBA", (32, 38), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((3, rng.randint(-5, 2)), ch, font=font(fs), fill=(rng.randint(0, 80),) * 3 + (255,))
        layer = layer.rotate(rng.randint(-18, 18), resample=Image.Resampling.BICUBIC, expand=False)
        im.paste(layer.convert("L"), (x, rng.randint(0, 4)), layer.getchannel("A"))
        x += rng.randint(20, 24)
    # Noise lines may cross glyphs, as in the endpoint's Producer output.
    for _ in range(rng.randint(2, 4)):
        d.line((rng.randint(0, w // 3), rng.randint(0, h), rng.randint(w * 2 // 3, w), rng.randint(0, h)), fill=rng.randint(80, 180), width=rng.randint(1, 2))
    for _ in range(rng.randint(10, 35)):
        x1, y1 = rng.randrange(w), rng.randrange(h)
        d.point((x1, y1), fill=rng.randint(100, 220))
    return im.filter(ImageFilter.GaussianBlur(radius=rng.choice([0, 0, 0.25])))

def generate(root: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        label = ''.join(rng.choice(CHARS) for _ in range(LENGTH))
        # Unique filename avoids collisions while retaining label in stem.
        render(label, rng).save(root / f"{label}_{i:06d}.png")
    print(json.dumps({"output": str(root), "images": n, "seed": seed}))

class CaptchaDataset(Dataset):
    def __init__(self, root: Path):
        self.items = []
        for p in sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")) + list(root.glob("*.jpeg"))):
            label = p.stem.split("_")[0].lower()
            if len(label) == LENGTH and all(c in CHARS for c in label):
                self.items.append((p, torch.tensor([CHARS.index(c) for c in label], dtype=torch.long)))
        if not self.items:
            raise ValueError(f"No labelled PNG files found in {root}; expected <5-char-label>[_id].png")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = Image.open(p).convert("L").resize((W, H))
        x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).float().view(H, W) / 255.0
        return x.unsqueeze(0), y

class CaptchaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((3, 15)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 15, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, LENGTH * len(CHARS)))
    def forward(self, x): return self.head(self.features(x)).view(-1, LENGTH, len(CHARS))

def evaluate(model, loader, device):
    model.eval(); chars = total = exact = 0; loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)); y = y.to(device)
            loss_sum += sum(ce(out[:, i], y[:, i]).item() for i in range(LENGTH)) * len(x)
            pred = out.argmax(-1); chars += (pred == y).sum().item(); total += y.numel(); exact += (pred == y).all(1).sum().item()
    return {"loss": loss_sum / max(1, len(loader.dataset) * LENGTH), "char_accuracy": chars / max(1, total), "captcha_accuracy": exact / max(1, len(loader.dataset))}

def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    ds = CaptchaDataset(Path(args.data)); n = len(ds); nv = max(1, int(n * args.val_ratio)); nt = n - nv
    train_ds, val_ds = random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CaptchaNet().to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); ce = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, args.batch, shuffle=True); val_loader = DataLoader(val_ds, args.batch)
    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for x, y in train_loader:
            out = model(x.to(device)); y = y.to(device); loss = sum(ce(out[:, i], y[:, i]) for i in range(LENGTH)) / LENGTH
            opt.zero_grad(); loss.backward(); opt.step(); running += loss.item() * len(x)
        metrics = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "train_loss": running / len(train_ds), **metrics}))
        if metrics["captcha_accuracy"] >= best:
            best = metrics["captcha_accuracy"]
            Path(args.model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "chars": CHARS, "length": LENGTH, "size": [W, H], "val": metrics}, args.model)
    print(json.dumps({"model": args.model, "device": str(device), "best_val_captcha_accuracy": best}))

def predict(args):
    ckpt = torch.load(args.model, map_location="cpu"); model = CaptchaNet(); model.load_state_dict(ckpt["state_dict"]); model.eval()
    im = Image.open(args.image).convert("L").resize((W, H)); x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).float().view(1, 1, H, W) / 255.0
    out = model(x).softmax(-1)[0]; pred = out.argmax(-1)
    print(json.dumps({"image": args.image, "prediction": ''.join(CHARS[i] for i in pred.tolist()), "confidence": [round(float(out[i, pred[i]]), 4) for i in range(LENGTH)]}))

def test(args):
    ckpt = torch.load(args.model, map_location="cpu")
    model = CaptchaNet(); model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    metrics = evaluate(model.to(device), DataLoader(CaptchaDataset(Path(args.data)), args.batch), device)
    print(json.dumps({"model": args.model, "data": str(args.data), **metrics}))

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("output", type=Path); g.add_argument("--count", type=int, default=3000); g.add_argument("--seed", type=int, default=20261114); g.set_defaults(fn=lambda a: generate(a.output, a.count, a.seed))
    t = sub.add_parser("train"); t.add_argument("data", type=Path); t.add_argument("--model", default="artifacts/captcha_cnn.pt"); t.add_argument("--epochs", type=int, default=8); t.add_argument("--batch", type=int, default=64); t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--val-ratio", type=float, default=.2); t.add_argument("--seed", type=int, default=20261114); t.set_defaults(fn=train)
    q = sub.add_parser("predict"); q.add_argument("model"); q.add_argument("image"); q.set_defaults(fn=predict)
    e = sub.add_parser("test"); e.add_argument("data", type=Path); e.add_argument("--model", required=True); e.add_argument("--batch", type=int, default=64); e.set_defaults(fn=test)
    a = p.parse_args(); a.fn(a)
if __name__ == "__main__": main()
