#!/usr/bin/env python3
"""Generate labelled Kaptcha-like images without ML dependencies."""
import argparse, random, string
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
CHARS=string.digits+string.ascii_lowercase
LENGTH=5
WIDTH, HEIGHT = 120, 38
FONTS=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']
def font(size):
    for p in FONTS:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()
def render(text,rng,w=WIDTH,h=HEIGHT):
    # production Kaptcha defaults: white background, black 30px text,
    # five-pixel character spacing, no border, and DefaultNoise in black.
    im=Image.new('L',(w,h),255); d=ImageDraw.Draw(im); x=5
    for ch in text:
        d.text((x, 1 + rng.randint(-1, 1)), ch, font=font(30), fill=0)
        x += 22 + 5
    for _ in range(2):
        d.line((rng.randrange(w // 3), rng.randrange(h), rng.randrange(w), rng.randrange(h)), fill=0, width=1)
    return im
def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--count',type=int,default=3000); p.add_argument('--seed',type=int,default=20261114); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); r=random.Random(a.seed)
    for i in range(a.count):
        while True:
            label=''.join(r.choice(CHARS) for _ in range(LENGTH))
            if any(c.isdigit() for c in label) and any(c.isalpha() for c in label):
                break
        render(label,r).save(a.output/f'{label}_{i:06d}.png')
    print(f'generated {a.count} images in {a.output}')
if __name__=='__main__': main()
