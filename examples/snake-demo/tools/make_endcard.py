"""Render a score-board end card (PNG) from session summaries.

Style matches snake/replay.py TerminalRaster: dark panel, mono font, green accents.
Usage: python tools/make_endcard.py summaries.json --output endcard.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = "#090f13"
PANEL = "#0d161b"
FG = "#e3f3ef"
MUTED = "#68868c"
DIM = "#20353c"
GREEN = "#62f5b5"
AMBER = "#ffce73"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("A monospace font is required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, help="JSON: list of session summary dicts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--title", default="SYSTEM1-OMNI  ×  LAYA NATIVE  ×  H800")
    parser.add_argument("--footer", default="real recorded decisions · playback 1× · native Rust/CUDA worker · BF16")
    args = parser.parse_args()

    sessions = json.loads(args.summaries.read_text())
    total_steps = sum(s["steps"] for s in sessions)
    deaths = sum(s["deaths"] for s in sessions)
    best_speed = max(s["steps_per_second"] for s in sessions)
    mean_ms = sum(s["mean_inference_ms"] for s in sessions) / len(sessions)
    best_score = max(s["score"] for s in sessions)

    stats = [
        ("STEPS", f"{total_steps}", f"{len(sessions)} sessions"),
        ("DEATHS", f"{deaths}", "cycle shield on"),
        ("MOVES / S", f"{best_speed:.0f}", "max-speed, unpaced"),
        ("DECISION", f"{mean_ms:.2f} ms", "mean, same-host HTTP"),
        ("BEST SCORE", f"{best_score}", "single session"),
    ]

    img = Image.new("RGB", (args.width, args.height), BG)
    draw = ImageDraw.Draw(img)
    margin = 90
    draw.rounded_rectangle(
        (margin - 24, margin - 52, args.width - margin + 24, args.height - margin + 24),
        radius=18, fill=PANEL, outline=DIM, width=2,
    )
    f_title = load_font(44)
    f_big = load_font(110)
    f_label = load_font(30)
    f_small = load_font(24)
    draw.text((args.width // 2, margin + 8), args.title, font=f_title, fill=GREEN, anchor="mt")

    rows = (stats[:3], stats[3:])
    y_rows = ((290, 355, 525), (640, 705, 875))
    for row, (y_label, y_big, y_sub) in zip(rows, y_rows):
        n = len(row)
        col_w = (args.width - 2 * margin) / n
        for i, (label, value, sub) in enumerate(row):
            cx = margin + col_w * (i + 0.5)
            draw.text((cx, y_label), label, font=f_label, fill=MUTED, anchor="mt")
            draw.text((cx, y_big), value, font=f_big, fill=FG, anchor="mt")
            draw.text((cx, y_sub), sub, font=f_small, fill=DIM, anchor="mt")

    draw.text((args.width // 2, args.height - margin - 40), args.footer, font=f_small, fill=AMBER, anchor="mt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
