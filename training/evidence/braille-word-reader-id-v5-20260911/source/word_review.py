"""Render source-context contact sheets for geometric word boundary review."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("C:/bm-datasets/AngelinaDataset"))
    parser.add_argument("--split", choices=("validation", "external"), default="external")
    parser.add_argument("--limit", type=int, default=48)
    args = parser.parse_args()
    manifest = args.data/f"{args.split}-words.json"
    records = json.loads(manifest.read_text())["records"]
    out = args.data/f"{args.split}-boundary-review-{hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]}"
    out.mkdir(exist_ok=True)
    # Stratify by source and length before inference; do not pick correct output.
    selected = sorted(range(len(records)), key=lambda i: (len(records[i]["expected"]), records[i]["source"], i))
    if args.limit and len(selected)>args.limit:
        selected = [selected[round(i*(len(selected)-1)/(args.limit-1))] for i in range(args.limit)]
    for page in range(0, len(selected), 12):
        sheet = Image.new("RGB", (1200, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, index in enumerate(selected[page:page+12]):
            record = records[index]
            x, y = (slot%3)*400, (slot//3)*300
            left, top, right, bottom = record.get("guide", record["crop"])
            margin_x, margin_y = (right-left)*.22, (bottom-top)*.4
            context = (int(left-margin_x), int(top-margin_y), int(right+margin_x), int(bottom+margin_y))
            with Image.open(args.dataset/record["source"]) as source:
                crop = source.convert("RGB").crop(context).resize((390, 240), Image.Resampling.BILINEAR)
            painter = ImageDraw.Draw(crop)
            sx, sy = 390/(context[2]-context[0]), 240/(context[3]-context[1])
            painter.rectangle(((left-context[0])*sx, (top-context[1])*sy, (right-context[0])*sx, (bottom-context[1])*sy), outline="red", width=2)
            sheet.paste(crop, (x, y+40))
            draw.text((x+4, y+4), f"{index}: {record['expected']}", fill="black")
            draw.text((x+4, y+20), record["source"][-55:], fill="black")
        sheet.save(out/f"sheet-{page//12:03d}.jpg", quality=95)
    print(out, flush=True)


if __name__ == "__main__":
    main()
