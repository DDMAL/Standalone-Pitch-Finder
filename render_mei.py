"""
render_mei.py  —  在手稿原图上，把音名字母标注在每个 neume 正下方

用法：
    python3 render_mei.py <mei_file> <image_file> [output.jpg]

例：
    python3 render_mei.py mei_reviewed/CH-E_611_081r.mei CH-E-611_Einsiedeln/81r.jpg out.jpg
"""
import sys, xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

NS = 'http://www.music-encoding.org/ns/mei'

def parse_mei(path):
    tree = ET.parse(path)
    root = tree.getroot()

    zones = {}
    for z in root.iter(f'{{{NS}}}zone'):
        zid = z.get(f'{{http://www.w3.org/XML/1998/namespace}}id')
        try:
            zones[zid] = (int(z.get('ulx')), int(z.get('uly')),
                          int(z.get('lrx')), int(z.get('lry')))
        except (TypeError, ValueError):
            pass

    # list of (cx, bottom_y, pname)  — one entry per nc
    notes = []
    for nc in root.iter(f'{{{NS}}}nc'):
        facs = nc.get('facs', '').lstrip('#')
        pname = nc.get('pname')
        if facs in zones and pname:
            ulx, uly, lrx, lry = zones[facs]
            cx = (ulx + lrx) // 2
            notes.append((cx, lry, pname.lower()))
    return notes

def render(mei_path, img_path, out_path, scale=0.35):
    notes = parse_mei(mei_path)
    if not notes:
        print("No notes found"); return

    img = Image.open(img_path).convert('RGB')
    W, H = img.size

    # font size relative to original image width (~4800px → 80px letters)
    font_size = max(40, W // 60)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    pad = font_size // 6

    for cx, lry, pname in notes:
        # white halo for legibility, then black letter
        y = lry + pad
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            draw.text((cx + dx, y + dy), pname, font=font, fill=(255,255,255), anchor='mt')
        draw.text((cx, y), pname, font=font, fill=(20,20,20), anchor='mt')

    new_w, new_h = int(W * scale), int(H * scale)
    img.resize((new_w, new_h), Image.LANCZOS).save(out_path, quality=88)
    print(f"Saved → {out_path}  ({new_w}×{new_h}px,  {len(notes)} notes)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    render(sys.argv[1], sys.argv[2],
           sys.argv[3] if len(sys.argv) > 3 else 'overlay.jpg')
