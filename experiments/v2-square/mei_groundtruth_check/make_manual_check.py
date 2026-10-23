"""Build a blind manual spot-check page for the option-(a) note boxes.

Samples (fixed seed) 100 'no_blob' + 60 'shifted' + 40 'verified' notes from
einsiedeln/bbox_verification/per_note.csv (run verify_bboxes.py first), shuffles them,
and writes a self-contained HTML page: one tile per note, the box under test in
red, the other notes of the same neume in grey, and three buttons per tile.
The tile's verification status is NOT shown (it's in key.csv), so the judge
isn't primed. Verdicts persist in the browser's localStorage and are exported
as a CSV from the page.

    python make_manual_check.py --image-dir <dir with Einsiedeln__Stiftsbibliothek__Codex_611_<folio>.jpg>
"""
import argparse
import base64
import csv
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from fix_zones import EINSIEDELN, M, PROCESSED, REVIEWED, XMLID, parse, staff_step_px

OUT = EINSIEDELN / "manual_check"
SAMPLE = {"no_blob": 100, "shifted": 60, "verified": 40}
SEED = 0
SCALE = 1.7


def neume_members(path):
    root = ET.parse(path).getroot()
    out = {}
    for neume in root.iter(M + "neume"):
        ids = [nc.get(XMLID) for nc in neume.findall(M + "nc")]
        for i in ids:
            out[i] = ids
    return out


def tile(img, box, siblings, step, wz):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = 2.4 * wz, 3.5 * step
    X0, Y0 = int(max(0, cx - hw)), int(max(0, cy - hh))
    X1, Y1 = int(min(img.shape[1], cx + hw)), int(min(img.shape[0], cy + hh))
    crop = cv2.resize(img[Y0:Y1, X0:X1].copy(), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

    def rect(b, color, t):
        p = lambda x, y: (int((x - X0) * SCALE), int((y - Y0) * SCALE))
        cv2.rectangle(crop, p(b[0], b[1]), p(b[2], b[3]), color, t)
    for b in siblings:
        rect(b, (150, 150, 150), 2)
    rect(box, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Note bbox manual check</title>
<style>
body{font-family:-apple-system,Helvetica,sans-serif;margin:0;background:#f4f1ea;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:10px 16px;z-index:9}
header h1{font-size:17px;margin:0 0 6px}
header p{margin:2px 0;font-size:13px;line-height:1.5}
#bar{height:6px;background:#ddd;border-radius:3px;margin-top:6px}#fill{height:6px;background:#2a8;border-radius:3px;width:0}
.grid{display:flex;flex-wrap:wrap;gap:12px;padding:16px}
.card{background:#fff;border:3px solid #ddd;border-radius:6px;padding:6px;width:min(420px,100%)}
.card.cur{border-color:#36c}.card.v-ok{background:#e3f6e8}.card.v-bad{background:#fbe3e1}.card.v-unsure{background:#fff4d6}
.card img{width:100%;display:block}
.row{display:flex;justify-content:space-between;align-items:center;margin-top:6px}
.num{font-weight:bold}
button{font-size:14px;padding:5px 10px;margin-left:4px;border:1px solid #999;border-radius:4px;background:#fafafa;cursor:pointer}
button.sel{background:#333;color:#fff}
#export{background:#2a8;color:#fff;border-color:#2a8}
</style></head><body>
<header>
<h1>音符框人工抽查（共 __N__ 张）</h1>
<p><b>判断标准：</b>红框是不是框在了它自己那个符头上？灰框是同一个neume里的其他音符，用来帮你区分"框住了自己"还是"框住了旁边那个"。</p>
<p><b>✓ 对</b>：红框落在一个符头上，而且是它自己那个（不是灰框对应的那个）。<b>✗ 错</b>：红框落在空白、文字、谱线上，或者明显框的是旁边那个音符。<b>? 不确定</b>：看不清或拿不准。</p>
<p><b>注意：</b>这批框普遍比符头<b>略低一点</b>（大约半个框高），这是所有框统一的偏差、之后可以整体校正。请判断红框<b>对应的是不是这个符头</b>，不要因为没有严丝合缝就判错。</p>
<p>快捷键：<b>1</b>=对 <b>2</b>=错 <b>3</b>=不确定（自动跳到下一张），<b>←/→</b> 切换。结果自动保存在浏览器里；做完点 <button id="export">导出结果 CSV</button>
<span id="prog"></span></p><div id="bar"><div id="fill"></div></div>
</header>
<div class="grid" id="grid"></div>
<script>
const TILES = __TILES__;
const KEY = "manual_check_v1";
let verdict = {}; try { verdict = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let cur = 0;
const grid = document.getElementById("grid");
const LABEL = {ok: "✓ 对", bad: "✗ 错", unsure: "? 不确定"};
TILES.forEach((t, i) => {
  const c = document.createElement("div"); c.className = "card"; c.id = "c" + i;
  c.innerHTML = `<img src="data:image/jpeg;base64,${t.img}"><div class="row"><span class="num">#${t.n}</span><span>` +
    Object.keys(LABEL).map(v => `<button data-v="${v}">${LABEL[v]}</button>`).join("") + `</span></div>`;
  c.querySelectorAll("button").forEach(b => b.onclick = () => { cur = i; mark(b.dataset.v, false); });
  c.onclick = () => { cur = i; refresh(); };
  grid.appendChild(c);
});
function save() { try { localStorage.setItem(KEY, JSON.stringify(verdict)); } catch (e) {} }
function mark(v, advance) {
  verdict[TILES[cur].n] = v; save();
  if (advance && cur < TILES.length - 1) cur++;
  refresh(true);
}
function refresh(scroll) {
  TILES.forEach((t, i) => {
    const c = document.getElementById("c" + i), v = verdict[t.n];
    c.className = "card" + (i === cur ? " cur" : "") + (v ? " v-" + v : "");
    c.querySelectorAll("button").forEach(b => b.classList.toggle("sel", b.dataset.v === v));
  });
  const done = Object.keys(verdict).length;
  document.getElementById("prog").textContent = ` 已完成 ${done} / ${TILES.length}`;
  document.getElementById("fill").style.width = (100 * done / TILES.length) + "%";
  if (scroll) document.getElementById("c" + cur).scrollIntoView({block: "center", behavior: "smooth"});
}
document.addEventListener("keydown", e => {
  if (e.key === "1") mark("ok", true); else if (e.key === "2") mark("bad", true);
  else if (e.key === "3") mark("unsure", true);
  else if (e.key === "ArrowRight") { cur = Math.min(TILES.length - 1, cur + 1); refresh(true); }
  else if (e.key === "ArrowLeft") { cur = Math.max(0, cur - 1); refresh(true); }
});
document.getElementById("export").onclick = () => {
  const lines = ["tile,folio,id,verdict"].concat(TILES.map(t => [t.n, t.folio, t.id, verdict[t.n] || ""].join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")], {type: "text/csv"}));
  a.download = "manual_check_results.csv"; a.click();
};
refresh(false);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    rows = [r for r in csv.DictReader(open(EINSIEDELN / "bbox_verification" / "per_note.csv"))
            if r["variant"] == "a" and r["kind"] == "nc"]
    rng = random.Random(SEED)
    picked = []
    for status, k in SAMPLE.items():
        pool = [r for r in rows if r["status"] == status]
        picked += rng.sample(pool, k)
    rng.shuffle(picked)

    cache, tiles, key = {}, [], []
    for n, r in enumerate(picked, 1):
        f = r["folio"]
        if f not in cache:
            rev = parse(REVIEWED / f"CH-E_611_{f}.mei")
            proc = {p["id"]: p["box"] for p in parse(PROCESSED / f"CH-E_611_{f}.mei")}
            steps = staff_step_px(rev)
            wz = sorted(p["box"][2] - p["box"][0] for p in rev if p["kind"] == "nc")
            img = cv2.imread(str(args.image_dir / f"Einsiedeln__Stiftsbibliothek__Codex_611_{f}.jpg"))
            cache[f] = (proc, steps, wz[len(wz) // 2], img, neume_members(REVIEWED / f"CH-E_611_{f}.mei"))
        proc, steps, wz, img, members = cache[f]
        box = proc[r["id"]]
        sib = [proc[i] for i in members.get(r["id"], []) if i != r["id"] and i in proc]
        step = steps[int(r["staff"])][0]
        tiles.append(dict(n=n, folio=f, id=r["id"], img=tile(img, box, sib, step, wz)))
        key.append(dict(tile=n, folio=f, id=r["id"], status=r["status"], neume_size=r["neume_size"],
                        res_dy_steps=r["res_dy_steps"]))

    (OUT / "manual_check.html").write_text(
        HTML.replace("__N__", str(len(tiles))).replace("__TILES__", json.dumps(tiles)), encoding="utf-8")
    with open(OUT / "key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    print(f"wrote {OUT / 'manual_check.html'} ({len(tiles)} tiles) and {OUT / 'key.csv'}")


if __name__ == "__main__":
    main()
