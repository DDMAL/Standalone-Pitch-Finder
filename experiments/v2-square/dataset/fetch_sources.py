"""Fetch what build_dataset.py reads into the repo's gitignored data/ folder:

  data/mei/einsiedeln/{reviewed,reviewed_once,processed}/   from DDMAL/e2e-omr-resources (public)
  data/mei/ms73/processed/
  data/hf_listings/{einsiedeln,ms73}.json                    file listings of the private HF
                                                              image folders (names only, no images)
  data/images/{einsiedeln,ms73}/                             with --images: the page image of every
                                                              Reviewed / Reviewed_once / MS73 MEI page

MEI files already present with the right size are skipped. The HF listings
need HF_TOKEN (an HF token with DDMAL-lab org access) in the environment and
are skipped without it if a cached listing exists.

    HF_TOKEN=... python fetch_sources.py [--images]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
GH_TREE = "https://api.github.com/repos/DDMAL/e2e-omr-resources/git/trees/main?recursive=1"
GH_RAW = "https://raw.githubusercontent.com/DDMAL/e2e-omr-resources/main/"
MEI_FOLDERS = {
    "resulting_mei_files/Einsiedeln/Reviewed no pitch correction needed": "mei/einsiedeln/reviewed",
    "resulting_mei_files/Einsiedeln/Reviewed_once": "mei/einsiedeln/reviewed_once",
    "resulting_mei_files/Einsiedeln/Processed": "mei/einsiedeln/processed",
    "resulting_mei_files/MS73/Processed": "mei/ms73/processed",
}
HF_API = "https://huggingface.co/api/datasets/DDMAL-lab/All-Images/tree/main/"
HF_RESOLVE = "https://huggingface.co/datasets/DDMAL-lab/All-Images/resolve/main/"
HF_FOLDERS = {"Pics Einsiedeln": "hf_listings/einsiedeln.json", "Pics MS73": "hf_listings/ms73.json"}


def fetch_mei():
    with urllib.request.urlopen(GH_TREE) as r:
        tree = json.load(r)["tree"]
    jobs = []
    for src, dst in MEI_FOLDERS.items():
        (DATA / dst).mkdir(parents=True, exist_ok=True)
        for x in tree:
            if x["path"].startswith(src + "/") and x["path"].endswith(".mei"):
                out = DATA / dst / x["path"].rsplit("/", 1)[1]
                if not out.exists() or out.stat().st_size != x["size"]:
                    jobs.append((x["path"], out, x["size"]))

    def get(job):
        path, out, size = job
        subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(out), GH_RAW + urllib.parse.quote(path)])
        return out.exists() and out.stat().st_size == size, path

    with cf.ThreadPoolExecutor(16) as ex:
        failed = [p for ok, p in ex.map(get, jobs) if not ok]
    for dst in MEI_FOLDERS.values():
        print(f"{dst}: {len(list((DATA / dst).glob('*.mei')))} files")
    print(f"downloaded {len(jobs) - len(failed)}, failed {len(failed)} {failed[:5]}")


def fetch_listings():
    token = os.environ.get("HF_TOKEN")
    for folder, dst in HF_FOLDERS.items():
        out = DATA / dst
        if not token:
            print(f"{dst}: {'cached' if out.exists() else 'MISSING'} (no HF_TOKEN set)")
            continue
        url, entries = HF_API + urllib.parse.quote(folder) + "?limit=1000", []
        while url:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req) as r:
                entries += json.load(r)
                m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link") or "")
                url = m.group(1) if m else None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(entries))
        print(f"{dst}: {len(entries)} entries")


def fetch_images():
    from build_dataset import einsiedeln_image, ms73_image, listing
    token = os.environ["HF_TOKEN"]
    jobs = []
    for ms, folder, mei_dirs in (("einsiedeln", "Pics Einsiedeln", ("reviewed", "reviewed_once")),
                                 ("ms73", "Pics MS73", ("processed",))):
        names = listing(ms)
        sizes = {e["path"].split("/", 1)[1]: (e.get("lfs") or {}).get("size", e["size"])
                 for e in json.loads((DATA / HF_FOLDERS[folder]).read_text()) if e["type"] == "file"}
        wanted = set()
        for d in mei_dirs:
            for f in (DATA / "mei" / ms / d).glob("*.mei"):
                if ms == "einsiedeln":
                    wanted.add(einsiedeln_image(f.stem.rsplit("_", 1)[1], names))
                else:
                    wanted.add(ms73_image(int(re.search(r"[-_](\d{3})$", f.stem).group(1)), names))
        wanted.discard("")
        (DATA / "images" / ms).mkdir(parents=True, exist_ok=True)
        for name in sorted(wanted):
            out = DATA / "images" / ms / name
            if not out.exists() or out.stat().st_size != sizes[name]:
                jobs.append((f"{folder}/{name}", out, sizes[name]))
    print(f"images to download: {len(jobs)}, {sum(j[2] for j in jobs) / 1e9:.1f} GB")

    def get(job):
        path, out, size = job
        tmp = out.with_suffix(out.suffix + ".part")
        subprocess.run(["curl", "-sfL", "--retry", "3", "-H", f"Authorization: Bearer {token}",
                        "-o", str(tmp), HF_RESOLVE + urllib.parse.quote(path)])
        if tmp.exists() and tmp.stat().st_size == size:
            tmp.rename(out)
            return True, path
        tmp.unlink(missing_ok=True)
        return False, path

    done = 0
    with cf.ThreadPoolExecutor(8) as ex:
        failed = []
        for ok, path in ex.map(get, jobs):
            done += 1
            if not ok:
                failed.append(path)
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"images downloaded {len(jobs) - len(failed)}, failed {len(failed)} {failed[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="also download the page images (~6 GB)")
    args = ap.parse_args()
    fetch_mei()
    fetch_listings()
    if args.images:
        fetch_images()
