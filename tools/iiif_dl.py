#!/usr/bin/env python3
"""
IIIF Manifest Image Downloader

Downloads manuscript page images from any IIIF manifest to disk.
Supports IIIF Presentation API v2, v3, and IIP (Image API v1.x) manifests.

Usage:
    # All pages, JPG, saved to current directory
    python tools/iiif_dl.py --manifest "https://example.org/manifest.json"

    # Pages 1-5 as TIFF, saved to a specific directory
    python tools/iiif_dl.py --manifest "https://example.org/manifest.json" \\
        --format tiff --pages 1-5 --output-dir ./downloads

    # Specific pages from a local manifest
    python tools/iiif_dl.py --manifest ./my_manifest.json --pages 1,3,7
"""

import argparse
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


# ── HTTP session ───────────────────────────────────────────────────────────────

# Several IIIF hosts (notably Gallica/BnF) reject the default `python-requests/x.y`
# User-Agent with 403 Forbidden, so every request goes through a session that
# identifies as a normal browser.
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/ld+json, application/json, image/*, */*",
    }
)


# ── IIIF helpers ───────────────────────────────────────────────────────────────


def _label(obj: dict) -> str:
    """Extract a plain string label from a IIIF v2 or v3 label field."""
    lbl = obj.get("label", "")
    if isinstance(lbl, dict):
        for lang_vals in lbl.values():
            if lang_vals:
                return str(lang_vals[0])
    if isinstance(lbl, list):
        return str(lbl[0]) if lbl else ""
    return str(lbl)


def _service_id(service) -> str | None:
    """Extract @id/id from a IIIF image service block (v2 or v3)."""
    if isinstance(service, list):
        service = service[0]
    if isinstance(service, dict):
        return service.get("@id") or service.get("id")
    return None


def _service_context(service) -> str | None:
    """Extract the @context from a service block to detect Image API version."""
    if isinstance(service, list):
        service = service[0]
    if isinstance(service, dict):
        ctx = service.get("@context", "")
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        return ctx
    return None


def _is_static_url(url: str) -> bool:
    """Return True if url looks like a direct image file rather than a IIIF base."""
    return bool(re.search(r"\.(jpe?g|tiff?|png|webp)(\?.*)?$", url, re.IGNORECASE))


def parse_manifest(manifest: dict) -> list[tuple[str, str, str | None]]:
    """Return a list of (canvas_label, service_or_image_id, service_context) triples.

    service_or_image_id is either:
      - a IIIF Image API service base URL  (append /full/size/0/quality.fmt)
      - a direct static image URL          (use as-is; flagged by _is_static_url)

    service_context is the @context string from the service block, used to
    determine Image API version (v1 / v2 / v3).
    """
    context = manifest.get("@context", "")
    if isinstance(context, list):
        context = " ".join(context)
    is_v3 = "presentation/3" in context or manifest.get("type") == "Manifest"

    canvases: list[tuple[str, str, str | None]] = []

    if is_v3:
        for canvas in manifest.get("items", []):
            label = _label(canvas)
            for anno_page in canvas.get("items", []):
                for anno in anno_page.get("items", []):
                    body = anno.get("body", {})
                    service = body.get("service")
                    if service:
                        sid = _service_id(service)
                        sctx = _service_context(service)
                        if sid:
                            canvases.append((label, sid, sctx))
                            break
                    # Fallback: body id is a static image or bare service base
                    body_id = body.get("id") or body.get("@id")
                    if body_id:
                        canvases.append((label, body_id, None))
                        break
    else:
        # Prezi v2
        for sequence in manifest.get("sequences", []):
            for canvas in sequence.get("canvases", []):
                label = _label(canvas)
                for image in canvas.get("images", []):
                    resource = image.get("resource", {})
                    service = resource.get("service")
                    if service:
                        sid = _service_id(service)
                        sctx = _service_context(service)
                        if sid:
                            canvases.append((label, sid, sctx))
                            break
                    res_id = resource.get("@id") or resource.get("id", "")
                    if res_id:
                        canvases.append((label, res_id, None))
                        break

    return canvases


# ── URL construction ───────────────────────────────────────────────────────────


def build_image_url(
    service_id: str,
    fmt: str,
    max_dim: int | None,
    service_ctx: str | None,
) -> str:
    """Build a IIIF Image API request URL.

    fmt should be 'jpg' or 'tif'.
    Detects Image API v1.x (IIP) vs v2/v3 to pick the right quality keyword.
    If service_id is already a static image URL it is returned unchanged.
    """
    if _is_static_url(service_id):
        return service_id

    base = service_id.rstrip("/")
    ctx = service_ctx or ""
    if max_dim:
        size = f"!{max_dim},{max_dim}"
    elif "image/3" in ctx or "/iiif/3/" in base:
        size = "max"   # 'full' is invalid in Image API v3
    else:
        size = "full"
    quality = "native" if "image/1" in ctx else "default"
    return f"{base}/full/{size}/0/{quality}.{fmt}"


# ── page filter ────────────────────────────────────────────────────────────────


def parse_pages(spec: str) -> set[int]:
    """Parse '1-5', '1,3,5', or mixed '1-3,7' into a set of 1-indexed integers."""
    pages: set[int] = set()
    for part in re.split(r"[,\s]+", spec):
        part = part.strip()
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            pages.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.fullmatch(r"\d+", part):
            pages.add(int(part))
        else:
            raise ValueError(f"Cannot parse page spec: '{part}'")
    return pages


# ── download ───────────────────────────────────────────────────────────────────


def _get_with_retry(url: str, timeout: int, retries: int = 3) -> requests.Response:
    """GET url, retrying up to `retries` times on 5xx/429 responses with backoff."""
    delay = 5
    for attempt in range(retries):
        resp = SESSION.get(url, timeout=timeout)
        if resp.status_code < 500 and resp.status_code != 429:
            return resp
        if attempt < retries - 1:
            # Honour Retry-After when the server throttles us (Gallica does).
            wait = delay
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, int(retry_after))
            time.sleep(wait)
            delay *= 2
    return resp  # return last response for the caller to handle


def _manifest_error_hint(resp: requests.Response) -> str:
    """Build an actionable error message for a failed manifest fetch."""
    code = resp.status_code
    lines = [f"ERROR: could not load manifest — HTTP {code} {resp.reason} for {resp.url}"]

    if code in (401, 403):
        lines.append(
            "  The server refused the request. It may require authentication, or be "
            "blocking this client — try --user-agent with your browser's UA string."
        )
    elif code == 404:
        lines.append("  No manifest at that URL — check the identifier and the URL pattern.")
    elif code == 429:
        lines.append("  Rate limited. Wait a few minutes and retry.")
    elif code >= 500:
        lines.append(
            "  This is a server-side failure, not a problem with your request. The "
            "identifier may not exist or may not be exposed over IIIF."
        )

    # Servers often explain themselves in the body; surface it when it is short.
    body = resp.text.strip()
    if body and len(body) < 500 and "<html" not in body[:200].lower():
        lines.append(f"  Server said: {body}")

    return "\n".join(lines)


def _save_as_tiff_from_bytes(data: bytes, dest: Path) -> None:
    """Decode image bytes (any format) and save as lossless TIFF."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(dest, format="TIFF", compression="none")


def download_and_save(
    url: str,
    dest: Path,
    fmt: str,
    timeout: int,
    service_id: str,
    service_ctx: str | None,
    max_dim: int | None,
) -> None:
    """Download image at url and save to dest.

    For TIFF, tries the server's native TIFF endpoint first; falls back to
    downloading JPEG and converting via Pillow if the server returns any error
    (400 unsupported format, 404, 504 timeout, etc.).
    """
    resp = _get_with_retry(url, timeout)

    if fmt == "tif" and not resp.ok:
        # Server can't serve TIFF (wrong format, timeout, unsupported) —
        # fall back to JPG download + lossless TIFF conversion
        jpg_url = build_image_url(service_id, "jpg", max_dim, service_ctx)
        resp = _get_with_retry(jpg_url, timeout)
        resp.raise_for_status()
        _save_as_tiff_from_bytes(resp.content, dest)
        return

    resp.raise_for_status()

    if fmt == "tif":
        # Server returned bytes claiming to be TIFF — verify, then save or re-encode
        try:
            img = Image.open(BytesIO(resp.content))
            img.verify()
            dest.write_bytes(resp.content)
        except Exception:
            _save_as_tiff_from_bytes(resp.content, dest)
    else:
        dest.write_bytes(resp.content)


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download manuscript images from a IIIF manifest (Prezi 2, 3, IIP)"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        metavar="URL_OR_PATH",
        help="IIIF manifest URL or local path to a manifest JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to save images (default: current working directory)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["jpg", "tiff"],
        default="jpg",
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--pages",
        "-p",
        default=None,
        metavar="SPEC",
        help="Pages to download, e.g. '1-10', '1,3,5', '1-3,7' (default: all)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        metavar="N",
        help="Cap image size at N pixels on the longest side (default: full resolution)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="S",
        help="Per-image HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        metavar="STRING",
        help="Override the User-Agent header (default: a browser UA, since some "
        "hosts reject the requests default with 403)",
    )
    args = parser.parse_args()

    if args.user_agent:
        SESSION.headers["User-Agent"] = args.user_agent

    # Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise format: 'tiff' → 'tif' for IIIF URL construction
    fmt_url = "tif" if args.format == "tiff" else "jpg"
    fmt_ext = "tiff" if args.format == "tiff" else "jpg"

    # Parse page filter
    page_filter: set[int] | None = None
    if args.pages and args.pages.lower() != "all":
        try:
            page_filter = parse_pages(args.pages)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    # Load manifest
    src = args.manifest
    try:
        if src.startswith("http://") or src.startswith("https://"):
            print(f"Fetching manifest: {src}")
            resp = _get_with_retry(src, 30)
            if not resp.ok:
                print(_manifest_error_hint(resp), file=sys.stderr)
                sys.exit(1)
            manifest = resp.json()
        else:
            with open(src) as fh:
                manifest = json.load(fh)
    except Exception as e:
        print(f"ERROR: could not load manifest — {e}", file=sys.stderr)
        sys.exit(1)

    canvases = parse_manifest(manifest)
    if not canvases:
        print("ERROR: no canvases found in manifest.", file=sys.stderr)
        sys.exit(1)

    # Apply page filter
    if page_filter:
        canvases = [(lbl, svc, sctx) for i, (lbl, svc, sctx) in enumerate(canvases, 1) if i in page_filter]
        if not canvases:
            print("ERROR: page filter matched no canvases.", file=sys.stderr)
            sys.exit(1)

    total = len(canvases)
    print(f"\nManifest : {src}")
    print(f"Pages    : {args.pages or 'all'} ({total} canvases)")
    print(f"Format   : {args.format}")
    print(f"Max dim  : {args.max_dim or 'full resolution'}")
    print(f"Output   : {output_dir}\n")

    downloaded = 0
    skipped = 0

    for idx, (label, service_id, service_ctx) in enumerate(
        tqdm(canvases, unit="page", desc="Downloading"), start=1
    ):
        safe_label = re.sub(r"[^\w\-.]", "_", label).strip("_") or f"page_{idx:04d}"
        filename = f"{idx:04d}_{safe_label}.{fmt_ext}"
        dest = output_dir / filename

        url = build_image_url(service_id, fmt_url, args.max_dim, service_ctx)

        try:
            download_and_save(url, dest, fmt_url, args.timeout, service_id, service_ctx, args.max_dim)
            downloaded += 1
        except Exception as e:
            tqdm.write(f"  WARNING [{idx}/{total}] {label!r} — {e}")
            skipped += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped → {output_dir}")


if __name__ == "__main__":
    main()
