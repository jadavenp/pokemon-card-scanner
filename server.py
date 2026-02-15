"""
server.py — FastAPI server for the Pokemon card scanner

Wires together the camera feed, scanner pipeline, pricing cache, and
web dashboard into a single HTTP server. Designed to run on the Jetson
Orin Nano and be accessed from any browser on the local network.

Architecture:
    Jetson Orin Nano (or Mac in mock mode)
    ├── Camera capture      (camera.py)
    ├── Scanner pipeline    (scanner.py — process_image)
    ├── Pricing cache       (pricing_cache.py)
    └── FastAPI server      (this file)
        ├── GET  /              → web dashboard (scanner_dashboard.html)
        ├── GET  /video_feed    → live MJPEG stream from camera
        ├── POST /scan          → trigger scan, return results + cache stats
        ├── GET  /api/status    → camera status, API quota, cache stats
        └── POST /api/cache     → cache management (clear, stats)

Usage:
    # Development (Mac, mock camera):
    python server.py

    # Production (Jetson, CSI camera):
    python server.py --host 0.0.0.0 --port 8080

    # Override camera mode:
    python server.py --camera-mode mock --mock-dir test_images/
"""

import argparse
import asyncio
import json
import logging
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
import uvicorn

# ─────────────────────────────────────────────────────────────
# PROJECT IMPORTS
# These are loaded from the same directory as server.py.
# On first import, heavy resources (OCR model, hash DB) are NOT
# loaded — that happens in the lifespan startup handler below.
# ─────────────────────────────────────────────────────────────

from camera import Camera
from pricing_cache import get_cache

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ─────────────────────────────────────────────────────────────
# GLOBALS
# These are initialized during the lifespan startup event.
# ─────────────────────────────────────────────────────────────

# Camera instance (set in lifespan)
cam: Optional[Camera] = None

# Scanner pipeline resources (loaded once at startup)
# These are the heavy objects that process_image() needs.
scanner_resources: dict = {
    "reader": None,          # EasyOCR reader
    "index": None,           # card_index.json data
    "stamp_template": None,  # 1st Edition stamp template
    "hash_db": None,         # Perceptual hash database
    "loaded": False,         # True once all resources are ready
}

# Lock to prevent concurrent scans (process_image is not thread-safe
# due to shared OCR model state)
scan_lock = threading.Lock()

# Server start time (for uptime display)
start_time: float = 0.0

# CLI args (set in main, read in lifespan)
cli_args: Optional[argparse.Namespace] = None


# ─────────────────────────────────────────────────────────────
# RESOURCE LOADING
# ─────────────────────────────────────────────────────────────

def load_scanner_resources() -> None:
    """
    Load the heavy scanner pipeline resources:
      - EasyOCR reader (~4s)
      - card_index.json (~73 MB)
      - 1st Edition stamp template
      - Perceptual hash database (~2.7 MB, 18,391 hashes)

    Called once during server startup. These objects are reused
    for every /scan request.
    """
    import easyocr
    from database import load_database
    from stamp import load_stamp_template
    from image_match import load_hash_database

    logger.info("Loading scanner resources...")
    t0 = time.time()

    # EasyOCR reader (English, GPU if available)
    logger.info("  Loading EasyOCR model...")
    scanner_resources["reader"] = easyocr.Reader(["en"], gpu=True, verbose=False)

    # Card index database (path resolved internally via config.DATABASE_FILE)
    logger.info("  Loading card index...")
    scanner_resources["index"], _ = load_database()

    # Stamp template for 1st Edition detection (path resolved via config.TEMPLATE_PATH)
    logger.info("  Loading stamp template...")
    scanner_resources["stamp_template"] = load_stamp_template()

    # Perceptual hash database (path resolved via image_match.HASH_DB_FILE)
    logger.info("  Loading hash database...")
    scanner_resources["hash_db"] = load_hash_database()

    scanner_resources["loaded"] = True
    elapsed = time.time() - t0
    logger.info("Scanner resources loaded in %.1fs", elapsed)


# ─────────────────────────────────────────────────────────────
# FASTAPI LIFESPAN (startup/shutdown)
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load scanner resources + start camera.
    Shutdown: stop camera + cleanup.
    """
    global cam, start_time

    start_time = time.time()

    # ── Load scanner pipeline resources ──
    # Run in a thread so we don't block the event loop during
    # the ~5-6s load time.
    logger.info("Starting server initialization...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_scanner_resources)

    # ── Initialize camera ──
    args = cli_args
    camera_mode = getattr(args, "camera_mode", None) if args else None
    mock_dir = getattr(args, "mock_dir", None) if args else None

    cam = Camera(mode=camera_mode, mock_dir=mock_dir)
    if cam.start():
        logger.info("Camera started: %s", cam)
    else:
        logger.warning("Camera failed to start — dashboard will show placeholder")

    logger.info("Server ready")

    # Pre-create temp directory for phone uploads (avoids race on first upload)
    import tempfile
    Path(tempfile.gettempdir(), "card-scanner-captures").mkdir(parents=True, exist_ok=True)

    yield  # ── Server is running ──

    # ── Shutdown ──
    if cam is not None:
        cam.stop()
    logger.info("Server shutdown complete")


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Card Scanner — All C's Collectibles",
    lifespan=lifespan,
)

# Resolve paths relative to this file
PROJECT_DIR = Path(__file__).parent
DASHBOARD_PATH = PROJECT_DIR / "scanner_dashboard.html"
MOBILE_DASHBOARD_PATH = PROJECT_DIR / "scanner_mobile.html"


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

# ──────────────── GET / ── Dashboard ────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the scanner dashboard HTML."""
    if not DASHBOARD_PATH.exists():
        return HTMLResponse(
            content="<h1>Dashboard not found</h1>"
                    f"<p>Expected at: {DASHBOARD_PATH}</p>",
            status_code=404,
        )
    return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))


# ──────────────── GET /video_feed ── MJPEG Stream ────────────────

@app.get("/video_feed")
async def video_feed():
    """
    Live MJPEG stream from the camera.

    The dashboard embeds this as <img src="/video_feed">.
    Each frame is a JPEG boundary in a multipart stream.
    In mock mode, serves the current mock image at ~10 FPS.
    """
    if cam is None or not cam.is_running:
        return JSONResponse(
            content={"error": "Camera not running"},
            status_code=503,
        )

    return StreamingResponse(
        _generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _generate_mjpeg_frames():
    """
    Async generator yielding MJPEG frame boundaries.

    Runs in a loop, reading frames from the camera and yielding
    them as multipart boundaries. Frame rate is controlled by a
    small sleep to avoid saturating the connection.
    """
    while True:
        # Read frame in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        jpeg_bytes = await loop.run_in_executor(None, cam.read_jpeg)

        if jpeg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

        # Target ~15 FPS for the stream (CSI runs at 30, but the
        # MJPEG stream doesn't need that much bandwidth)
        await asyncio.sleep(0.066)


# ──────────────── POST /scan ── Trigger Scan ────────────────

@app.post("/scan")
async def scan_card(request: Request):
    """
    Trigger a card scan using the current camera frame.

    Request body (JSON):
        {
            "use_cache": true   // true = batch mode (cached), false = single (fresh)
        }

    Response (JSON):
        {
            "result": { ... process_image() output ... },
            "cache": { ... pricing cache stats ... },
            "camera": { ... camera status ... },
            "scan_time": 2.8
        }

    The result dict matches the exact structure expected by the
    dashboard's addCardToSession() function.
    """
    # ── Validate prerequisites ──
    if not scanner_resources["loaded"]:
        return JSONResponse(
            content={"error": "Scanner resources still loading — try again in a few seconds"},
            status_code=503,
        )

    if cam is None or not cam.is_running:
        return JSONResponse(
            content={"error": "Camera not running"},
            status_code=503,
        )

    # ── Parse request body ──
    try:
        body = await request.json()
    except Exception:
        body = {}

    use_cache = body.get("use_cache", True)

    # ── Capture frame ──
    # In mock mode, use the original PNG path directly (better quality
    # than re-encoding to JPEG). In CSI mode, save frame to temp file.
    if cam.mode == "mock" and cam.current_mock_path:
        img_path = cam.current_mock_path
    else:
        img_path = cam.capture_to_file()

    if img_path is None:
        return JSONResponse(
            content={"error": "Failed to capture frame from camera"},
            status_code=500,
        )

    # ── Run scanner pipeline ──
    # process_image() is CPU-bound and not thread-safe (shared OCR state),
    # so we serialize access with a lock and run in a thread pool.
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            _run_scan,
            img_path,
            use_cache,
        )
    except Exception as e:
        logger.exception("Scan failed: %s", e)
        return JSONResponse(
            content={"error": f"Scan failed: {str(e)}"},
            status_code=500,
        )

    # ── Advance mock camera to next image ──
    cam.advance()

    # ── Build response ──
    cache_stats = get_cache().stats()

    return JSONResponse(content={
        "result": result,
        "cache": cache_stats,
        "camera": cam.status(),
        "scan_time": result.get("time", 0),
    })


def _run_scan(img_path: str, use_cache: bool) -> dict:
    """
    Execute the scanner pipeline (blocking, called from thread pool).

    This wraps process_image() with timing and the pricing cache flag.
    The scan_lock ensures only one scan runs at a time.

    Args:
        img_path: Path to the image file to scan
        use_cache: Whether to use the pricing cache (batch mode)

    Returns:
        dict matching the process_image() return format, with added
        'time' and '_cached' fields.
    """
    from scanner import process_image

    with scan_lock:
        t0 = time.time()

        result = process_image(
            img_path=img_path,
            reader=scanner_resources["reader"],
            index=scanner_resources["index"],
            stamp_template=scanner_resources["stamp_template"],
            hash_db=scanner_resources["hash_db"],
            verbose=False,
            use_cache=use_cache,
        )

        elapsed = time.time() - t0
        result["time"] = round(elapsed, 2)

        # Tag pricing mode for dashboard display
        result["_pricing_mode"] = "batch" if use_cache else "single"

        logger.info(
            "Scan complete: %s (%s) — %s — %.2fs",
            result.get("name", "?"),
            result.get("card_id", "?"),
            result.get("price", "N/A"),
            elapsed,
        )

        return result



# ──────────────── GET /mobile ── Mobile Scanner UI ────────────────

@app.get("/mobile", response_class=HTMLResponse)
async def serve_mobile_dashboard():
    """Serve the mobile scanner UI (phone-optimized)."""
    if not MOBILE_DASHBOARD_PATH.exists():
        return HTMLResponse(
            content="<h1>Mobile UI not found</h1>"
                    f"<p>Expected at: {MOBILE_DASHBOARD_PATH}</p>",
            status_code=404,
        )
    return HTMLResponse(content=MOBILE_DASHBOARD_PATH.read_text(encoding="utf-8"))


# ──────────────── POST /scan/upload ── Phone Image Upload ────────────────

@app.post("/scan/upload")
async def scan_uploaded_image(
    image: UploadFile = File(...),
    request: Request = None,
):
    """
    Scan a card image uploaded from a phone camera.

    Accepts a multipart file upload (JPEG/PNG from phone camera),
    saves it to a temp file, runs process_image(), and returns
    the same JSON structure as /scan.

    Test Mode (Phase 1 hash testing):
        Add ?test_mode=true&condition=bare to auto-save the uploaded
        image to data/test_phone/capture_NNN.jpg and log scan results
        to manifest.json. The scan still runs normally — this just
        saves a side copy with metadata for later analysis by
        test_phone.py. Condition values: bare, sleeve, toploader.

    Usage from phone JS:
        const formData = new FormData();
        formData.append('image', blob, 'card.jpg');
        fetch('/scan/upload?use_cache=true&test_mode=true&condition=bare', {
            method: 'POST', body: formData
        });
    """
    import shutil
    import tempfile

    # ── Validate prerequisites ──
    if not scanner_resources["loaded"]:
        return JSONResponse(
            content={"error": "Scanner resources still loading — try again in a few seconds"},
            status_code=503,
        )

    # ── Validate file type ──
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        return JSONResponse(
            content={"error": f"Expected image file, got {content_type}"},
            status_code=400,
        )

    # ── Save uploaded file to temp location ──
    suffix = ".jpg" if "jpeg" in content_type or "jpg" in content_type else ".png"
    tmp = tempfile.NamedTemporaryFile(
        dir=str(Path(tempfile.gettempdir()) / "card-scanner-captures"),
        suffix=suffix,
        delete=False,
    )
    try:
        shutil.copyfileobj(image.file, tmp)
        tmp.close()
        img_path = tmp.name
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to save upload: {str(e)}"},
            status_code=500,
        )

    # ── Parse params ──
    test_mode = (request.query_params.get("test_mode", "").lower() == "true") if request else False
    test_condition = request.query_params.get("condition", "bare").strip() if request else "bare"
    test_saved_as = None

    use_cache_param = request.query_params.get("use_cache", "true") if request else "true"
    use_cache = use_cache_param.lower() != "false"

    # ── Run scanner pipeline ──
    # Card detection + perspective correction is handled inside process_image().
    # No need to pre-process here — just pass the raw upload through.
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            _run_scan,
            img_path,
            use_cache,
        )
    except Exception as e:
        logger.exception("Scan failed: %s", e)
        return JSONResponse(
            content={"error": f"Scan failed: {str(e)}"},
            status_code=500,
        )

    # ── Test capture: auto-save + manifest (Phase 1 hash testing) ──
    # Only saves when test_mode=true is passed from the mobile UI.
    # Saves the raw upload as capture_NNN.jpg and appends scan results
    # to manifest.json for later analysis by test_phone.py.
    if test_mode:
        try:
            test_dir = PROJECT_DIR / "data" / "test_phone"
            test_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = test_dir / "manifest.json"

            # Load existing manifest (or start fresh)
            manifest = []
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r") as mf:
                        manifest = json.load(mf)
                except (json.JSONDecodeError, Exception):
                    manifest = []

            # Sequential capture number
            seq = len(manifest) + 1
            test_filename = f"capture_{seq:03d}.jpg"
            test_path = test_dir / test_filename

            shutil.copy2(img_path, str(test_path))

            # Append entry to manifest
            manifest.append({
                "seq": seq,
                "file": test_filename,
                "condition": test_condition,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "scan_result": {
                    "card_id": result.get("card_id"),
                    "name": result.get("name"),
                    "set_name": result.get("set_name"),
                    "number": result.get("number"),
                    "id_method": result.get("id_method"),
                    "hash_distance": result.get("hash_distance"),
                    "name_conf": result.get("name_conf"),
                    "num_conf": result.get("num_conf"),
                    "price": result.get("price"),
                    "stamp_1st": result.get("stamp_1st"),
                    "time": result.get("time"),
                },
            })

            with open(manifest_path, "w") as mf:
                json.dump(manifest, mf, indent=2)

            test_saved_as = test_filename
            logger.info("Test capture saved: %s (seq %d, condition=%s)", test_filename, seq, test_condition)
        except Exception as e:
            logger.warning("Failed to save test capture: %s", e)

    # Clean up temp file
    try:
        Path(img_path).unlink(missing_ok=True)
    except Exception:
        pass

    # ── Build response ──
    cache_stats = get_cache().stats()

    response_data = {
        "result": result,
        "cache": cache_stats,
        "scan_time": result.get("time", 0),
    }

    # Include test capture info in response so the UI can confirm saves
    if test_saved_as:
        response_data["test_capture"] = test_saved_as

    return JSONResponse(content=response_data)


# ──────────────── GET /test/captures ── Test Capture Status ────────────────

@app.get("/test/captures")
async def test_capture_status():
    """
    List all saved test captures from manifest.json in data/test_phone/.
    Useful for checking progress without SSH-ing into the Jetson.
    """
    test_dir = PROJECT_DIR / "data" / "test_phone"
    manifest_path = test_dir / "manifest.json"

    if not manifest_path.exists():
        return JSONResponse(content={"captures": [], "count": 0})

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, Exception):
        return JSONResponse(content={"captures": [], "count": 0, "error": "manifest parse error"})

    # Summary by condition
    conditions = sorted(set(e.get("condition", "unknown") for e in manifest))
    by_condition = {
        cond: len([e for e in manifest if e.get("condition") == cond])
        for cond in conditions
    }

    # Summary by identified card_id
    card_ids = sorted(set(
        e.get("scan_result", {}).get("card_id", "unknown") or "unidentified"
        for e in manifest
    ))

    return JSONResponse(content={
        "captures": manifest,
        "count": len(manifest),
        "by_condition": by_condition,
        "unique_cards_identified": len(card_ids),
    })


# ──────────────── GET /api/status ── Server Status ────────────────

@app.get("/api/status")
async def api_status():
    """
    Return server status for the dashboard header indicators.

    Response includes camera state, cache stats, scanner readiness,
    and server uptime.
    """
    return JSONResponse(content={
        "scanner_ready": scanner_resources["loaded"],
        "camera": cam.status() if cam else {"mode": "none", "running": False},
        "cache": get_cache().stats(),
        "uptime_seconds": round(time.time() - start_time, 1),
    })


# ──────────────── POST /api/cache ── Cache Management ────────────────

@app.post("/api/cache")
async def cache_management(request: Request):
    """
    Cache management endpoint.

    Request body (JSON):
        { "action": "clear" }   → clear all cached prices
        { "action": "stats" }   → return cache statistics
        { "action": "purge" }   → remove expired entries only

    Response:
        { "action": "...", "result": ..., "cache": { stats } }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    action = body.get("action", "stats")
    cache = get_cache()

    if action == "clear":
        count = cache.clear()
        result = {"cleared": count}
    elif action == "purge":
        count = cache.purge_expired()
        result = {"purged": count}
    elif action == "stats":
        result = {}
    else:
        return JSONResponse(
            content={"error": f"Unknown action: {action}"},
            status_code=400,
        )

    return JSONResponse(content={
        "action": action,
        "result": result,
        "cache": cache.stats(),
    })


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Card Scanner — FastAPI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                              # Mac dev (mock camera, localhost:8080)
  python server.py --host 0.0.0.0               # Jetson (CSI camera, LAN-accessible)
  python server.py --camera-mode mock --mock-dir test_images/
  python server.py --port 9000 --log-level debug
        """,
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1, use 0.0.0.0 for LAN access)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port number (default: 8080)",
    )
    parser.add_argument(
        "--camera-mode", choices=["csi", "mock"], default=None,
        help="Force camera mode (default: auto-detect based on platform)",
    )
    parser.add_argument(
        "--mock-dir", default=None,
        help="Directory of images for mock camera mode",
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--ssl-certfile", default=None,
        help="Path to SSL certificate file (enables HTTPS)",
    )
    parser.add_argument(
        "--ssl-keyfile", default=None,
        help="Path to SSL private key file (requires --ssl-certfile)",
    )
    return parser.parse_args()


def main():
    """Entry point — parse args and start the uvicorn server."""
    global cli_args

    args = parse_args()
    cli_args = args

    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    logger.info("Starting Card Scanner server...")
    logger.info("  Host: %s", args.host)
    logger.info("  Port: %d", args.port)
    logger.info("  Camera mode: %s", args.camera_mode or "auto-detect")
    if args.mock_dir:
        logger.info("  Mock dir: %s", args.mock_dir)
    if args.ssl_certfile:
        logger.info("  SSL: ENABLED (HTTPS)")

    uvicorn_kwargs = {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
    }
    if args.ssl_certfile and args.ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile

    uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
