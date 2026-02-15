"""
camera.py — Camera abstraction for the Pokemon card scanner

Provides a unified interface for two capture backends:
  1. CSI mode  — Arducam IMX708 via GStreamer/OpenCV on Jetson Orin Nano
  2. Mock mode — Cycles through static card images from data/Ref Images/
                 (or a user-specified directory) for Mac development

Both backends implement the same interface so server.py and the scanner
pipeline don't care which one is active. The mode is auto-detected based
on platform (aarch64 = Jetson, else = mock), or can be forced via the
constructor.

Usage:
    from camera import Camera

    cam = Camera()                          # auto-detect
    cam = Camera(mode="mock", mock_dir="data/Ref Images/base1")  # force mock
    cam = Camera(mode="csi")                # force CSI

    cam.start()
    frame = cam.read()                      # numpy array (BGR, OpenCV format)
    path  = cam.capture_to_file()           # saves frame, returns path
    jpg   = cam.read_jpeg()                 # JPEG bytes for MJPEG streaming
    cam.stop()

Frame contract:
    - read()            → numpy.ndarray (H, W, 3) BGR uint8, or None if unavailable
    - read_jpeg()       → bytes (JPEG-encoded), or None
    - capture_to_file() → str (path to saved .jpg), used by process_image()
"""

import os
import glob
import time
import logging
import platform
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# CSI camera defaults (Arducam IMX708 on Jetson Orin Nano)
CSI_WIDTH = 1920
CSI_HEIGHT = 1080
CSI_FPS = 30
CSI_FLIP = 0           # 0 = no rotation, 2 = 180°; adjust for mount orientation
CSI_SENSOR_ID = 0       # CAM0 connector on Jetson

# Mock mode defaults
MOCK_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")

# Temp directory for captured frames (used by capture_to_file)
CAPTURE_DIR = Path(tempfile.gettempdir()) / "card-scanner-captures"


# ─────────────────────────────────────────────────────────────
# GSTREAMER PIPELINE (Jetson CSI only)
# ─────────────────────────────────────────────────────────────

def _build_gstreamer_pipeline(
    width: int = CSI_WIDTH,
    height: int = CSI_HEIGHT,
    fps: int = CSI_FPS,
    flip: int = CSI_FLIP,
    sensor_id: int = CSI_SENSOR_ID,
) -> str:
    """
    Build a GStreamer pipeline string for nvarguscamerasrc (Jetson CSI).

    The pipeline captures from the CSI sensor, applies ISP processing
    via the Jetson's hardware, and outputs BGR frames to OpenCV via
    appsink.

    Args:
        width: Capture width in pixels
        height: Capture height in pixels
        fps: Target framerate
        flip: nvarguscamerasrc flip-method (0=none, 2=180°)
        sensor_id: CSI connector index (0 or 1)

    Returns:
        GStreamer pipeline string for cv2.VideoCapture()
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){fps}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1"
    )


# ─────────────────────────────────────────────────────────────
# CAMERA CLASS
# ─────────────────────────────────────────────────────────────

class Camera:
    """
    Unified camera interface for CSI capture (Jetson) and mock mode (Mac).

    Attributes:
        mode: "csi" or "mock"
        is_running: Whether the camera/mock feed is active
        resolution: Tuple (width, height) of the current frame size
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        mock_dir: Optional[str] = None,
        width: int = CSI_WIDTH,
        height: int = CSI_HEIGHT,
        fps: int = CSI_FPS,
        flip: int = CSI_FLIP,
        sensor_id: int = CSI_SENSOR_ID,
    ):
        """
        Initialize the camera.

        Args:
            mode: "csi" or "mock". If None, auto-detects based on platform
                  (aarch64 → csi, else → mock).
            mock_dir: Directory of images to cycle through in mock mode.
                      Defaults to data/Ref Images/ relative to project root.
            width: CSI capture width (ignored in mock mode)
            height: CSI capture height (ignored in mock mode)
            fps: CSI framerate (ignored in mock mode)
            flip: CSI flip-method (ignored in mock mode)
            sensor_id: CSI sensor index (ignored in mock mode)
        """
        # ── Auto-detect mode ──
        if mode is None:
            is_jetson = platform.machine() == "aarch64"
            self.mode = "csi" if is_jetson else "mock"
            logger.info("Camera mode auto-detected: %s (arch: %s)",
                        self.mode, platform.machine())
        else:
            self.mode = mode.lower()
            if self.mode not in ("csi", "mock"):
                raise ValueError(f"Invalid camera mode: '{mode}'. Use 'csi' or 'mock'.")

        # ── State ──
        self.is_running = False
        self.resolution = (width, height)
        self._cap: Optional[cv2.VideoCapture] = None

        # ── CSI config ──
        self._width = width
        self._height = height
        self._fps = fps
        self._flip = flip
        self._sensor_id = sensor_id

        # ── Mock config ──
        self._mock_images: list[str] = []
        self._mock_index = 0
        self._mock_dir = mock_dir

        # Ensure capture temp dir exists
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start the camera or load mock images.

        Returns:
            True if started successfully, False otherwise.
        """
        if self.is_running:
            logger.warning("Camera already running")
            return True

        if self.mode == "csi":
            return self._start_csi()
        else:
            return self._start_mock()

    def stop(self) -> None:
        """Release the camera or clear mock state."""
        if self.mode == "csi" and self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("CSI camera released")
        elif self.mode == "mock":
            self._mock_images = []
            self._mock_index = 0
            logger.info("Mock camera stopped (%d images unloaded)", len(self._mock_images))

        self.is_running = False

    def _start_csi(self) -> bool:
        """Initialize the GStreamer CSI pipeline via OpenCV."""
        pipeline = _build_gstreamer_pipeline(
            width=self._width,
            height=self._height,
            fps=self._fps,
            flip=self._flip,
            sensor_id=self._sensor_id,
        )
        logger.info("Opening CSI camera: %s", pipeline)

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            logger.error("Failed to open CSI camera. Check ribbon cable and sensor_id.")
            self._cap = None
            return False

        self.is_running = True
        self.resolution = (self._width, self._height)
        logger.info("CSI camera started: %dx%d @ %dfps", self._width, self._height, self._fps)
        return True

    def _start_mock(self) -> bool:
        """Load mock images from the specified directory."""
        mock_dir = self._resolve_mock_dir()
        if mock_dir is None:
            return False

        # Gather all image files
        images = []
        for ext in MOCK_EXTENSIONS:
            images.extend(glob.glob(os.path.join(mock_dir, "**", ext), recursive=True))

        if not images:
            logger.error("No images found in mock directory: %s", mock_dir)
            return False

        self._mock_images = sorted(images)
        self._mock_index = 0
        self.is_running = True

        # Set resolution from first image
        first = cv2.imread(self._mock_images[0])
        if first is not None:
            h, w = first.shape[:2]
            self.resolution = (w, h)

        logger.info("Mock camera started: %d images from %s", len(self._mock_images), mock_dir)
        return True

    def _resolve_mock_dir(self) -> Optional[str]:
        """Resolve the mock image directory path."""
        if self._mock_dir:
            p = Path(self._mock_dir)
            if p.is_dir():
                return str(p)
            logger.error("Mock directory not found: %s", self._mock_dir)
            return None

        # Default: look for data/Ref Images/ relative to this file's location
        project_root = Path(__file__).parent
        candidates = [
            project_root / "data" / "Ref Images",
            project_root / "data" / "ref_images",
            project_root / "test_images",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)

        logger.error(
            "No mock directory found. Searched: %s. "
            "Pass mock_dir= to Camera() or create a test_images/ folder.",
            [str(c) for c in candidates],
        )
        return None

    # ─────────────────────────────────────────────────────────
    # FRAME CAPTURE
    # ─────────────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Read a single frame.

        Returns:
            numpy.ndarray (H, W, 3) BGR uint8, or None if unavailable.
            In mock mode, cycles through loaded images sequentially.
        """
        if not self.is_running:
            logger.warning("Camera not running — call start() first")
            return None

        if self.mode == "csi":
            return self._read_csi()
        else:
            return self._read_mock()

    def _read_csi(self) -> Optional[np.ndarray]:
        """Grab a frame from the CSI pipeline."""
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("CSI frame read failed")
            return None
        return frame

    def _read_mock(self) -> Optional[np.ndarray]:
        """Read the next mock image in sequence."""
        if not self._mock_images:
            return None

        path = self._mock_images[self._mock_index]
        frame = cv2.imread(path)

        if frame is None:
            logger.warning("Failed to read mock image: %s", path)
            # Skip to next
            self._mock_index = (self._mock_index + 1) % len(self._mock_images)
            return None

        return frame

    def advance(self) -> None:
        """
        Advance to the next mock image (mock mode only).

        In CSI mode this is a no-op — the camera feed is continuous.
        Call this after processing a mock frame so the next read()
        returns a different card image.
        """
        if self.mode == "mock" and self._mock_images:
            self._mock_index = (self._mock_index + 1) % len(self._mock_images)

    def read_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """
        Read a frame and return it as JPEG bytes.

        Used by server.py for the MJPEG stream at /video_feed.

        Args:
            quality: JPEG compression quality (0-100)

        Returns:
            bytes of the JPEG-encoded frame, or None.
        """
        frame = self.read()
        if frame is None:
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            logger.warning("JPEG encode failed")
            return None

        return buffer.tobytes()

    def capture_to_file(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Capture a frame and save it to disk.

        This is the bridge between the camera and process_image() in
        scanner.py, which expects a file path as input.

        Args:
            filename: Optional filename. If None, generates a timestamped
                      name like 'capture_1707912345.jpg'.

        Returns:
            Absolute path to the saved JPEG, or None on failure.
        """
        frame = self.read()
        if frame is None:
            return None

        if filename is None:
            filename = f"capture_{int(time.time())}.jpg"

        path = CAPTURE_DIR / filename
        success = cv2.imwrite(str(path), frame)

        if not success:
            logger.error("Failed to save capture: %s", path)
            return None

        logger.debug("Frame captured: %s", path)
        return str(path)

    # ─────────────────────────────────────────────────────────
    # STATUS / INFO
    # ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return camera status for the dashboard /api/status endpoint.

        Returns dict with:
            mode: "csi" or "mock"
            running: bool
            resolution: [width, height]
            mock_count: number of mock images (mock mode only)
            mock_current: filename of current mock image (mock mode only)
        """
        info = {
            "mode": self.mode,
            "running": self.is_running,
            "resolution": list(self.resolution),
        }

        if self.mode == "mock" and self._mock_images:
            info["mock_count"] = len(self._mock_images)
            info["mock_current"] = os.path.basename(
                self._mock_images[self._mock_index]
            )

        return info

    @property
    def current_mock_path(self) -> Optional[str]:
        """
        Return the file path of the current mock image.

        Useful for passing directly to process_image() in mock mode
        without needing to capture_to_file() first — avoids the extra
        JPEG re-encode when you already have the original PNG.

        Returns None in CSI mode or if no images are loaded.
        """
        if self.mode == "mock" and self._mock_images:
            return self._mock_images[self._mock_index]
        return None

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        if self.mode == "mock" and self._mock_images:
            return (f"Camera(mode=mock, {status}, "
                    f"{len(self._mock_images)} images, "
                    f"idx={self._mock_index})")
        return f"Camera(mode={self.mode}, {status}, {self.resolution[0]}x{self.resolution[1]})"
