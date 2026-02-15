#!/usr/bin/env python3
"""
enable_ocr_debug.py — Toggle debug image saving for OCR number extraction.

Usage:
    python3 enable_ocr_debug.py on    # Enable: saves to debug_ocr/
    python3 enable_ocr_debug.py off   # Disable
    python3 enable_ocr_debug.py clean  # Delete debug images
"""

import sys
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.py"
DEBUG_DIR = Path(__file__).parent / "debug_ocr"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 enable_ocr_debug.py [on|off|clean]")
        sys.exit(1)

    action = sys.argv[1].lower()
    content = CONFIG_FILE.read_text()

    if action == "on":
        content = content.replace(
            'OCR_DEBUG_DIR = None',
            'OCR_DEBUG_DIR = Path("debug_ocr")'
        )
        CONFIG_FILE.write_text(content)
        DEBUG_DIR.mkdir(exist_ok=True)
        print(f"Debug ON — images will save to {DEBUG_DIR}/")

    elif action == "off":
        content = content.replace(
            'OCR_DEBUG_DIR = Path("debug_ocr")',
            'OCR_DEBUG_DIR = None'
        )
        CONFIG_FILE.write_text(content)
        print("Debug OFF")

    elif action == "clean":
        if DEBUG_DIR.exists():
            import shutil
            count = len(list(DEBUG_DIR.glob("*.png")))
            shutil.rmtree(DEBUG_DIR)
            print(f"Deleted {count} debug images from {DEBUG_DIR}/")
        else:
            print("No debug directory found")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
