"""
check_quota.py
Quick check of JustTCG API usage and remaining quota.
Makes a single lightweight API call and reads the usage metadata.

Usage:
    cd ~/card-scanner && source venv/bin/activate
    python3 check_quota.py
"""

import json
import os
import sys
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

import requests

JUSTTCG_API_KEY = os.getenv("JUSTTCG_API_KEY", "")
if not JUSTTCG_API_KEY:
    print("ERROR: JUSTTCG_API_KEY not found in .env")
    sys.exit(1)

# Make a minimal API call (Pikachu base1-58 — cheap, always exists)
r = requests.get(
    "https://api.justtcg.com/v1/cards",
    headers={"x-api-key": JUSTTCG_API_KEY},
    params={
        "tcgplayerId": "86581",
        "game": "pokemon",
        "include_price_history": "false",
        "include_statistics": "",
    },
    timeout=10,
)

if r.status_code == 200:
    data = r.json()
    meta = data.get("meta", {})
    usage = meta.get("usage", meta)

    print("=" * 45)
    print("  JustTCG API Usage")
    print("=" * 45)
    print(f"  Plan:             {usage.get('plan', '?')}")
    print(f"  Monthly limit:    {usage.get('totalLimit', '?')}")
    print(f"  Monthly used:     {usage.get('totalUsed', '?')}")
    print(f"  Monthly remaining:{usage.get('totalRemaining', '?')}")
    print("-" * 45)
    print(f"  Daily limit:      {usage.get('dailyLimit', '?')}")
    print(f"  Daily used:       {usage.get('dailyUsed', '?')}")
    print(f"  Daily remaining:  {usage.get('dailyRemaining', '?')}")
    print("-" * 45)
    print(f"  Per-minute limit: {usage.get('minuteLimit', '?')}")
    print(f"  Error:            {usage.get('error', 'None')}")
    print("=" * 45)

    # Dump raw meta for debugging if fields are different
    if not usage.get("plan"):
        print("\nRaw meta/usage response:")
        print(json.dumps(meta, indent=2))
elif r.status_code == 429:
    print("Rate limited — you've hit a quota ceiling.")
else:
    print(f"API returned status {r.status_code}")
    print(r.text[:500])
