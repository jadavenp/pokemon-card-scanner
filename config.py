"""
config.py
Central configuration for the card scanner.
API keys loaded from .env file (not committed to git).
"""

import os
from pathlib import Path

# ============================================
# Paths
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATABASE_FILE = DATA_DIR / "card_index.json"
IMAGE_DIR = DATA_DIR
TEMPLATE_PATH = DATA_DIR / "stamp_template.png"

# ============================================
# .env loader (no external dependency)
# ============================================
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# ============================================
# API Configuration
# ============================================
# pokemontcg.io (build-time only — not used at runtime)
POKEMONTCG_API_KEY = os.environ.get("POKEMONTCG_API_KEY", "")
POKEMONTCG_BASE = "https://api.pokemontcg.io/v2"

# ============================================
# Stamp Detection Thresholds
# ============================================
STAMP_THRESHOLD = 0.65          # Score above this = 1st Edition
STAMP_CONTRAST_MIN = 20         # Patch stddev below this = holo noise penalty
STAMP_CONTRAST_PENALTY = 0.5    # Multiplier applied to low-contrast matches
STAMP_SCALE_MIN = 0.05
STAMP_SCALE_MAX = 0.6
STAMP_SCALE_STEPS = 25

# Bilateral filter params for holo suppression
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ============================================
# OCR Configuration
# ============================================
# Words to filter out during name extraction.
# These appear on cards but are never the card name.
NAME_SKIP_WORDS = [
    # Card structure
    'basic', 'pokemon', 'pokémon', 'stage', 'hp', 'trainer', 'energy',
    'lv', 'lv.', 'evolves', 'from', 'put',
    # Copyright / footer
    'illus', 'illus.', 'wizards', 'nintendo', 'creatures', 'gamefreak',
    '1995', '1996', '1998', '1999', '2000', '2001', '2002',
    # Common body text
    'weakness', 'resistance', 'retreat', 'cost', 'length', 'weight',
    'power', 'damage', 'coin', 'flip', 'attach', 'shuffle',
    'your', 'you', 'each', 'the', 'this', 'does', 'cards',
    # Add common attack/ability words that appear on Pokemon cards
    # ONLY full-word matches that are never a Pokemon card name.
    # Avoid substrings that could match names (e.g. "slash" is in "Sandslash")
    'tackle', 'scratch', 'leer', 'growl', 'roar',
    'selfdestruct', 'self-destruct', 'metronome',
    'withdraw', 'harden', 'recover', 'teleport',
    'rain dance', 'solar beam', 'fire spin', 'ice beam',
    'bubble beam', 'hyper beam', 'dream eater', 'night shade',
    'boulder crush', 'fury swipes', 'vine whip', 'razor leaf',
    'string shot', 'poison sting', 'thunder shock', 'thundershock',
    'body slam', 'double kick', 'mega punch', 'mega kick',
    'fire punch', 'ice punch', 'thunder punch',
    'karate chop', 'low kick', 'seismic toss',
    'does damage', 'flip a coin', 'coin flip',
    'attached energy', 'benched pokemon', 'active pokemon',
    'defending pokemon', 'this attack', 'next turn',
]

# Name extraction scoring weights
NAME_POSITION_WEIGHT = 0.3
NAME_CONFIDENCE_WEIGHT = 0.3
NAME_SIZE_WEIGHT = 0.0004
NAME_MIN_CONFIDENCE = 0.3
NAME_MAX_Y_RATIO = 0.40    # Name must be in top 40% of card

# Number extraction
NUMBER_CROP_Y = 0.80       # Bottom 20% of image
NUMBER_CROP_X = 0.45       # Right 55% of image
NUMBER_UPSCALE_LARGE = 2   # Scale factor for images >= 800px wide
NUMBER_UPSCALE_SMALL = 3   # Scale factor for images < 800px wide
NUMBER_EARLY_EXIT_CONF = 0.80

# ============================================
# Set Names
# ============================================
SET_NAMES = {
    "base1": "Base Set", "base2": "Jungle", "base3": "Fossil",
    "base4": "Base Set 2", "base5": "Team Rocket", "base6": "Legendary Collection",
    "gym1": "Gym Heroes", "gym2": "Gym Challenge",
    "neo1": "Neo Genesis", "neo2": "Neo Discovery",
    "neo3": "Neo Revelation", "neo4": "Neo Destiny",
    "si1": "Southern Islands",
    "ecard1": "Expedition Base Set", "ecard2": "Aquapolis", "ecard3": "Skyridge",
    "ex1": "Ruby & Sapphire", "ex2": "Sandstorm", "ex3": "Dragon",
    "ex4": "Team Magma vs Team Aqua", "ex5": "Hidden Legends",
    "ex6": "FireRed & LeafGreen", "ex7": "Team Rocket Returns",
    "ex8": "Deoxys", "ex9": "Emerald", "ex10": "Unseen Forces",
    "ex11": "Delta Species", "ex12": "Legend Maker", "ex13": "Holon Phantoms",
    "ex14": "Crystal Guardians", "ex15": "Dragon Frontiers", "ex16": "Power Keepers",
    "dp1": "Diamond & Pearl", "dp2": "Mysterious Treasures",
    "dp3": "Secret Wonders", "dp4": "Great Encounters",
    "dp5": "Majestic Dawn", "dp6": "Legends Awakened", "dp7": "Stormfront",
    "pl1": "Platinum", "pl2": "Rising Rivals",
    "pl3": "Supreme Victors", "pl4": "Arceus",
    "hgss1": "HeartGold & SoulSilver", "hgss2": "Unleashed",
    "hgss3": "Undaunted", "hgss4": "Triumphant",
    "bw1": "Black & White", "bw2": "Emerging Powers",
    "bw3": "Noble Victories", "bw4": "Next Destinies",
    "bw5": "Dark Explorers", "bw6": "Dragons Exalted",
    "bw7": "Boundaries Crossed", "bw8": "Plasma Storm",
    "bw9": "Plasma Freeze", "bw10": "Plasma Blast",
    "bw11": "Legendary Treasures",
    "xy0": "Kalos Starter Set", "xy1": "XY", "xy2": "Flashfire",
    "xy3": "Furious Fists", "xy4": "Phantom Forces",
    "xy5": "Primal Clash", "xy6": "Roaring Skies",
    "xy7": "Ancient Origins", "xy8": "BREAKthrough",
    "xy9": "BREAKpoint", "xy10": "Fates Collide",
    "xy11": "Steam Siege", "xy12": "Evolutions",
    "sm1": "Sun & Moon", "sm2": "Guardians Rising",
    "sm3": "Burning Shadows", "sm4": "Crimson Invasion",
    "sm5": "Ultra Prism", "sm6": "Forbidden Light",
    "sm7": "Celestial Storm", "sm8": "Lost Thunder",
    "sm9": "Team Up", "sm10": "Unbroken Bonds",
    "sm11": "Unified Minds", "sm12": "Cosmic Eclipse",
    "swsh1": "Sword & Shield", "swsh2": "Rebel Clash",
    "swsh3": "Darkness Ablaze", "swsh4": "Vivid Voltage",
    "swsh5": "Battle Styles", "swsh6": "Chilling Reign",
    "swsh7": "Evolving Skies", "swsh8": "Fusion Strike",
    "swsh9": "Brilliant Stars", "swsh10": "Astral Radiance",
    "swsh11": "Lost Origin", "swsh12": "Silver Tempest",
    "swsh12pt5": "Crown Zenith",
    "sv1": "Scarlet & Violet", "sv2": "Paldea Evolved",
    "sv3": "Obsidian Flames", "sv3pt5": "151",
    "sv4": "Paradox Rift", "sv4pt5": "Paldean Fates",
    "sv5": "Temporal Forces", "sv6": "Twilight Masquerade",
    "sv6pt5": "Shrouded Fable", "sv7": "Stellar Crown",
    "sv8": "Surging Sparks",
    "sv8pt5": "Prismatic Evolutions",
}

VINTAGE_SET_IDS = {
    "base1", "base2", "base3", "base4", "base5",
    "gym1", "gym2", "neo1", "neo2", "neo3", "neo4",
}

# ============================================
# Output variant label map
# ============================================
VARIANT_DISPLAY = {
    "1stEditionHolofoil": "1st Holo",
    "1stEditionNormal": "1st Norm",
    "holofoil": "Holo",
    "reverseHolofoil": "Rev Holo",
    "normal": "Normal",
    # JustTCG condition-based labels
    "1st_ed_nm": "1st NM",
    "1st_ed_lp": "1st LP",
    "1st_ed_mp": "1st MP",
    "1st_ed_hp": "1st HP",
    "1st_ed_dmg": "1st DMG",
    "unlimited_nm": "Unl NM",
    "unlimited_lp": "Unl LP",
    "unlimited_mp": "Unl MP",
    "unlimited_hp": "Unl HP",
    "unlimited_dmg": "Unl DMG",
    "unlimited*_nm": "Unl* NM",
    "unlimited*_lp": "Unl* LP",
    "unlimited*_mp": "Unl* MP",
    "unlimited*_hp": "Unl* HP",
    "unlimited*_dmg": "Unl* DMG",
    "holofoil_nm": "Holo NM",
}

# ============================================
# OCR v2 — Enhanced number extraction (Session 5)
# ============================================

# Number region crops by card type (tighter than v1)
# Format: {y_start, y_end, x_start, x_end} as fraction of card dimensions
NUMBER_REGIONS = {
    "pokemon": {"y_start": 0.95, "y_end": 1.00, "x_start": 0.00, "x_end": 1.00},
    "trainer": {"y_start": 0.95, "y_end": 1.00, "x_start": 0.00, "x_end": 1.00},
    "energy":  {"y_start": 0.95, "y_end": 1.00, "x_start": 0.00, "x_end": 1.00},
}

# Minimum card width after perspective correction (ensures number text is readable)
CARD_MIN_OUTPUT_WIDTH = 480

# CLAHE parameters for number strip
NUMBER_CLAHE_CLIP = 2.5
NUMBER_CLAHE_GRID = (4, 4)

# Highlight reduction — clamp V channel in HSV above this value
NUMBER_HIGHLIGHT_CLAMP = 200

# Unsharp mask parameters
NUMBER_UNSHARP_SIGMA = 2.0
NUMBER_UNSHARP_WEIGHT = 1.5   # original weight
NUMBER_UNSHARP_BLUR_WEIGHT = -0.5  # blur subtraction weight

# Bilateral filter for denoising
NUMBER_BILATERAL_D = 5
NUMBER_BILATERAL_SIGMA = 40

# EasyOCR parameters for number strip
NUMBER_ALLOWLIST = "0123456789/"
NUMBER_CONTRAST_THS = 0.05
NUMBER_ADJUST_CONTRAST = 0.7
NUMBER_MIN_SIZE = 5

# Upscale factor for number ROI (v2)
NUMBER_UPSCALE_V2 = 4

# Debug: save preprocessed number strip images
# Set to a directory path to enable, None to disable
OCR_DEBUG_DIR = Path("debug_ocr")  # e.g., Path("debug_ocr") to enable

# Confidence thresholds for identification
CONFIDENCE_NAME_WEIGHT = 0.40     # Weight of name OCR confidence in composite score
CONFIDENCE_NUMBER_WEIGHT = 0.35   # Weight of number OCR confidence
CONFIDENCE_HASH_WEIGHT = 0.20     # Weight of hash match confidence
CONFIDENCE_STAMP_WEIGHT = 0.05    # Weight of stamp detection (low — not an ID signal)

# Known set totals for validation (card_number / total)
# Maps total → list of possible set IDs
KNOWN_SET_TOTALS = {
    "102": ["base1"],
    "64": ["base2", "gym1", "gym2"],
    "62": ["base3"],
    "130": ["base4"],
    "83": ["base5"],
    "82": ["base6"],
    "111": ["neo1"],
    "75": ["neo2"],
    "66": ["neo3"],
    "113": ["neo4"],
    "165": ["ecard1"],
    "147": ["ecard2"],
    "182": ["ecard3"],
    "109": ["ex1"],
    "100": ["ex2"],
    "203": ["swsh1", "bw1"],
}
