"""
pricing_cache.py — TTL-based in-memory pricing cache for JustTCG API

Sits between scanner.py and pricing_justtcg.py to minimize API calls
against the JustTCG free tier limits (1,000/month, 100/day, 10/min).

Two pricing modes (toggleable from the dashboard UI):
  1. BATCH (default) — Queues up to 20 cards per request, uses a single
     API call. Best for scanning sessions where you're running through
     a stack of cards. Results are cached with configurable TTL.
  2. SINGLE — Fetches fresh pricing for one card immediately, bypassing
     the cache. Use when you need a real-time spot price on a single
     card (e.g., negotiating a buy at the counter).

Cache key: (card_id, is_1st_edition, rarity, condition)
Default TTL: 4 hours
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

DEFAULT_TTL_SECONDS = 4 * 60 * 60  # 4 hours
MAX_BATCH_SIZE = 20                 # JustTCG max cards per request


# ─────────────────────────────────────────────────────────────
# CACHE ENTRY
# ─────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """Single cached pricing result with expiry timestamp."""
    data: dict                  # {"price": "$XX.XX", "variant": "...", "source": "JustTCG"}
    expires_at: float           # time.time() when this entry becomes stale
    created_at: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────
# PRICING CACHE
# ─────────────────────────────────────────────────────────────

class PricingCache:
    """
    In-memory TTL cache for JustTCG pricing results.

    Usage:
        cache = PricingCache(ttl_seconds=14400)

        # Check cache first
        result = cache.get(card_id, is_1st_edition, rarity)
        if result is None:
            result = fetch_live_pricing(card_id, is_1st_edition, rarity)
            cache.put(card_id, is_1st_edition, rarity, result)

        # Stats
        print(cache.stats())
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(card_id: str, is_1st_edition: bool = False,
                  rarity: str = "Common", condition: str = "nm") -> tuple:
        """Build a normalized cache key."""
        return (
            card_id.strip().lower(),
            bool(is_1st_edition),
            rarity.strip().lower(),
            condition.strip().lower(),
        )

    def get(self, card_id: str, is_1st_edition: bool = False,
            rarity: str = "Common", condition: str = "nm") -> Optional[dict]:
        """
        Look up a cached pricing result.

        Returns:
            dict with pricing data if cache hit and not expired, else None.
        """
        key = self._make_key(card_id, is_1st_edition, rarity, condition)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check TTL expiry
        if time.time() > entry.expires_at:
            del self._cache[key]
            self._misses += 1
            logger.debug("Cache expired: %s", key)
            return None

        self._hits += 1
        logger.debug("Cache hit: %s", key)
        return entry.data

    def put(self, card_id: str, is_1st_edition: bool = False,
            rarity: str = "Common", condition: str = "nm",
            data: Optional[dict] = None) -> None:
        """
        Store a pricing result in the cache.

        Args:
            card_id: pokemontcg.io card ID (e.g. "base1-4")
            is_1st_edition: Whether this is a 1st Edition card
            rarity: Card rarity string from card_index.json
            condition: Price condition (nm, lp, mp, hp, dmg)
            data: Pricing result dict from fetch_live_pricing()
        """
        if data is None:
            return

        key = self._make_key(card_id, is_1st_edition, rarity, condition)
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=time.time() + self.ttl_seconds,
        )
        logger.debug("Cache store: %s (TTL: %ds)", key, self.ttl_seconds)

    def invalidate(self, card_id: str, is_1st_edition: bool = False,
                   rarity: str = "Common", condition: str = "nm") -> bool:
        """Remove a specific entry from the cache. Returns True if removed."""
        key = self._make_key(card_id, is_1st_edition, rarity, condition)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """Clear all cached entries. Returns the number of entries removed."""
        count = len(self._cache)
        self._cache.clear()
        logger.info("Cache cleared: %d entries removed", count)
        return count

    def purge_expired(self) -> int:
        """Remove all expired entries. Returns count of purged entries."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now > v.expires_at]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug("Purged %d expired entries", len(expired_keys))
        return len(expired_keys)

    def stats(self) -> dict:
        """
        Return cache statistics for display in the dashboard/logs.

        Returns dict with:
            size: Current number of cached entries
            hits: Total cache hits since init
            misses: Total cache misses since init
            hit_rate: Hit percentage (0-100), or None if no lookups yet
            ttl_seconds: Configured TTL
        """
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round((self._hits / total) * 100, 1) if total > 0 else None,
            "ttl_seconds": self.ttl_seconds,
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters without clearing cached data."""
        self._hits = 0
        self._misses = 0


# ─────────────────────────────────────────────────────────────
# CACHED PRICING WRAPPER
# ─────────────────────────────────────────────────────────────

# Module-level cache instance (shared across scanner runs)
_cache = PricingCache()


def get_cache() -> PricingCache:
    """Return the module-level cache instance."""
    return _cache


def fetch_cached_pricing(card_id: str, is_1st_edition: bool = False,
                         rarity: str = "Common",
                         use_cache: bool = True) -> dict:
    """
    Wrapper around pricing_justtcg.fetch_live_pricing() with caching.

    This is the primary entry point for scanner.py to call instead of
    calling fetch_live_pricing() directly.

    Args:
        card_id: pokemontcg.io card ID (e.g. "base1-4")
        is_1st_edition: Whether to fetch 1st Edition pricing
        rarity: Card rarity string
        use_cache: If True (batch mode), check/update cache.
                   If False (single mode), bypass cache entirely and
                   fetch a fresh price from JustTCG.

    Returns:
        dict: {"price": "$XX.XX", "variant": "...", "source": "JustTCG",
               "cached": True/False}
    """
    from pricing_justtcg import fetch_live_pricing

    # ── SINGLE MODE: bypass cache, get fresh price ──
    if not use_cache:
        logger.info("Single mode: fresh fetch for %s (1st=%s)", card_id, is_1st_edition)
        result = fetch_live_pricing(card_id, is_1st_edition, rarity)
        result["cached"] = False
        return result

    # ── BATCH MODE: check cache first ──
    cached = _cache.get(card_id, is_1st_edition, rarity)
    if cached is not None:
        cached["cached"] = True
        return cached

    # Cache miss — fetch from API
    result = fetch_live_pricing(card_id, is_1st_edition, rarity)
    _cache.put(card_id, is_1st_edition, rarity, data=result)
    result["cached"] = False
    return result
