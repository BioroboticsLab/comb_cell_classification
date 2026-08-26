"""Filename parsing and discovery for the annotation tool."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# regular expression for timestamp in original file namings
TIMESTAMP_RE = re.compile(r"(?P<ts>\d{8}T\d{6})(?:\.(?P<frac>[0-9.]+?))?Z")
TIMESTAMP_FALLBACK_RE = re.compile(r"(?P<ts>\d{8}T\d{6})")

# force immutability with frozen=True
@dataclass(frozen=True)
class ImageEntry:
    """A single parsed image filename."""

    path: Path
    timestamp: str
    fraction: str

    @property
    def sort_key(self) -> tuple[str, float]:
        """Sort key: time string, then numeric fraction in case of tie."""
        return (self.timestamp, float(self.fraction))

def parse_timestamp(name: str) -> Optional[tuple[str, str]]:
    """Extract ``(timestamp, fraction)`` from a filename, or ``None`` if absent."""
    match = TIMESTAMP_RE.search(name)
    if match:
        return match.group("ts"), (match.group("frac") or "0")
    match = TIMESTAMP_FALLBACK_RE.search(name)
    if match:
        return match.group("ts"), "0"
    return None

def parse_name(path: Union[str, Path]) -> Optional[ImageEntry]:
    """Parse a single image path into an :class:`ImageEntry`."""
    path = Path(path)
    ts = parse_timestamp(path.name)
    if ts is None:
        logger.warning("Skipping file with no parseable timestamp: %s", path.name)
        return None
    return ImageEntry(path=path, timestamp=ts[0], fraction=ts[1])

def find_images_and_sort(folder: Union[str, Path]) -> list[ImageEntry]:
    """Find all timestamped ``.png`` files in ``folder`` and sort them by time."""
    entries = [entry for path in sorted(Path(folder).glob("*.png")) if (entry := parse_name(path)) is not None]
    entries.sort(key=lambda e: e.sort_key)
    return entries
