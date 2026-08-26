"""Lazy, windowed loading of the image frames."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .parsing import ImageEntry

def load_gray_scale_image(path: Path) -> np.ndarray:
    """Load an image as a grayscale ``float32`` array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Failed to read image: {path}")
    return img.astype(np.float32)
class FrameStore:
    """On-demand frame loader with a bounded, windowed cache."""
    def __init__(
        self,
        entries: Iterable[ImageEntry],
        radius: int = 1,
        proxy_scale: int = 1,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._entries: list[ImageEntry] = list(entries)
        self._radius = radius
        self._cache: dict[int, np.ndarray] = {}
        self._shape: Optional[tuple[int, int]] = None
        self.proxy_scale = max(1, int(proxy_scale))
        self._cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "annotation_tool_v2_proxies"
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, idx: int) -> np.ndarray:
        """Return frame ``idx``, loading and caching it if needed."""
        with self._lock:
            cached = self._cache.get(idx)
            if cached is not None:
                return cached
        img = self.read(idx)
        with self._lock:
            self._cache[idx] = img
        return img

    def read(self, idx: int) -> np.ndarray:
        """Load frame ``idx`` from disk without touching the cache."""
        return load_gray_scale_image(self._proxy_path(idx))

    def _proxy_path(self, idx: int) -> Path:
        """Path to frame ``idx``'s JPEG proxy, generating it if missing/stale."""
        src = Path(self._entries[idx].path)
        f = self.proxy_scale
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.md5(f"{src}|{f}".encode()).hexdigest()
        out = self._cache_dir / f"{key}.jpg"
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            if self._shape is None:
                full = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if full is not None:
                    self._shape = full.shape[:2]
            return out
        full = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if full is None:
            raise OSError(f"Failed to read image: {src}")
        self._shape = full.shape[:2]
        h, w = full.shape[:2]
        if f == 1:
            small, quality = full, 95
        else:
            small = cv2.resize(
                full, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA
            )
            quality = 85
        tmp = self._cache_dir / f"{key}.{threading.get_ident()}.tmp.jpg"
        cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        os.replace(tmp, out)
        return out

    @property
    def original_shape(self) -> tuple[int, int]:
        """The full-res original (H, W) of the frames (reads the first frame once if needed)."""
        if self._shape is None and self._entries:
            self.read(0)
        return self._shape or (0, 0)

    @property
    def display_shape(self) -> tuple[int, int]:
        """The (H, W) of what :meth:`get` returns (proxied when ``proxy_scale > 1``)."""
        h, w = self.original_shape
        f = self.proxy_scale
        return (h, w) if f <= 1 else (h // f, w // f)

    def prefetch(self, center: int, keep: Iterable[int] = ()) -> None:
        """Cache the window around ``center`` (plus ``keep``) and evict the rest."""
        n = len(self._entries)
        wanted = set(range(max(0, center - self._radius), min(n, center + self._radius + 1)))
        wanted |= {k for k in keep if 0 <= k < n}
        for k in wanted:
            self.get(k)
        with self._lock:
            for k in [k for k in self._cache if k not in wanted]:
                del self._cache[k]
