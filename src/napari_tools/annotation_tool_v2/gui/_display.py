"""Display layer for :class:`annotation_tool_v2.app.AnnotationApp` (a mixin)."""

from __future__ import annotations

import dask.array as da
import numpy as np
import cv2
from dask import delayed

WARP_CACHE_MAX = 64  # max cached warped frames

class DisplayMixin:
    """napari layer management: the lazily-warped frame stack."""

    def _display_factor(self) -> int:
        """The active decimation factor (driven by the JPEG-proxy scale)."""
        return max(1, int(getattr(self.store, "proxy_scale", 1)))

    def _display_shape(self) -> tuple[int, int]:
        """The (H, W) shape of a displayed (possibly proxied) frame."""
        return self.store.display_shape

    def _effective_pins(self, idx: int) -> list[dict]:
        """Frame ``idx``'s corner pins, or ``[]`` when warping is toggled off."""
        if not getattr(self, "warp_enabled", True):
            return []
        return self.pins[idx] if idx < len(self.pins) else []

    def _warp_perspective(self, img, warp_3x3, out_size):
        """Applies warp using a 3x3 inverse-map homography. ``out_size`` is (width, height)."""
        return cv2.warpPerspective(src=img, M=warp_3x3, dsize=out_size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    def _warped_frame(self, idx: int) -> np.ndarray:
        """Frame ``idx`` with each corner pin applied."""
        def _scale_m(scale_factor: float, inverse: bool = False):
            """Scale matrix and inverse downscale matrix for the homography matrix."""
            if scale_factor == 0:
                raise ValueError("Downscale factor cannot be zero.")
            f = scale_factor if inverse else 1.0 / scale_factor
            return np.array(
                [[f, 0, 0],
                 [0, f, 0],
                 [0, 0, 1]], dtype=np.float32)

        cached = self._warp_cache.get(idx)
        if cached is not None:
            return cached

        image_source = self.store.get(idx)
        pins = self._effective_pins(idx)
        if not pins:
            return image_source

        scale_factor = self._display_factor()
        image_height, image_width = image_source.shape[:2]

        image_out = image_source.copy()
        
        for pin in pins:
            crop_x, crop_y, crop_width, crop_height = pin["crop"]
            homography_matrix = np.asarray(pin["H"], dtype=np.float32)
            
            # when downscale factor is used, apply to homography matrix
            if scale_factor > 1:
                crop_x, crop_y, crop_width, crop_height = crop_x // scale_factor, crop_y // scale_factor, crop_width // scale_factor, crop_height // scale_factor
                homography_matrix = (_scale_m(scale_factor) @ homography_matrix @ _scale_m(scale_factor, inverse=True)).astype(np.float32)
                
            pin_x0, pin_y0 = max(0, crop_x), max(0, crop_y)
            pin_x1, pin_y1 = min(crop_x + crop_width, image_width), min(crop_y + crop_height, image_height)

            if pin_x1 <= pin_x0 or pin_y1 <= pin_y0:
                continue

            source_image_cropped = image_source[pin_y0:pin_y1, pin_x0:pin_x1]
            if source_image_cropped.size == 0:
                continue

            # shifts the crop-local homography to the clamped pin rect
            translation_matrix = np.array(
                [[1, 0, pin_x0 - crop_x],
                 [0, 1, pin_y0 - crop_y],
                 [0, 0, 1               ]], dtype=np.float32)

            combined_matrix = (homography_matrix @ translation_matrix).astype(np.float32)

            image_out[pin_y0:pin_y1, pin_x0:pin_x1] = self._warp_perspective(source_image_cropped, combined_matrix, (pin_x1 - pin_x0, pin_y1 - pin_y0))

        self._warp_cache[idx] = image_out
        # ponytail: FIFO cap; switch to true LRU if scrubbing patterns thrash it
        while len(self._warp_cache) > WARP_CACHE_MAX:
            self._warp_cache.pop(next(iter(self._warp_cache)))
        return image_out

    def _moving_dask(self) -> da.Array:
        """(N, H, W) dask stack of warped frames -> only sliced frames load."""
        h, w = self._display_shape()
        return da.stack(
            [
                da.from_delayed(delayed(self._warped_frame)(i), shape=(h, w), dtype=np.float32)
                for i in range(len(self.entries))
            ]
        )

    def _current_index(self) -> int:
        """Index of the frame currently shown by the dimension slider, clamped to the loaded frames."""
        step = self.viewer.dims.current_step
        idx = int(step[0]) if step else 0
        # Slider keeps its position when a folder or camera with fewer frames is loaded
        # napari only re-ranges dims once new image layer exists — until then the raw value points past the last frame.
        if not self.entries:
            return 0
        return max(0, min(idx, len(self.entries) - 1))

    def _set_current_index(self, idx: int) -> None:
        """Move the dimension slider to frame ``idx`` (clamped)."""
        if not self.entries:
            return
        idx = max(0, min(len(self.entries) - 1, idx))
        step = list(self.viewer.dims.current_step)
        if step:
            step[0] = idx
            self.viewer.dims.current_step = tuple(step)

    def _rebuild_view(self) -> None:
        """Drop the image layer and rebuild it as a fresh lazy stack of warped frames."""
        self._warp_cache.clear()  # pins / warp toggle / display scale may have changed
        idx = self._current_index()
        self._remove_layer("image_layer")
        if not self.entries:
            return
        current = self._warped_frame(idx)
        lo, hi = float(current.min()), float(current.max())
        if hi <= lo:
            hi = lo + 1.0
        f = self._display_factor()
        self.image_layer = self.viewer.add_image(
            self._moving_dask(),
            name="frames",
            colormap="gray",
            contrast_limits=[lo, hi],
            scale=(1, f, f),
        )
        self._fix_overlay_layer_order()
        self._set_current_index(idx)

    def _fix_overlay_layer_order(self) -> None:
        """Keep the overlays stacked images < grid < points < brush (top)."""
        # move(src, dest) inserts BEFORE dest (napari decrements dest when dest > src), so moving to the top needs dest = len, not len - 1.
        for attr in ("grid_layer", "ann_layer", "brush_layer"):
            layer = getattr(self, attr, None)
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.move(self.viewer.layers.index(layer), len(self.viewer.layers))

    def _remove_layer(self, attr: str) -> None:
        """Remove a managed layer from the viewer and clear its attribute."""
        layer = getattr(self, attr, None)
        if layer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)
        setattr(self, attr, None)

