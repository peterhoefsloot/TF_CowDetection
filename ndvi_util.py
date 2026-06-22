"""Shared NDVI computation so prepare_data.py and predict.py scale it identically.

Band stacking order is blue, green, red, nir -> indices 0,1,2,3.
NDVI = (nir - red) / (nir + red), in [-1, 1], stored as uint16 via
((ndvi + 1) / 2) * 65535 so it lives in the same native-uint16 stack as the
raw bands (per-band mean/std normalization in train.py rescales it anyway).
nodata pixels (nir+red == 0) get ndvi 0 -> 32767.
"""

import numpy as np

RED_IDX = 2
NIR_IDX = 3


def fill_ndvi_channel(img, ndvi_idx, block: int = 2048) -> None:
    """Fill img[:, :, ndvi_idx] in place with the uint16-scaled NDVI of the
    red/nir channels. Processed in row blocks to bound peak RAM on the full
    scene. The red/nir channels (RED_IDX, NIR_IDX) must already be populated."""
    h = img.shape[0]
    for r0 in range(0, h, block):
        r1 = min(h, r0 + block)
        red = img[r0:r1, :, RED_IDX].astype(np.float32)
        nir = img[r0:r1, :, NIR_IDX].astype(np.float32)
        denom = nir + red
        ndvi = np.zeros_like(red)
        np.divide(nir - red, denom, out=ndvi, where=denom > 0)
        img[r0:r1, :, ndvi_idx] = np.clip((ndvi + 1.0) * 0.5 * 65535.0,
                                          0, 65535).astype(np.uint16)
