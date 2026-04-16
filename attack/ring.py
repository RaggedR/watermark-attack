"""Concentric ring rotation attack.

Divides an image into concentric annular rings centred on the image
and rotates alternate rings in opposite directions.  This scrambles
spatial and frequency-domain watermark patterns without destroying
global image structure.

Ported from watermark-rs/ring_attack.py.
"""

import numpy as np
from PIL import Image
from scipy import ndimage


def _create_ring_mask(shape, center, inner_radius, outer_radius):
    """Boolean mask for pixels in [inner_radius, outer_radius) from center."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    cy, cx = center
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return (distance >= inner_radius) & (distance < outer_radius)


def _rotate_ring(image_array, mask, angle):
    """Rotate pixels within *mask* by *angle* degrees (in-place region)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    region = image_array[rmin : rmax + 1, cmin : cmax + 1].copy()
    region_mask = mask[rmin : rmax + 1, cmin : cmax + 1]

    result = image_array.copy()

    if len(region.shape) == 3:
        rotated = np.zeros_like(region)
        for c in range(region.shape[2]):
            rotated[:, :, c] = ndimage.rotate(
                region[:, :, c], angle, reshape=False, order=1, mode="nearest"
            )
        for c in range(image_array.shape[2]):
            result[rmin : rmax + 1, cmin : cmax + 1, c] = np.where(
                region_mask,
                rotated[:, :, c],
                result[rmin : rmax + 1, cmin : cmax + 1, c],
            )
    else:
        rotated = ndimage.rotate(region, angle, reshape=False, order=1, mode="nearest")
        result[rmin : rmax + 1, cmin : cmax + 1] = np.where(
            region_mask, rotated, result[rmin : rmax + 1, cmin : cmax + 1]
        )

    return result


def ring_rotation_attack(
    image: Image.Image,
    num_rings: int = 15,
    max_angle: float = 1.5,
) -> Image.Image:
    """Apply concentric ring rotation attack.

    Alternate rings rotate clockwise / counter-clockwise by max_angle.

    Args:
        image: Input PIL image.
        num_rings: Number of concentric rings.
        max_angle: Rotation magnitude in degrees (alternates ±).

    Returns:
        Attacked PIL image.
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    center = (h // 2, w // 2)
    max_radius = np.sqrt(center[0] ** 2 + center[1] ** 2)
    ring_width = max_radius / num_rings

    result = img_array.copy()
    for i in range(num_rings):
        inner_r = i * ring_width
        outer_r = (i + 1) * ring_width
        angle = max_angle if (i % 2 == 0) else -max_angle
        mask = _create_ring_mask(img_array.shape[:2], center, inner_r, outer_r)
        result = _rotate_ring(result, mask, angle)

    return Image.fromarray(result.astype(np.uint8))
