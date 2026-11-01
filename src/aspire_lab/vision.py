"""Task-specific RGB-D perception. No simulator body IDs or object poses."""

import numpy as np


def point_cloud(depth, calibration):
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    f = calibration["focal_px"]
    camera = np.stack(((u - (w - 1) / 2) * depth / f, -(v - (h - 1) / 2) * depth / f, -depth), axis=-1)
    return camera @ np.asarray(calibration["rotation"]).reshape(3, 3).T + calibration["position"]


def components(mask):
    """Small dependency-free 4-connected components on a camera mask."""
    remaining = set(zip(*np.where(mask), strict=True))
    groups = []
    while remaining:
        first = remaining.pop()
        todo, group = [first], [first]
        while todo:
            y, x = todo.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    todo.append(neighbor)
                    group.append(neighbor)
        if len(group) >= 12:
            groups.append(np.asarray(group))
    return groups


def detect(rgb, depth, calibration):
    cloud = point_cloud(depth, calibration)
    r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
    orange = (r > 1.3 * g) & (g > 40) & (b < 60) & np.isfinite(depth)
    groups = components(orange)
    result = {"source": "RGB_color_masks_and_rendered_metric_depth", "cube": None, "tray": None}
    if groups:
        pixels = max(groups, key=len)
        pts = cloud[pixels[:, 0], pixels[:, 1]]
        # Known task object is a 4-cm cube; estimate center from visible top surface.
        top = pts[pts[:, 2] >= np.percentile(pts[:, 2], 70) - 0.002]
        center = np.median(top, axis=0)
        center[2] -= 0.02
        result["cube"] = {"position": center.tolist(), "pixels": len(pixels)}
    # White tabletop gives its height; black tray candidates must form a ring,
    # have the declared task dimensions, and sit above that observed table.
    white = (r > 200) & (g > 200) & (b > 200)
    if white.sum() < 100:
        return result
    table_z = float(np.median(cloud[white, 2]))
    result["table_z"] = table_z
    dark = (np.maximum.reduce([r, g, b]) < 65) & (cloud[:, :, 2] > table_z + 0.003)
    dark &= cloud[:, :, 2] < table_z + 0.07
    candidates = []
    for pixels in components(dark):
        pts = cloud[pixels[:, 0], pixels[:, 1]]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        size = hi - lo
        if 0.10 < size[0] < 0.19 and 0.10 < size[1] < 0.19:
            center = (lo + hi) / 2
            center[2] = table_z
            candidates.append((len(pixels), center))
    if len(candidates) == 1:
        result["tray"] = {"position": candidates[0][1].tolist(), "pixels": candidates[0][0]}
    return result
