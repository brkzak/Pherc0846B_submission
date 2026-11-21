"""Read tifxyz segments with reader-v3 and show the ink prediction.

Steps:
1. Render the segment's surface volume from the public CT scan: 17 layers, one voxel apart, centred on the traced
   surface, with the normals pointing towards the scroll's centre (the umbilicus side). The CT is read at the
   pyramid level nearest 9 um, reader-v3's resolution.
2. Run villa's flat ink inference (``vesuvius.ink_detection.inference.infer``) on it, unchanged, with the
   reader-v3 checkpoint (https://huggingface.co/domenicor046/reader-v3): 384 px patches, 50 % overlap, Hann blend.
3. Save the prediction, stretched to the full grey range, at full resolution (and the raw prediction and the CT on
   the surface) to ``output/`` and open it.

Everything comes from villa's ``vesuvius`` package and public packages (requirements.txt):
    uv venv --python 3.14 .venv && uv pip install --python .venv -r requirements.txt

Input: every tifxyz folder ``input/<name>/`` has its settings next to it in ``input/<name>.json`` (keys: CONFIG):
    {"volume": "https://vesuvius-challenge-open-data.s3.amazonaws.com/<scroll>/volumes/<volume>.zarr"}

Usage:
    python infer_letters.py            # every input/<name>.json
    python infer_letters.py NAME ...   # only these
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import zarr
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.ndimage import map_coordinates
from vesuvius.ink_detection.inference import infer
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding
from vesuvius.tifxyz_label_transfer.io import load_surface
from vesuvius.tifxyz_label_transfer.self_render_tifxyz import surface_tile_geometry

HERE = Path(__file__).resolve().parent
# reader-v3, pinned by revision (main on 2026-10-19).
REPO, REVISION, FILENAME = "domenicor046/reader-v3", "39f626e595304c35c6ef574ffc0b6cc1e0ed44cb", "reader-v3.pth"
MODEL_UM = 9.0  # reader-v3 reads ~9 um voxels
LAYERS = 17  # reader-v3's input depth
TILE = 256  # render tile (output pixels)


def render(surface, volume, scale: int, umbilicus_xy, flip: bool, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """The surface volume (LAYERS, H, W) uint8 at the volume's level, and its valid mask (H, W).

    ``scale`` = level-0 voxels per voxel of ``volume``; ``umbilicus_xy`` = the scroll's axis in level-0 voxels.
    Positions and normals follow vc_render_tifxyz (villa's
    ``surface_tile_geometry``); the CT is sampled trilinearly. Layer k is k - LAYERS // 2 voxels along the normal."""
    full_h, full_w = surface.full_resolution_shape
    shape = (math.ceil(full_h / scale), math.ceil(full_w / scale))

    # One sign for the whole segment: the normals point towards the scroll's axis (the umbilicus side) on most of
    # it. (The segment's own centroid is no guide: near the centre a segment may curve either way.)
    xyz, normals, valid = surface_tile_geometry(surface, surface.shape, (0, surface.shape[0], 0, surface.shape[1]))
    towards_axis = ((np.asarray(umbilicus_xy, np.float64) - xyz[valid, :2]) * normals[valid, :2]).sum(1) > 0
    sign = (1.0 if towards_axis.mean() >= 0.5 else -1.0) * (-1.0 if flip else 1.0)
    print(f"normals towards the umbilicus at {tuple(round(float(v)) for v in umbilicus_xy)} on "
          f"{max(towards_axis.mean(), 1 - towards_axis.mean()):.0%} of the segment")
    offsets = np.arange(LAYERS, dtype=np.float64) - LAYERS // 2

    out = np.zeros((LAYERS, *shape), np.uint8)
    mask = np.zeros(shape, bool)
    peak = np.iinfo(volume.dtype).max if volume.dtype.kind in "ui" else 255.0

    def tile(bounds):
        r0, r1, c0, c1 = bounds
        xyz, normals, valid = surface_tile_geometry(surface, shape, bounds)
        if not valid.any():
            return
        points = xyz[valid] / scale  # level-0 voxels -> this level's voxels, (N, 3) as x, y, z
        points = points[None] + sign * offsets[:, None, None] * normals[valid][None]  # (LAYERS, N, 3)
        zyx = points[..., ::-1].reshape(-1, 3)
        lo = np.floor(zyx.min(0)).astype(int) - 1
        hi = np.ceil(zyx.max(0)).astype(int) + 2
        for attempt in range(5):  # S3 sometimes drops a connection; fetched chunks stay in the disk cache
            try:
                block, _ = read_bbox_with_padding(volume, (*lo, *hi))
                break
            except Exception as error:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"\nCT read failed ({error!r}), retrying")
                time.sleep(2 ** attempt)
        values = map_coordinates(block.astype(np.float32), (zyx - lo).T, order=1, mode="constant", cval=0.0)
        values = np.clip(np.rint(values * (255.0 / peak)), 0, 255).astype(np.uint8).reshape(LAYERS, -1)
        view = out[:, r0:r1, c0:c1]
        view[:, valid] = values
        mask[r0:r1, c0:c1] = valid

    tiles = [(r, min(r + TILE, shape[0]), c, min(c + TILE, shape[1]))
             for r in range(0, shape[0], TILE) for c in range(0, shape[1], TILE)]
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(tile, tiles), 1):
            print(f"\rrender: {i}/{len(tiles)} tiles", end="", flush=True)
    print()
    return out, mask


CONFIG = {  # keys of input/<tifxyz name>.json, with their defaults (None: required)
    "volume": None,  # OME-zarr URL (or path) of the CT scan the segment was traced on
    "voxel_um": None,  # the scan's level-0 voxel size; null: read from the volume name ("...-9.362um-...")
    "umbilicus_xy": None,  # the scroll's axis [x, y] in level-0 voxels; null: the centre of the volume
    "flip_normals": False,  # read the other side of the sheet
    "batch_size": 4,
    "workers": 8,  # render threads
    "show": True,  # open the prediction when done
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    unknown = set(config) - set(CONFIG)
    if unknown:
        raise SystemExit(f"{path.name}: unknown keys {sorted(unknown)}; known: {sorted(CONFIG)}")
    config = {**CONFIG, **config}
    if not config["volume"]:
        raise SystemExit(f"{path.name}: 'volume' is required")
    if config["voxel_um"] is None:
        match = re.search(r"(\d+(?:\.\d+)?)um", config["volume"])
        if not match:
            raise SystemExit(f"{path.name}: cannot read the voxel size from the volume name; set 'voxel_um'")
        config["voxel_um"] = float(match.group(1))
    return config


def normalise(prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The prediction stretched to 0..255 between its 1st and 99.9th percentile on the segment (0 off it)."""
    values = prediction[mask].astype(np.float32)
    if values.size == 0:
        return np.zeros_like(prediction)
    lo, hi = np.percentile(values, (1.0, 99.9))
    out = np.clip((prediction.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return np.where(mask, np.rint(out), 0).astype(np.uint8)


def infer_segment(segment: Path, config: dict) -> None:
    """Render, infer and save one tifxyz, all at the render level's full resolution (nothing is downsampled)."""
    level = max(0, round(math.log2(MODEL_UM / config["voxel_um"])))
    print(f"{segment.name}: {config['voxel_um']} um scan, level {level} ({config['voxel_um'] * 2 ** level:.2f} um)")

    work = HERE / "work" / segment.name
    work.mkdir(parents=True, exist_ok=True)
    volume = open_volume(config["volume"], str(level), cache_dir=HERE / "work" / "ct-cache")
    umbilicus_xy = config["umbilicus_xy"] or (volume.shape[2] * 2 ** level / 2, volume.shape[1] * 2 ** level / 2)
    layers, mask = render(load_surface(segment), volume, 2 ** level, umbilicus_xy, config["flip_normals"],
                          config["workers"])
    surface_zarr = work / "surface_volume.zarr"
    zarr.open_array(str(surface_zarr), mode="w", shape=layers.shape, chunks=(LAYERS, TILE, TILE), dtype=np.uint8,
                    zarr_format=2)[:] = layers
    tifffile.imwrite(work / "mask.tif", mask.astype(np.uint8) * 255)

    checkpoint = hf_hub_download(REPO, FILENAME, revision=REVISION)
    prediction = work / "prediction.tif"
    infer.main([str(surface_zarr), checkpoint, str(prediction), "--mask-path", str(work / "mask.tif"),
                "--overlap", "0.5", "--blend-mode", "hann", "--direction", "forward",
                "--batch-size", str(config["batch_size"]), "--num-workers", "0", "--no-compile"])

    out = HERE / "output"
    out.mkdir(exist_ok=True)
    raw = tifffile.imread(prediction)
    tifffile.imwrite(out / f"{segment.name}_reader-v3_raw.tif", raw)  # villa's probability x 255, unnormalised
    ink = Image.fromarray(normalise(raw, mask))
    ink.save(out / f"{segment.name}_reader-v3.png")
    Image.fromarray(layers[LAYERS // 2]).save(out / f"{segment.name}_surface.png")
    print(f"saved {out / f'{segment.name}_reader-v3.png'} ({raw.shape[1]} x {raw.shape[0]} px), "
          "the raw prediction and the CT on the surface next to it")
    if config["show"]:
        ink.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("segments", nargs="*", help="tifxyz names in input/ (default: every input/<name>.json)")
    names = ap.parse_args().segments or sorted(p.stem for p in (HERE / "input").glob("*.json"))
    if not names:
        raise SystemExit("no input/<tifxyz name>.json found")
    for name in names:
        config, segment = HERE / "input" / f"{name}.json", HERE / "input" / name
        if not config.is_file():
            raise SystemExit(f"missing {config}")
        if not (segment / "x.tif").is_file():
            raise SystemExit(f"{config.name}: no tifxyz folder {segment}")
    for name in names:
        infer_segment(HERE / "input" / name, load_config(HERE / "input" / f"{name}.json"))


if __name__ == "__main__":
    main()
