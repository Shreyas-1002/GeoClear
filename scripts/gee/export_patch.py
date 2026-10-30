"""GeoClear — Google Earth Engine Python Export Script.

Provides automated export of co-registered Sentinel-1 and Sentinel-2 patches
to Google Drive for users who have `earthengine-api` installed and authenticated.

Requirements:
    pip install earthengine-api
    earthengine authenticate

Usage:
    python scripts/gee/export_patch.py --lon 11.5761 --lat 48.1371 --out data/real_patch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export Sentinel-1 and Sentinel-2 patches via Google Earth Engine.")
    parser.add_argument("--lon", type=float, default=11.5761, help="Center longitude (default 11.5761).")
    parser.add_argument("--lat", type=float, default=48.1371, help="Center latitude (default 48.1371).")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size in pixels (must be multiple of 32, default 256).")
    parser.add_argument("--scale", type=int, default=10, help="Pixel resolution in meters (default 10).")
    parser.add_argument("--s2-start", type=str, default="2023-06-01", help="Cloudy optical search start date.")
    parser.add_argument("--s2-end", type=str, default="2023-08-31", help="Cloudy optical search end date.")
    parser.add_argument("--drive-folder", type=str, default="GeoClear_Real_Patch", help="Destination folder in Google Drive.")
    parser.add_argument("--out-manifest", type=str, default="data/real_patch/manifest.json", help="Path to write manifest.json.")

    args = parser.parse_args()

    if args.patch_size % 32 != 0:
        print(f"Error: patch-size must be divisible by 32, got {args.patch_size}", file=sys.stderr)
        sys.exit(1)

    try:
        import ee
    except ImportError:
        print(
            "Error: earthengine-api is not installed in your Python environment.\n"
            "To use the GEE Python API, run: pip install earthengine-api && earthengine authenticate\n"
            "Alternatively, use the zero-install Code Editor script at scripts/gee/export_patch.js.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        ee.Initialize()
    except Exception as exc:
        print(f"Failed to initialize Earth Engine: {exc}\nRun 'earthengine authenticate' first.", file=sys.stderr)
        sys.exit(1)

    span_meters = args.patch_size * args.scale
    center = ee.Geometry.Point([args.lon, args.lat])
    aoi = center.buffer(span_meters / 2).bounds()

    s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

    # 1. Target cloudy Sentinel-2 Level-1C TOA
    s2_col = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(args.s2_start, args.s2_end)
        .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', 25))
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 75))
        .sort('CLOUDY_PIXEL_PERCENTAGE', False)
    )

    s2_img = ee.Image(s2_col.first()).select(s2_bands)
    s2_date = ee.Date(s2_img.get('system:time_start'))

    # 2. Sentinel-1 SAR (IW, VV/VH in dB)
    sar_start = s2_date.advance(-5, 'day')
    sar_end = s2_date.advance(5, 'day')

    s1_col = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(aoi)
        .filterDate(sar_start, sar_end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    )

    def compute_diff(img):
        t_diff = ee.Number(img.get('system:time_start')).subtract(s2_date.millis()).abs()
        return img.set('timeDiff', t_diff)

    s1_img = s1_col.map(compute_diff).sort('timeDiff').first().select(['VV', 'VH'])
    s1_date = ee.Date(s1_img.get('system:time_start'))
    time_offset_hours = s1_date.difference(s2_date, 'hour').getInfo()

    # Reproject SAR to match Sentinel-2 B4 UTM grid
    s2_proj = s2_img.select('B4').projection()
    s1_reprojected = s1_img.reproject(crs=s2_proj, scale=args.scale).clip(aoi)
    s2_clipped = s2_img.clip(aoi)

    # 3. Reference optical
    ref_col = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterBounds(aoi)
        .filterDate('2023-04-01', '2023-10-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .sort('CLOUDY_PIXEL_PERCENTAGE', True)
    )
    ref_img = ee.Image(ref_col.first()).select(s2_bands).clip(aoi)

    # Launch export tasks
    task_s1 = ee.batch.Export.image.toDrive(
        image=s1_reprojected.toFloat(),
        description="s1_sar",
        folder=args.drive_folder,
        fileNamePrefix="s1_sar",
        region=aoi,
        scale=args.scale,
        crs=s2_proj.crs().getInfo(),
        fileFormat="GeoTIFF",
    )
    task_s1.start()

    task_s2 = ee.batch.Export.image.toDrive(
        image=s2_clipped.toFloat(),
        description="s2_cloudy",
        folder=args.drive_folder,
        fileNamePrefix="s2_cloudy",
        region=aoi,
        scale=args.scale,
        crs=s2_proj.crs().getInfo(),
        fileFormat="GeoTIFF",
    )
    task_s2.start()

    task_ref = ee.batch.Export.image.toDrive(
        image=ref_img.toFloat(),
        description="s2_reference",
        folder=args.drive_folder,
        fileNamePrefix="s2_reference",
        region=aoi,
        scale=args.scale,
        crs=s2_proj.crs().getInfo(),
        fileFormat="GeoTIFF",
    )
    task_ref.start()

    print(f"Export tasks submitted to Google Drive folder '{args.drive_folder}'.")

    # Save manifest
    manifest_info = {
        "s2_collection": "COPERNICUS/S2_HARMONIZED",
        "s2_product_type": "Level-1C (Top-Of-Atmosphere Reflectance)",
        "s2_scene_id": s2_img.get("system:index").getInfo(),
        "s2_sensing_date": s2_date.format("YYYY-MM-dd HH:mm:ss").getInfo(),
        "s1_collection": "COPERNICUS/S1_GRD",
        "s1_mode": "IW",
        "s1_scene_id": s1_img.get("system:index").getInfo(),
        "s1_sensing_date": s1_date.format("YYYY-MM-dd HH:mm:ss").getInfo(),
        "temporal_offset_hours": round(float(time_offset_hours), 2),
        "reference_scene_id": ref_img.get("system:index").getInfo(),
        "crs": s2_proj.crs().getInfo(),
        "scale_meters": args.scale,
        "patch_size_pixels": args.patch_size,
        "sar_bands": ["VV", "VH"],
        "optical_bands": s2_bands,
    }

    manifest_path = Path(args.out_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    print(f"Wrote provenance manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
