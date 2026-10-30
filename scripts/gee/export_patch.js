/**
 * GeoClear — Google Earth Engine Export Script for Real Sentinel Patches
 * 
 * Target Model: ECRformer-Light (Multimodal Optical-SAR Cloud Reconstruction)
 * 
 * Scientific & Technical Provenance:
 * 1. Optical Collection: COPERNICUS/S2_HARMONIZED (Sentinel-2 Level-1C Top-Of-Atmosphere)
 *    - All 13 bands: ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
 *    - Value semantics: Top-Of-Atmosphere (TOA) reflectance scaled by 10,000 (DN).
 *      Dividing by 10000.0 and clipping to [0.0, 1.0] scales to [0, 1] as expected by ECRformer.
 *      These are radiometrically calibrated and orthorectified TOA values, NOT raw instrument detector counts.
 *    - Level-2A (COPERNICUS/S2_SR_HARMONIZED) CANNOT be used because it omits Band 10 (Cirrus).
 * 2. SAR Collection: COPERNICUS/S1_GRD (Sentinel-1 Ground Range Detected, IW mode)
 *    - Polarizations: ['VV', 'VH'] in that exact order.
 *    - Radiometrically terrain-corrected sigma0 backscatter in decibels (dB).
 *    - ECRformer clips [-25, 0] dB and normalizes via (x + 25) / 25. No additional log-conversion.
 * 3. Exact Shared Pixel Grid:
 *    - Target optical B4 projection defines the projected CRS (e.g. UTM) and nominal 10m grid spacing.
 *    - Upper-left origin is snapped to the nearest 10m UTM grid boundary.
 *    - Exact region bounds: planar 2560m x 2560m polygon (geodesic: false).
 *    - crsTransform: [10, 0, xMin, 0, -10, yMax].
 *    - Guaranteed exact output dimensions: 256 x 256 pixels at 10m resolution.
 *    - Parameters crs, crsTransform, and region are specified; scale and dimensions are omitted
 *      to avoid Earth Engine parameter exclusivity conflicts.
 * 4. AOI Cloud Assessment (s2cloudless vs QA60 Historical Gap):
 *    - IMPORTANT HISTORICAL GAP: Beginning January 25, 2022 (Processing Baseline 04.00), ESA changed
 *      cloud mask generation. In Earth Engine's harmonized archive for 2022 and 2023, the QA60 band
 *      is frequently all zeros, reporting 0% clouds even across cloud-covered scenes.
 *    - DO NOT rely on QA60 simply because the band exists.
 *    - Cloud assessment is performed by joining COPERNICUS/S2_CLOUD_PROBABILITY (s2cloudless) using system:index.
 *    - Cloud probability (continuous 0-100%) is kept distinct from binary classified cloud pixels.
 *    - If cloud probability is unavailable, the script reports "Unknown" and warns explicitly.
 *      Missing cloud data is NEVER silently treated as 0% cloud cover.
 * 
 * Instructions:
 * 1. Open Google Earth Engine Code Editor: https://code.earthengine.google.com/
 * 2. Paste this entire script into the editor.
 * 3. Click "Run" at the top of the editor.
 * 4. Inspect the interactive map layers and console diagnostics.
 * 5. Open the "Tasks" tab in the top-right panel.
 * 6. Click "Run" next to the export tasks to export to Google Drive:
 *    - s1_sar (Sentinel-1 VV/VH, 2 bands in dB)
 *    - s2_cloudy (Sentinel-2 L1C TOA, 13 bands in DN [0, 10000])
 *    - s2_reference (optional Sentinel-2 clean comparison scene)
 * 7. Download exported files into: D:\GeoClear\data\real_patch\
 * 8. Copy the JSON block printed in the Console into: D:\GeoClear\data\real_patch\manifest.json
 */

// =========================================================================
// 1. CONFIGURATION PARAMETERS (Adjust as needed)
// =========================================================================

var CONFIG = {
  // Center coordinates (Longitude, Latitude in WGS84)
  // Default: Mixed agricultural & forested terrain near Munich, Germany
  center_lon: 11.5761,
  center_lat: 48.1371,

  // Patch specifications
  // Must be divisible by 32 for ECRformer (256x256 pixels at 10m = 2560m x 2560m)
  patch_size_pixels: 256,
  pixel_scale_meters: 10,

  // Target cloudy Sentinel-2 search window & scene-wide cloud percentage filters
  s2_target_start: '2023-06-01',
  s2_target_end: '2023-08-31',
  s2_target_cloud_min: 20, // Min scene-wide cloud % to locate cloud-impacted imagery
  s2_target_cloud_max: 80, // Max scene-wide cloud % to avoid totally saturated scenes

  // Sentinel-1 SAR temporal search window around target S2 acquisition date (in days)
  sar_window_days: 5,

  // Reference optical (clean/comparison) search window & filters
  export_reference: true,  // Set to false if no clean comparison is available
  ref_start: '2023-05-01',
  ref_end: '2023-09-30',
  ref_cloud_max: 10,       // Max scene-wide cloud % for candidate reference scenes

  // Cloud probability classification threshold (0 - 100%)
  // Pixels with s2cloudless probability >= threshold are classified as cloudy
  cloud_probability_threshold: 40,

  // Google Drive destination folder
  drive_folder: 'GeoClear_Real_Patch'
};

// =========================================================================
// 2. CONFIGURATION VALIDATION
// =========================================================================

if (CONFIG.patch_size_pixels % 32 !== 0) {
  throw new Error('CONFIG.patch_size_pixels (' + CONFIG.patch_size_pixels + ') must be divisible by 32 for ECRformer processing.');
}

var centerPoint = ee.Geometry.Point([CONFIG.center_lon, CONFIG.center_lat]);
var roughAoi = centerPoint.buffer(CONFIG.patch_size_pixels * CONFIG.pixel_scale_meters / 2);

// Sentinel-2 L1C required band order (SEN12MS-CR benchmark convention)
var S2_BANDS_13 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'];

// =========================================================================
// 3. TARGET SENTINEL-2 QUERY & VACANCY AUDIT
// =========================================================================

var s2TargetCol = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterBounds(roughAoi)
  .filterDate(CONFIG.s2_target_start, CONFIG.s2_target_end)
  .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', CONFIG.s2_target_cloud_min))
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CONFIG.s2_target_cloud_max))
  .sort('CLOUDY_PIXEL_PERCENTAGE', false);

var s2TargetCount = s2TargetCol.size().getInfo();
print('================== 1. SCENE SELECTION AUDIT ==================');
print('Target Sentinel-2 candidates found:', s2TargetCount);

if (s2TargetCount === 0) {
  print('❌ [CRITICAL ERROR] Target Sentinel-2 collection is empty.');
  print('Diagnostic: No scenes found for AOI [' + CONFIG.center_lon + ', ' + CONFIG.center_lat + '] between ' +
    CONFIG.s2_target_start + ' and ' + CONFIG.s2_target_end + ' with scene cloud cover ' +
    CONFIG.s2_target_cloud_min + '%-' + CONFIG.s2_target_cloud_max + '%.');
  print('Action required: Expand date range or adjust cloud percentage thresholds in CONFIG.');
} else {

  // Select target Sentinel-2 scene
  var s2CloudyImg = s2TargetCol.first();
  var s2Date = ee.Date(s2CloudyImg.get('system:time_start'));
  var s2Id = s2CloudyImg.get('system:index');
  var s2SceneCloud = s2CloudyImg.get('CLOUDY_PIXEL_PERCENTAGE');

  // =======================================================================
  // 4. SENTINEL-1 SAR QUERY & TEMPORAL MATCHING AUDIT
  // =======================================================================

  var sarStart = s2Date.advance(-CONFIG.sar_window_days, 'day');
  var sarEnd = s2Date.advance(CONFIG.sar_window_days, 'day');

  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roughAoi)
    .filterDate(sarStart, sarEnd)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  var s1Count = s1Col.size().getInfo();
  print('Sentinel-1 SAR candidates found (±' + CONFIG.sar_window_days + ' days):', s1Count);

  if (s1Count === 0) {
    print('❌ [CRITICAL ERROR] Sentinel-1 SAR collection is empty.');
    print('Diagnostic: No IW VV/VH acquisitions found within ±' + CONFIG.sar_window_days +
      ' days of target Sentinel-2 acquisition.');
    print('Action required: Increase CONFIG.sar_window_days (e.g. to 10 or 15 days) to find an overlapping SAR pass.');
  } else {

    // Select SAR scene with minimum absolute temporal offset to optical sensing time
    var s1Matched = s1Col.map(function (img) {
      var timeDiff = ee.Number(img.get('system:time_start')).subtract(s2Date.millis()).abs();
      return img.set('timeDiffMillis', timeDiff);
    }).sort('timeDiffMillis').first();

    var s1Date = ee.Date(s1Matched.get('system:time_start'));
    var s1Id = s1Matched.get('system:index');
    var temporalOffsetHours = s1Date.difference(s2Date, 'hour');

    // =====================================================================
    // 5. REFERENCE SENTINEL-2 QUERY & SELECTION
    // =====================================================================

    var s2RefCol = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
      .filterBounds(roughAoi)
      .filterDate(CONFIG.ref_start, CONFIG.ref_end)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.ref_cloud_max))
      .sort('CLOUDY_PIXEL_PERCENTAGE', true);

    var s2RefCount = s2RefCol.size().getInfo();
    print('Reference Sentinel-2 candidates found:', s2RefCount);
    var hasReference = CONFIG.export_reference && (s2RefCount > 0);
    var s2RefImg = hasReference ? s2RefCol.first() : null;

    // =====================================================================
    // 6. EXACT SHARED PIXEL GRID & AFFINE TRANSFORM DERIVATION
    // =====================================================================
    // Obtain native UTM CRS string from Sentinel-2 Band 4 (e.g. 'EPSG:32632')
    var s2Proj = s2CloudyImg.select('B4').projection();
    var targetCrs = s2Proj.crs(); // 'EPSG:32632'

    // Construct a projected coordinate system using the CRS string directly (units: meters)
    // CRITICAL ROOT-CAUSE FIX: Do NOT pass s2Proj into centerPoint.transform() directly!
    // s2Proj contains the image's pixel affine transform matrix [10, 0, x0, 0, -10, y0].
    // Passing an image projection converts the geometry into granule pixel coordinates (e.g. x=7880, y=7810 px).
    // Instead, passing targetCrs (e.g. 'EPSG:32632') or ee.Projection(targetCrs) transforms into
    // standard projected UTM coordinates in METERS (e.g. Easting ~691648 m, Northing ~5334748 m)!
    var utmProj = ee.Projection(targetCrs);
    var centerInUtm = centerPoint.transform(utmProj, 1);
    var centerCoords = centerInUtm.coordinates();
    var cx = ee.Number(centerCoords.get(0));
    var cy = ee.Number(centerCoords.get(1));

    var patchSpanMeters = ee.Number(CONFIG.patch_size_pixels).multiply(CONFIG.pixel_scale_meters); // 2560m
    var halfSpanMeters = patchSpanMeters.divide(2); // 1280m

    // Snap origin (upper-left corner) to the nearest 10m grid boundary in projected meter space
    var xMin = cx.subtract(halfSpanMeters).divide(CONFIG.pixel_scale_meters).round().multiply(CONFIG.pixel_scale_meters);
    var yMax = cy.add(halfSpanMeters).divide(CONFIG.pixel_scale_meters).round().multiply(CONFIG.pixel_scale_meters);
    var xMax = xMin.add(patchSpanMeters);
    var yMin = yMax.subtract(patchSpanMeters);

    // Evaluate grid parameters to obtain clean numerical values
    var gridInfo = ee.Dictionary({
      crs: targetCrs,
      center_x_meters: cx,
      center_y_meters: cy,
      xMin: xMin,
      yMax: yMax,
      xMax: xMax,
      yMin: yMin,
      width_meters: patchSpanMeters,
      height_meters: patchSpanMeters
    }).getInfo();

    print('================== 1b. PROJECTED GRID DIAGNOSTICS ==================');
    print('Original WGS84 AOI Center [lon, lat]:', [CONFIG.center_lon, CONFIG.center_lat]);
    print('Target Projected CRS:', gridInfo.crs);
    print('Transformed Center in Projected CRS [X, Y] (meters):', [gridInfo.center_x_meters, gridInfo.center_y_meters]);
    print('Snapped Grid Origin (Upper-Left) [xMin, yMax]:', [gridInfo.xMin, gridInfo.yMax]);
    print('Exact Grid Bounds [xMin, yMin, xMax, yMax]:', [gridInfo.xMin, gridInfo.yMin, gridInfo.xMax, gridInfo.yMax]);

    // Sanity check: verify coordinates are geographically plausible for UTM
    if (gridInfo.crs.indexOf('EPSG:326') === 0 || gridInfo.crs.indexOf('EPSG:327') === 0) {
      if (gridInfo.center_x_meters < 100000 || gridInfo.center_x_meters > 900000 || gridInfo.center_y_meters < 100000) {
        print('❌ [CRITICAL ERROR] Transformed projected coordinates are not geographically plausible for ' + gridInfo.crs + '!');
        print('Observed: X=' + gridInfo.center_x_meters + ', Y=' + gridInfo.center_y_meters + '. Expected UTM Easting ~100,000-900,000m, Northing >100,000m.');
        print('Stopping script execution to prevent invalid exports.');
        throw new Error(
          'GeoClear export aborted: projected UTM coordinates failed the sanity check.'
        );
      }
    }

    // Explicit 6-element affine transform: [xScale, xShear, xOrigin, yShear, yScale, yOrigin]
    var crsTransform = [
      CONFIG.pixel_scale_meters,
      0,
      gridInfo.xMin,
      0,
      -CONFIG.pixel_scale_meters,
      gridInfo.yMax
    ];

    // Construct exact planar bounding polygon in the projected CRS (geodesic: false).
    // CRITICAL ROOT-CAUSE FIX (257x257 -> 256x256):
    // In Earth Engine, Export.image.toDrive with an explicit crsTransform rasterizes all grid cells
    // that intersect the given region geometry. If the bounding coordinates fall exactly on the
    // outer grid boundary (xMax = xMin + 2560m, yMin = yMax - 2560m), boundary inclusion snapping
    // causes GEE to include the 257th cell on the right and bottom edges, producing a 257x257 GeoTIFF
    // whose 257th row and column are empty (NaNs).
    // Insetting the outer boundary by 1 meter ensures that only cells 0..255 in both X and Y
    // are intersected, producing an EXACT 256 x 256 pixel raster strictly aligned with crsTransform.
    var xMaxRegion = gridInfo.xMax - 1;
    var yMinRegion = gridInfo.yMin + 1;

    var exportRegion = ee.Geometry.Polygon({
      coords: [[[gridInfo.xMin, yMinRegion],
      [xMaxRegion, yMinRegion],
      [xMaxRegion, gridInfo.yMax],
      [gridInfo.xMin, gridInfo.yMax],
      [gridInfo.xMin, yMinRegion]]],
      proj: gridInfo.crs,
      geodesic: false
    });

    // Check spatial footprint intersection with target Sentinel-2
    var s2Footprint = s2CloudyImg.geometry();
    var s2Intersects = exportRegion.intersects(s2Footprint, 1).getInfo();
    print('Patch intersects target Sentinel-2 footprint:', s2Intersects);

    if (!s2Intersects) {
      print('⚠️ WARNING: Export patch region does NOT intersect the selected Sentinel-2 image footprint!');
    }

    // =====================================================================
    // 7. RESAMPLING & GRID ALIGNMENT POLICY
    // =====================================================================
    // Optical: Resample multispectral bands to common 10m grid using bilinear interpolation.
    // 10m bands (B2, B3, B4, B8) preserve native sampling.
    // 20m bands (B5-B7, B8A, B11, B12) and 60m bands (B1, B9, B10) are smoothly upsampled.
    // LIMITATION: Bilinear interpolation does NOT synthesize real sub-resolution physical detail;
    // Band 10 (Cirrus) remains spatially smooth at ~60m optical fidelity.
    var s2CloudySelected = s2CloudyImg.select(S2_BANDS_13).resample('bilinear');

    // SAR: Resample VV/VH to the common 10m UTM grid using bilinear interpolation.
    // Preserves smooth backscatter transitions and prevents nearest-neighbor grid steps.
    var s1Selected = s1Matched.select(['VV', 'VH']).resample('bilinear');

    var s1CoRegistered = s1Selected.reproject({
      crs: gridInfo.crs,
      crsTransform: crsTransform
    });

    // Check spatial footprint intersection with Sentinel-1
    var s1Footprint = s1CoRegistered.geometry();
    var s1Intersects = exportRegion.intersects(s1Footprint, 1).getInfo();
    print('Patch intersects target Sentinel-1 footprint:', s1Intersects);

    if (!s1Intersects) {
      print('⚠️ WARNING: Export patch region does NOT intersect the selected Sentinel-1 image footprint!');
    }

    var s2RefSelected = hasReference ? s2RefImg.select(S2_BANDS_13).resample('bilinear') : null;

    // =====================================================================
    // 8. AOI-SPECIFIC CLOUD ASSESSMENT (s2cloudless vs QA60 Gap)
    // =====================================================================
    // Historical QA60 Audit:
    // In Processing Baseline 04.00 (January 2022 onwards, including 2023), ESA changed cloud mask
    // encoding. In Earth Engine's archive, QA60 is frequently zero-filled and reports 0% cloud cover
    // even for overcast imagery. We do not rely on QA60.
    // Instead, we join COPERNICUS/S2_CLOUD_PROBABILITY (s2cloudless) by scene identifier (system:index).

    function assessSceneCloud(s2Image, sceneIdStr, label, imgDate) {
      var probCol = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filter(ee.Filter.eq('system:index', sceneIdStr));

      var matchCount = probCol.size().getInfo();

      // Secondary fallback: query by spatial overlap and sensing timestamp if exact system:index match is 0
      if (matchCount === 0 && imgDate) {
        probCol = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
          .filterBounds(exportRegion)
          .filterDate(imgDate.advance(-30, 'minute'), imgDate.advance(30, 'minute'));
        matchCount = probCol.size().getInfo();
      }

      print('--- Cloud Assessment: ' + label + ' ---');
      print('Selected S2 Image system:index:', sceneIdStr);
      print('Matching COPERNICUS/S2_CLOUD_PROBABILITY candidates:', matchCount);

      if (matchCount === 0) {
        var warnMsg = 'No matching s2cloudless image found in COPERNICUS/S2_CLOUD_PROBABILITY for ' + label +
          ' (scene ' + sceneIdStr + '). Cloud assessment is UNKNOWN. Inspect RGB/Cirrus visually.';
        print('⚠️ WARNING:', warnMsg);
        return {
          status: 'Unknown',
          valid: false,
          method: 'COPERNICUS/S2_CLOUD_PROBABILITY (s2cloudless)',
          threshold: CONFIG.cloud_probability_threshold,
          mean_probability: null,
          binary_cloud_percentage: null,
          warning: warnMsg,
          probImage: null,
          binaryMask: null
        };
      }

      var probImg = ee.Image(probCol.first()).select('probability');
      var probImgId = probImg.get('system:index').getInfo();
      var probProj = probImg.projection();
      var probCrs = probProj.crs().getInfo();

      print('Matched Cloud Probability Image ID:', probImgId);
      print('Matched Cloud Probability Projection/CRS:', probCrs);

      var probFootprint = probImg.geometry();
      var probIntersects = exportRegion.intersects(probFootprint, 1).getInfo();
      print('Cloud Probability raster intersects patch region:', probIntersects);

      if (!probIntersects) {
        var warnNoIntersect = 'Matched cloud probability raster does NOT overlap the export region for ' + label + '!';
        print('⚠️ WARNING:', warnNoIntersect);
        return {
          status: 'Unknown',
          valid: false,
          method: 'COPERNICUS/S2_CLOUD_PROBABILITY (s2cloudless)',
          threshold: CONFIG.cloud_probability_threshold,
          mean_probability: null,
          binary_cloud_percentage: null,
          warning: warnNoIntersect,
          probImage: probImg,
          binaryMask: null
        };
      }

      var binaryCloudMask = probImg.gte(CONFIG.cloud_probability_threshold);

      // Reduce over exact exportRegion in projected grid
      var statsDict = probImg.rename('prob')
        .addBands(binaryCloudMask.rename('binary'))
        .reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: exportRegion,
          crs: gridInfo.crs,
          crsTransform: crsTransform,
          maxPixels: 1e6
        }).getInfo();

      if (!statsDict || statsDict.prob === null || statsDict.prob === undefined) {
        var warnNullStats = 'Cloud probability reducer returned null values over export region for ' + label + '. Cloud cover is UNKNOWN.';
        print('⚠️ WARNING:', warnNullStats);
        return {
          status: 'Unknown',
          valid: false,
          method: 'COPERNICUS/S2_CLOUD_PROBABILITY (s2cloudless)',
          threshold: CONFIG.cloud_probability_threshold,
          mean_probability: null,
          binary_cloud_percentage: null,
          warning: warnNullStats,
          probImage: probImg,
          binaryMask: binaryCloudMask
        };
      }

      var meanProbVal = parseFloat(Number(statsDict.prob).toFixed(1));
      var binaryPctVal = parseFloat((Number(statsDict.binary) * 100).toFixed(1));

      return {
        status: 'Valid',
        valid: true,
        method: 'COPERNICUS/S2_CLOUD_PROBABILITY (s2cloudless)',
        threshold: CONFIG.cloud_probability_threshold,
        mean_probability: meanProbVal,
        binary_cloud_percentage: binaryPctVal,
        warning: null,
        probImage: probImg,
        binaryMask: binaryCloudMask
      };
    }

    var targetSceneId = s2Id.getInfo();
    var targetCloudAssessment = assessSceneCloud(s2CloudyImg, targetSceneId, 'Target S2', s2Date);

    // Assess reference scene if available
    var refSceneId = hasReference ? s2RefImg.get('system:index').getInfo() : null;
    var refCloudAssessment = hasReference ? assessSceneCloud(s2RefImg, refSceneId, 'Reference S2', ee.Date(s2RefImg.get('system:time_start'))) : null;

    // =====================================================================
    // 9. VALID-PIXEL QUALITY INSPECTION (Optical & SAR)
    // =====================================================================

    var optValidMask = s2CloudyImg.select('B4').mask();
    var optValidStats = optValidMask.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: exportRegion,
      crs: gridInfo.crs,
      crsTransform: crsTransform,
      maxPixels: 1e6
    });

    var sarValidMask = s1CoRegistered.select('VV').mask();
    var sarValidStats = sarValidMask.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: exportRegion,
      crs: gridInfo.crs,
      crsTransform: crsTransform,
      maxPixels: 1e6
    });

    var aoiStatsDict = {
      opt_valid_fraction: optValidStats.get('B4'),
      sar_valid_fraction: sarValidStats.get('VV'),
      s2_date_str: s2Date.format('yyyy-MM-dd HH:mm:ss', 'UTC'),
      s1_date_str: s1Date.format('yyyy-MM-dd HH:mm:ss', 'UTC'),
      s1_temporal_offset_hours: temporalOffsetHours
    };

    if (hasReference) {
      var refDate = ee.Date(s2RefImg.get('system:time_start'));
      aoiStatsDict.ref_date_str = refDate.format('yyyy-MM-dd HH:mm:ss', 'UTC');
      aoiStatsDict.ref_temporal_diff_days = refDate.difference(s2Date, 'day');
    }

    var aoiStats = ee.Dictionary(aoiStatsDict).getInfo();

    var optValidPct = aoiStats.opt_valid_fraction !== null ? (aoiStats.opt_valid_fraction * 100).toFixed(1) : 'Unknown';
    var sarValidPct = aoiStats.sar_valid_fraction !== null ? (aoiStats.sar_valid_fraction * 100).toFixed(1) : 'Unknown';

    print('================== 2. PATCH CLOUD & VALIDITY AUDIT ==================');
    print('Scene-wide cloud percentage (S2 metadata):', s2SceneCloud.getInfo(), '%');
    print('AOI Cloud Assessment Method:', targetCloudAssessment.method);
    print('AOI Cloud Assessment Status:', targetCloudAssessment.status);

    if (targetCloudAssessment.valid) {
      print('AOI Mean Cloud Probability (0-100%):', targetCloudAssessment.mean_probability, '%');
      print('AOI Classified Cloud Pixels (prob >=' + targetCloudAssessment.threshold + '%):', targetCloudAssessment.binary_cloud_percentage, '%');
    } else {
      print('⚠️ AOI Cloud Probability:', 'Unknown (Join to s2cloudless unavailable)');
      print('⚠️ AOI Classified Cloud Pixels:', 'Unknown');
    }

    print('AOI Optical Valid Pixel Percentage:', optValidPct, '%');
    print('AOI SAR Valid Pixel Percentage:', sarValidPct, '%');

    var warnings = [];

    if (!targetCloudAssessment.valid) {
      print('⚠️ WARNING:', targetCloudAssessment.warning);
      warnings.push(targetCloudAssessment.warning);
    } else {
      if (targetCloudAssessment.binary_cloud_percentage < 5) {
        var warnCloudLow = 'Target patch has low classified cloud cover (' + targetCloudAssessment.binary_cloud_percentage + '%). ' +
          'Cloud reconstruction evaluation is not meaningful on an already clear patch.';
        print('⚠️ WARNING:', warnCloudLow);
        warnings.push(warnCloudLow);
      } else if (targetCloudAssessment.binary_cloud_percentage > 95) {
        var warnCloudHigh = 'Target patch is almost completely cloud-saturated (' + targetCloudAssessment.binary_cloud_percentage + '%). ' +
          'Reconstruction will rely predominantly on SAR without optical edge cues.';
        print('ℹ️ NOTE:', warnCloudHigh);
        warnings.push(warnCloudHigh);
      }
    }

    if (parseFloat(optValidPct) < 100 || parseFloat(sarValidPct) < 100) {
      var warnCoverage = 'Patch contains missing/masked pixels (Optical: ' + optValidPct + '%, SAR: ' + sarValidPct + '%). ' +
        'ECRformer requires 100% valid finite numbers. Move center coordinates fully inside scene footprints.';
      print('⚠️ WARNING:', warnCoverage);
      warnings.push(warnCoverage);
    }

    var refIsStrictGroundTruth = false;
    if (hasReference) {
      print('Reference sensing date:', aoiStats.ref_date_str);
      print('Reference temporal difference:', Math.abs(Number(aoiStats.ref_temporal_diff_days)).toFixed(1), 'days');
      print('Reference Cloud Assessment Status:', refCloudAssessment.status);

      if (refCloudAssessment.valid) {
        print('Reference AOI Mean Cloud Probability:', refCloudAssessment.mean_probability, '%');
        print('Reference AOI Classified Cloud Pixels:', refCloudAssessment.binary_cloud_percentage, '%');

        if (refCloudAssessment.binary_cloud_percentage > 0) {
          var warnRefCloud = 'Reference patch contains detected clouds/cirrus (' + refCloudAssessment.binary_cloud_percentage + '%). ' +
            'It CANNOT be used as strict ground truth for quantitative accuracy evaluation.';
          print('⚠️ WARNING:', warnRefCloud);
          warnings.push(warnRefCloud);
        } else {
          refIsStrictGroundTruth = true;
        }
      } else {
        var warnRefUnknown = 'Reference cloud assessment is UNKNOWN. Cannot verify whether reference scene is clear over AOI.';
        print('⚠️ WARNING:', warnRefUnknown);
        warnings.push(warnRefUnknown);
      }
    }

    // =====================================================================
    // 10. VISUALIZATION LAYERS (Interactive Map Inspection)
    // =====================================================================

    Map.centerObject(exportRegion, 13);
    Map.setOptions('HYBRID');

    // 1. Target Cloudy Optical RGB
    Map.addLayer(s2CloudyImg.select(['B4', 'B3', 'B2']), { min: 0, max: 3000 }, '1. Target S2 RGB (B4/B3/B2)');

    // 2. Cirrus Band B10 (crucial for inspecting high thin cirrus in L1C)
    Map.addLayer(s2CloudyImg.select('B10'), { min: 0, max: 500, palette: ['black', 'blue', 'white'] }, '2. Target S2 Cirrus (B10)', false);

    // 3. Continuous Cloud Probability (s2cloudless)
    if (targetCloudAssessment.probImage) {
      Map.addLayer(
        targetCloudAssessment.probImage.clip(exportRegion),
        { min: 0, max: 100, palette: ['000000', 'blue', 'yellow', 'red'] },
        '3. Target Cloud Probability (0-100%)',
        false
      );
    }

    // 4. Classified Binary Cloud Mask
    if (targetCloudAssessment.binaryMask) {
      Map.addLayer(
        targetCloudAssessment.binaryMask.updateMask(targetCloudAssessment.binaryMask).clip(exportRegion),
        { palette: ['cyan'] },
        '4. Target Cloud Mask (prob >= ' + targetCloudAssessment.threshold + '%)',
        false
      );
    }

    // 5. Sentinel-1 SAR False Color (VV, VH, VV/VH ratio)
    var sarRgb = s1CoRegistered.select('VV')
      .addBands(s1CoRegistered.select('VH'))
      .addBands(s1CoRegistered.select('VV').subtract(s1CoRegistered.select('VH')).rename('ratio'));
    Map.addLayer(sarRgb, { min: [-25, -30, 0], max: [0, -5, 15] }, '5. Target S1 SAR (VV/VH/ratio)');

    // 6. Reference Optical RGB
    if (hasReference) {
      Map.addLayer(s2RefImg.select(['B4', 'B3', 'B2']), { min: 0, max: 3000 }, '6. Reference S2 RGB', false);
      if (refCloudAssessment && refCloudAssessment.binaryMask) {
        Map.addLayer(
          refCloudAssessment.binaryMask.updateMask(refCloudAssessment.binaryMask).clip(exportRegion),
          { palette: ['magenta'] },
          '7. Reference Cloud Mask (prob >= ' + refCloudAssessment.threshold + '%)',
          false
        );
      }
    }

    // 8. Exact 256x256 Patch Bounding Box
    Map.addLayer(exportRegion, { color: 'red' }, '8. Exact Patch Bounding Box (256x256 @ 10m)');

    // =====================================================================
    // 11. PROVENANCE MANIFEST GENERATION
    // =====================================================================

    var manifest = {
      manifest_version: '1.2.0',
      created_by: 'GeoClear GEE Export Workflow',
      export_timestamp_utc: new Date().toISOString(),
      grid_specification: {
        crs: gridInfo.crs,
        crs_transform: crsTransform,
        affine_parameters: {
          x_scale_meters: CONFIG.pixel_scale_meters,
          x_shear: 0,
          x_origin_upper_left: gridInfo.xMin,
          y_shear: 0,
          y_scale_meters: -CONFIG.pixel_scale_meters,
          y_origin_upper_left: gridInfo.yMax
        },
        pixel_dimensions: [CONFIG.patch_size_pixels, CONFIG.patch_size_pixels],
        width_pixels: CONFIG.patch_size_pixels,
        height_pixels: CONFIG.patch_size_pixels,
        resolution_meters: CONFIG.pixel_scale_meters,
        bounding_box_utm: {
          x_min: gridInfo.xMin,
          y_min: gridInfo.yMin,
          x_max: gridInfo.xMax,
          y_max: gridInfo.yMax
        },
        center_wgs84: [CONFIG.center_lon, CONFIG.center_lat],
        anchoring_method: 'Snapped to nearest 10m UTM grid boundary based on B4 projection'
      },
      sentinel_2_target: {
        collection: 'COPERNICUS/S2_HARMONIZED',
        product_level: 'Level-1C (Top-Of-Atmosphere Reflectance)',
        scene_id: targetSceneId,
        sensing_date_utc: aoiStats.s2_date_str,
        scene_wide_cloud_percentage: s2SceneCloud.getInfo(),
        cloud_assessment: {
          method: targetCloudAssessment.method,
          status: targetCloudAssessment.status,
          valid: targetCloudAssessment.valid,
          classification_threshold_percent: targetCloudAssessment.threshold,
          aoi_mean_cloud_probability_percent: targetCloudAssessment.mean_probability,
          aoi_binary_cloud_pixel_percentage: targetCloudAssessment.binary_cloud_percentage,
          qa60_historical_gap_note: 'QA60 band ignored due to documented 2022-2023 ESA baseline 04.00 zero-filling gap. s2cloudless probability used instead.',
          warning: targetCloudAssessment.warning
        },
        aoi_valid_pixel_percentage: parseFloat(optValidPct),
        band_names: S2_BANDS_13,
        band_order: 'B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12',
        band_count: 13,
        value_semantics: 'Top-Of-Atmosphere Reflectance x 10,000 (DN integers stored as float32)',
        expected_normalization: 'Divide by 10000.0, clip to [0.0, 1.0]',
        resampling_applied: 'Bilinear interpolation to 10m grid. 20m and 60m bands (especially B10) are smoothly upsampled without true sub-resolution physical detail.'
      },
      sentinel_1_sar: {
        collection: 'COPERNICUS/S1_GRD',
        instrument_mode: 'IW',
        scene_id: s1Id.getInfo(),
        sensing_date_utc: aoiStats.s1_date_str,
        temporal_offset_hours: parseFloat(aoiStats.s1_temporal_offset_hours.toFixed(2)),
        aoi_valid_pixel_percentage: parseFloat(sarValidPct),
        polarizations: ['VV', 'VH'],
        polarization_order: 'VV, VH',
        band_count: 2,
        value_semantics: 'Sigma0 backscatter coefficient in decibels (dB), radiometrically terrain-corrected',
        expected_normalization: 'Clip to [-25.0, 0.0] dB, min-max normalize via (x + 25.0) / 25.0 to [0.0, 1.0]. No additional log conversion.',
        resampling_applied: 'Bilinear interpolation to 10m UTM grid'
      },
      reference_optical: {
        available: hasReference,
        collection: hasReference ? 'COPERNICUS/S2_HARMONIZED' : null,
        scene_id: refSceneId,
        sensing_date_utc: hasReference ? aoiStats.ref_date_str : null,
        temporal_difference_days: (hasReference && aoiStats.ref_temporal_diff_days !== undefined && aoiStats.ref_temporal_diff_days !== null) ? parseFloat(Number(aoiStats.ref_temporal_diff_days).toFixed(1)) : null,
        cloud_assessment: hasReference ? {
          method: refCloudAssessment.method,
          status: refCloudAssessment.status,
          valid: refCloudAssessment.valid,
          classification_threshold_percent: refCloudAssessment.threshold,
          aoi_mean_cloud_probability_percent: refCloudAssessment.mean_probability,
          aoi_binary_cloud_pixel_percentage: refCloudAssessment.binary_cloud_percentage,
          warning: refCloudAssessment.warning
        } : null,
        is_strict_ground_truth: refIsStrictGroundTruth,
        scientific_caveat: 'Reference is a temporal proxy observation, NOT simultaneous ground truth. Phenology, seasonal vegetation changes, soil moisture, and farming operations between sensing dates mean differences reflect real land changes in addition to reconstruction errors.'
      },
      // Convenience aliases matching GeoClear ingestion parser
      s2_sensing_date: aoiStats.s2_date_str,
      s1_sensing_date: aoiStats.s1_date_str,
      temporal_offset_hours: parseFloat(aoiStats.s1_temporal_offset_hours.toFixed(2)),
      warnings: warnings,
      nodata_policy: 'Pixels outside swath are masked. Masked values must not be filled with zero. ECRformer requires 100% valid finite numbers.'
    };

    print('================== 3. GEOCLEAR PROVENANCE MANIFEST ==================');
    print('Copy the entire JSON block below into data/real_patch/manifest.json:');
    print(JSON.stringify(manifest, null, 2));
    print('=====================================================================');

    // =====================================================================
    // 12. EXPORT TASKS INITIALIZATION
    // =====================================================================
    // Note: Do NOT pass scale or dimensions when crsTransform and region are given.
    // Earth Engine treats crsTransform, scale, and dimensions as mutually exclusive.

    print('================== 4. EXPORT TASKS INITIALIZATION ==================');

    var optValidNum = parseFloat(optValidPct);
    var sarValidNum = parseFloat(sarValidPct);

    // Mandatory export safeguard: Abort task creation if either sensor has 0.0% coverage
    if (isNaN(optValidNum) || optValidNum <= 0.0 || isNaN(sarValidNum) || sarValidNum <= 0.0) {
      print('❌ [CRITICAL SAFEGUARD TRIGGERED] Export tasks ABORTED!');
      print('Reason: Optical valid-pixel coverage (' + optValidPct + '%) or SAR valid-pixel coverage (' + sarValidPct + '%) is 0.0%.');
      print('Do not export empty rasters. Review projected grid diagnostics and footprint intersections above.');
      print('No export tasks were created in the Tasks tab.');
    } else {
      // 1. Sentinel-1 SAR export
      Export.image.toDrive({
        image: s1CoRegistered.toFloat(),
        description: 's1_sar',
        folder: CONFIG.drive_folder,
        fileNamePrefix: 's1_sar',
        region: exportRegion,
        crs: gridInfo.crs,
        crsTransform: crsTransform,
        maxPixels: 1e9,
        fileFormat: 'GeoTIFF',
        formatOptions: {
          cloudOptimized: true
        }
      });
      print('✓ Export Task created: s1_sar (2 bands: VV, VH in dB on 10m grid)');

      // 2. Sentinel-2 Cloudy Target export
      Export.image.toDrive({
        image: s2CloudySelected.toFloat(),
        description: 's2_cloudy',
        folder: CONFIG.drive_folder,
        fileNamePrefix: 's2_cloudy',
        region: exportRegion,
        crs: gridInfo.crs,
        crsTransform: crsTransform,
        maxPixels: 1e9,
        fileFormat: 'GeoTIFF',
        formatOptions: {
          cloudOptimized: true
        }
      });
      print('✓ Export Task created: s2_cloudy (13 bands: B1-B12 in DN [0, 10000] on 10m grid)');

      // 3. Sentinel-2 Reference export (if available and requested)
      if (hasReference) {
        Export.image.toDrive({
          image: s2RefSelected.toFloat(),
          description: 's2_reference',
          folder: CONFIG.drive_folder,
          fileNamePrefix: 's2_reference',
          region: exportRegion,
          crs: gridInfo.crs,
          crsTransform: crsTransform,
          maxPixels: 1e9,
          fileFormat: 'GeoTIFF',
          formatOptions: {
            cloudOptimized: true
          }
        });
        print('✓ Export Task created: s2_reference (13 bands clean comparison scene)');
      } else {
        print('ℹ️ Reference export omitted (no clean reference scene or export_reference disabled)');
      }

      print('=====================================================================');
      print('Tasks created! Open the "Tasks" tab (top-right panel) and click "Run" on each task.');
    }
  }
}
