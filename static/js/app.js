/**
 * GeoClear Frontend Application Logic
 * Reactive Single Page Architecture with REST API Communication
 */

document.addEventListener('DOMContentLoaded', () => {

  // Global State in Frontend
  const state = {
    activeView: 'home',
    activeIngestTab: 'tab-local',
    selectedModelId: 'ecrformer',
    comparisonMode: 'cloudy-recon',
    currentPreviews: {
      cloudy: null,
      sar: null,
      recon: null,
      ref: null,
    },
    activeBandIndex: 3, // Default to B4 (Red)
    bandsList: [],
    sliderPos: 50, // Percentage
    isDraggingSlider: false,
  };

  // DOM Elements
  const navItems = document.querySelectorAll('.nav-item');
  const viewSections = document.querySelectorAll('.view-section');
  const statusDot = document.getElementById('status-dot');
  const headerDatasetLabel = document.getElementById('header-dataset-label');
  const headerModelLabel = document.getElementById('header-model-label');
  const inferenceModelIndicator = document.getElementById('inference-model-indicator');

  // Comparison Slider Elements
  const sliderContainer = document.getElementById('slider-container');
  const compHandle = document.getElementById('comp-handle');
  const compOverlay = document.getElementById('comp-overlay');
  const compBaseImg = document.getElementById('comp-base-img');
  const compOverlayImg = document.getElementById('comp-overlay-img');
  const badgeLeftLabel = document.getElementById('badge-left-label');
  const badgeRightLabel = document.getElementById('badge-right-label');

  // Preview Grid Elements
  const gridCloudyImg = document.getElementById('grid-cloudy-img');
  const gridSarImg = document.getElementById('grid-sar-img');
  const gridReconImg = document.getElementById('grid-recon-img');
  const gridRefImg = document.getElementById('grid-ref-img');
  const refCaptionText = document.getElementById('ref-caption-text');

  // Telemetry HUD Elements
  const telemetryHud = document.getElementById('telemetry-hud');
  const hudLatency = document.getElementById('hud-latency');
  const hudDevice = document.getElementById('hud-device');
  const hudRange = document.getElementById('hud-range');
  const hudFinite = document.getElementById('hud-finite');
  const inferenceSpinner = document.getElementById('inference-spinner');
  const runArtifactsNotif = document.getElementById('run-artifacts-notification');
  const savedPathsList = document.getElementById('saved-paths-list');

  // Metrics Elements
  const kpiMae = document.getElementById('kpi-mae');
  const kpiRmse = document.getElementById('kpi-rmse');
  const kpiPsnr = document.getElementById('kpi-psnr');
  const kpiSsim = document.getElementById('kpi-ssim');
  const perBandTableBody = document.getElementById('per-band-table-body');
  const metricsUnavailNotice = document.getElementById('metrics-unavail-notice');
  const metricsKpiGrid = document.getElementById('metrics-kpi-grid');
  const perBandErrorSection = document.getElementById('per-band-error-section');

  // Band Explorer Elements
  const bandPillsBar = document.getElementById('band-pills-bar');
  const bandPreviewImg = document.getElementById('band-preview-img');
  const bandSpecTitle = document.getElementById('band-spec-title');
  const bandSpecDesc = document.getElementById('band-spec-desc');
  const bandSpecWl = document.getElementById('band-spec-wl');
  const bandSpecGsd = document.getElementById('band-spec-gsd');
  const bandSpecMin = document.getElementById('band-spec-min');
  const bandSpecMax = document.getElementById('band-spec-max');
  const bandSpecMean = document.getElementById('band-spec-mean');


  // =========================================================================
  // 1. Navigation Controller
  // =========================================================================
  function navigateTo(viewId) {
    state.activeView = viewId;
    navItems.forEach(item => {
      if (item.getAttribute('data-view') === viewId) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    viewSections.forEach(section => {
      if (section.id === `view-${viewId}`) {
        section.classList.add('active');
      } else {
        section.classList.remove('active');
      }
    });

    // If navigating to visuals or metrics, refresh bands explorer if result exists
    if ((viewId === 'visuals' || viewId === 'metrics') && state.currentPreviews.recon) {
      loadBandExplorer();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navigateTo(item.getAttribute('data-view'));
    });
  });


  // =========================================================================
  // 2. Ingestion Tabs Controller
  // =========================================================================
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      state.activeIngestTab = targetId;

      tabBtns.forEach(b => b.classList.toggle('active', b === btn));
      tabPanes.forEach(pane => {
        pane.style.display = (pane.id === targetId) ? 'block' : 'none';
      });
    });
  });


  // =========================================================================
  // 3. API Actions: Ingestion
  // =========================================================================

  // Inspect Local Folder
  const btnInspectLocal = document.getElementById('btn-inspect-local');
  const localInspectResults = document.getElementById('local-inspect-results');
  const inspectValidBadge = document.getElementById('inspect-valid-badge');
  const inspectFilesGrid = document.getElementById('inspect-files-grid');
  const inspectManifestDetails = document.getElementById('inspect-manifest-details');

  if (btnInspectLocal) {
    btnInspectLocal.addEventListener('click', async () => {
      const folder = document.getElementById('input-local-dir').value.trim();
      btnInspectLocal.disabled = true;
      btnInspectLocal.textContent = 'Inspecting...';

      try {
        const formData = new FormData();
        formData.append('folder_path', folder);
        const res = await fetch('/api/ingest/inspect_local', { method: 'POST', body: formData });
        const data = await res.json();

        localInspectResults.style.display = 'block';
        if (data.is_valid) {
          inspectValidBadge.className = 'tag-badge';
          inspectValidBadge.style.color = '#34d399';
          inspectValidBadge.textContent = '✅ Metadata Verified';
        } else {
          inspectValidBadge.className = 'tag-badge';
          inspectValidBadge.style.color = '#f87171';
          inspectValidBadge.textContent = '❌ Validation Issues';
        }

        inspectFilesGrid.innerHTML = `
          <div style="font-size: 0.8rem; padding: 6px; background: rgba(0,0,0,0.3); border-radius: 4px;">
            <div style="color: var(--text-dim);">SAR File (2-band):</div>
            <strong style="color: ${data.sar_file ? '#38bdf8' : '#f87171'}">${data.sar_file || 'Missing'}</strong>
          </div>
          <div style="font-size: 0.8rem; padding: 6px; background: rgba(0,0,0,0.3); border-radius: 4px;">
            <div style="color: var(--text-dim);">Optical File (13-band):</div>
            <strong style="color: ${data.opt_file ? '#38bdf8' : '#f87171'}">${data.opt_file || 'Missing'}</strong>
          </div>
          <div style="font-size: 0.8rem; padding: 6px; background: rgba(0,0,0,0.3); border-radius: 4px;">
            <div style="color: var(--text-dim);">Clear Reference:</div>
            <strong style="color: ${data.ref_file ? '#a5b4fc' : '#94a3b8'}">${data.ref_file || 'None (Evaluation Disabled)'}</strong>
          </div>
        `;

        if (data.manifest) {
          inspectManifestDetails.innerHTML = `
            <strong>Provenance Manifest:</strong> Dataset: <code>${data.manifest.dataset_name}</code> | 
            Location: <code>${data.manifest.location}</code> | CRS: <code>${data.manifest.crs}</code> | 
            Target Cloudy Date: <code>${data.manifest.s2_sensing_date}</code> | 
            Clear Reference Date: <code>${data.manifest.reference_date}</code>
          `;
        } else {
          inspectManifestDetails.innerHTML = '<span style="color: var(--text-dim);">No manifest.csv or manifest.json in folder.</span>';
        }
      } catch (err) {
        alert('Failed to inspect local folder: ' + err.message);
      } finally {
        btnInspectLocal.disabled = false;
        btnInspectLocal.textContent = '🔍 Inspect Folder';
      }
    });
  }

  // Load Validated Local Patch
  const btnLoadLocal = document.getElementById('btn-load-local');
  if (btnLoadLocal) {
    btnLoadLocal.addEventListener('click', async () => {
      const folder = document.getElementById('input-local-dir').value.trim();
      btnLoadLocal.disabled = true;
      btnLoadLocal.textContent = 'Loading...';

      try {
        const formData = new FormData();
        formData.append('folder_path', folder);
        const res = await fetch('/api/ingest/load_local', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
          updateLoadedDataState(data);
          alert(`Successfully loaded real satellite patch: ${data.dataset_title}`);
        } else {
          alert('Error: ' + data.detail);
        }
      } catch (err) {
        alert('Failed to load local patch: ' + err.message);
      } finally {
        btnLoadLocal.disabled = false;
        btnLoadLocal.textContent = '📥 Load Validated Patch';
      }
    });
  }

  // Load Synthetic Demo Patch
  const btnGenSynthetic = document.getElementById('btn-generate-synthetic');
  const btnQuickSynthetic = document.getElementById('btn-quick-load-synthetic');

  async function loadSyntheticPatch() {
    try {
      const res = await fetch('/api/ingest/synthetic', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        updateLoadedDataState(data);
        alert('Loaded multi-band synthetic demonstration patch (128x128)!');
      }
    } catch (err) {
      alert('Error generating synthetic patch: ' + err.message);
    }
  }

  if (btnGenSynthetic) btnGenSynthetic.addEventListener('click', loadSyntheticPatch);
  if (btnQuickSynthetic) btnQuickSynthetic.addEventListener('click', loadSyntheticPatch);

  // Benchmark Presets
  const btnPresetMumbai = document.getElementById('btn-preset-mumbai');
  const btnPresetMunich = document.getElementById('btn-preset-munich');

  if (btnPresetMumbai) {
    btnPresetMumbai.addEventListener('click', async () => {
      document.getElementById('input-local-dir').value = 'data/real_patch';
      btnLoadLocal.click();
    });
  }

  if (btnPresetMunich) {
    btnPresetMunich.addEventListener('click', async () => {
      // In the absence of a separate munich folder, load synthetic with Munich dimensions
      loadSyntheticPatch();
      headerDatasetLabel.textContent = 'Munich Benchmark Demo (256x256)';
    });
  }

  // Browser Upload
  const btnSubmitUpload = document.getElementById('btn-submit-upload');
  if (btnSubmitUpload) {
    btnSubmitUpload.addEventListener('click', async () => {
      const s1 = document.getElementById('upload-sar-file').files[0];
      const s2 = document.getElementById('upload-opt-file').files[0];
      const ref = document.getElementById('upload-ref-file').files[0];

      if (!s1 || !s2) {
        alert('Please select both Sentinel-1 SAR and Sentinel-2 Optical raster files.');
        return;
      }

      btnSubmitUpload.disabled = true;
      btnSubmitUpload.textContent = 'Uploading & Validating...';

      try {
        const formData = new FormData();
        formData.append('sar_file', s1);
        formData.append('opt_file', s2);
        if (ref) formData.append('ref_file', ref);

        const res = await fetch('/api/ingest/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (res.ok && data.status === 'success') {
          updateLoadedDataState(data);
          alert('Successfully uploaded and validated satellite rasters!');
        } else {
          alert('Upload failed: ' + (data.detail || 'Validation error'));
        }
      } catch (err) {
        alert('Error uploading rasters: ' + err.message);
      } finally {
        btnSubmitUpload.disabled = false;
        btnSubmitUpload.textContent = 'Upload & Ingest Rasters';
      }
    });
  }

  // Update App State on Data Load
  function updateLoadedDataState(data) {
    headerDatasetLabel.textContent = data.dataset_title || 'Dataset Active';
    statusDot.className = 'status-indicator ready';

    state.currentPreviews.cloudy = data.cloudy_preview;
    state.currentPreviews.sar = data.sar_preview;
    state.currentPreviews.ref = data.ref_preview;
    state.currentPreviews.recon = null; // Clear prior reconstruction

    // Update Grid Showcase
    if (gridCloudyImg && data.cloudy_preview) gridCloudyImg.src = data.cloudy_preview;
    if (gridSarImg && data.sar_preview) gridSarImg.src = data.sar_preview;
    if (gridRefImg && data.ref_preview) {
      gridRefImg.src = data.ref_preview;
      refCaptionText.textContent = 'Clear optical scene for evaluation (temporally offset proxy; not simultaneous ground truth).';
    } else if (gridRefImg) {
      gridRefImg.src = '';
      refCaptionText.textContent = 'No clear reference provided (Quantitative evaluation metrics disabled).';
    }

    // Update Comparison Slider initial view
    updateComparisonSlider();
  }


  // =========================================================================
  // 4. Model Selection Controller
  // =========================================================================
  const modelCards = document.querySelectorAll('.model-card-item');

  modelCards.forEach(card => {
    card.addEventListener('click', async () => {
      const modelId = card.getAttribute('data-model');
      try {
        const formData = new FormData();
        formData.append('model_id', modelId);
        const res = await fetch('/api/models/select', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
          state.selectedModelId = modelId;
          modelCards.forEach(c => {
            const isSel = c.getAttribute('data-model') === modelId;
            c.classList.toggle('selected', isSel);
            const btn = c.querySelector('.btn-select-model');
            if (btn) {
              btn.textContent = isSel ? 'Active Model' : 'Select Model';
              btn.className = `btn ${isSel ? 'btn-primary' : 'btn-secondary'} btn-sm btn-select-model`;
            }
          });

          const displayName = modelId === 'ecrformer' ? 'ECRformer-Light' : 'DSen2-CR (CARL)';
          headerModelLabel.textContent = displayName;
          if (inferenceModelIndicator) {
            inferenceModelIndicator.innerHTML = `Model: <strong>${displayName}</strong>`;
          }
        }
      } catch (err) {
        console.error('Failed to select model:', err);
      }
    });
  });


  // =========================================================================
  // 5. Reconstruction / Inference Trigger
  // =========================================================================
  const btnRunInference = document.getElementById('btn-run-inference');
  const btnQuickReconstruct = document.getElementById('btn-quick-reconstruct');

  async function executeReconstruction() {
    if (!state.currentPreviews.cloudy) {
      alert('Please load satellite data first before running inference.');
      navigateTo('input');
      return;
    }

    if (btnRunInference) btnRunInference.disabled = true;
    if (btnQuickReconstruct) btnQuickReconstruct.disabled = true;
    if (inferenceSpinner) inferenceSpinner.style.display = 'flex';
    statusDot.className = 'status-indicator running';

    try {
      const formData = new FormData();
      formData.append('model_id', state.selectedModelId);

      const res = await fetch('/api/reconstruct', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok || data.status !== 'success') {
        throw new Error(data.detail || 'Inference failed');
      }

      // Update previews
      state.currentPreviews.recon = data.reconstructed_preview;
      if (gridReconImg) gridReconImg.src = data.reconstructed_preview;

      // Update Telemetry HUD
      if (telemetryHud) telemetryHud.style.display = 'grid';
      if (hudLatency) hudLatency.textContent = `${data.latency_ms} ms`;
      if (hudDevice) hudDevice.textContent = data.device;
      if (hudRange) hudRange.textContent = `[${data.output_min}, ${data.output_max}]`;
      if (hudFinite) hudFinite.textContent = data.finite_valid ? '✅ Valid' : '❌ Error';

      // Update Saved Artifacts Notification
      if (data.saved_paths && runArtifactsNotif && savedPathsList) {
        runArtifactsNotif.style.display = 'block';
        savedPathsList.innerHTML = Object.entries(data.saved_paths)
          .map(([k, v]) => `<div>&bull; <strong>${k}:</strong> ${v}</div>`)
          .join('');
      }

      // Update Metrics
      updateMetricsUI(data.metrics);

      // Refresh comparison slider & Multispectral Band Explorer
      updateComparisonSlider();
      loadBandExplorer();

      statusDot.className = 'status-indicator ready';

      // Automatically navigate to Visual Results
      navigateTo('visuals');

    } catch (err) {
      alert('Inference execution error: ' + err.message);
      statusDot.className = 'status-indicator';
    } finally {
      if (btnRunInference) btnRunInference.disabled = false;
      if (btnQuickReconstruct) btnQuickReconstruct.disabled = false;
      if (inferenceSpinner) inferenceSpinner.style.display = 'none';
    }
  }

  if (btnRunInference) btnRunInference.addEventListener('click', executeReconstruction);
  if (btnQuickReconstruct) btnQuickReconstruct.addEventListener('click', executeReconstruction);


  // =========================================================================
  // 6. Interactive Before / After Comparison Slider
  // =========================================================================
  const compModeBtns = document.querySelectorAll('.comp-mode-btn');

  compModeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      compModeBtns.forEach(b => b.classList.toggle('active', b === btn));
      state.comparisonMode = btn.getAttribute('data-mode');
      updateComparisonSlider();
    });
  });

  function updateComparisonSlider() {
    if (!compBaseImg || !compOverlayImg) return;

    const recon = state.currentPreviews.recon || state.currentPreviews.cloudy;
    compBaseImg.src = recon || '';

    if (state.comparisonMode === 'cloudy-recon') {
      compOverlayImg.src = state.currentPreviews.cloudy || '';
      badgeLeftLabel.textContent = 'Input Cloudy Optical';
      badgeRightLabel.textContent = 'Reconstructed Optical Imagery';
    } else if (state.comparisonMode === 'sar-recon') {
      compOverlayImg.src = state.currentPreviews.sar || '';
      badgeLeftLabel.textContent = 'Sentinel-1 SAR (Radar)';
      badgeRightLabel.textContent = 'Reconstructed Optical Imagery';
    } else if (state.comparisonMode === 'ref-recon') {
      compOverlayImg.src = state.currentPreviews.ref || state.currentPreviews.cloudy || '';
      badgeLeftLabel.textContent = 'Clear Optical Reference (Proxy)';
      badgeRightLabel.textContent = 'Reconstructed Optical Imagery';
    }

    setSliderPosition(state.sliderPos);
  }

  function setSliderPosition(percent) {
    state.sliderPos = Math.max(0, Math.min(100, percent));
    if (compOverlay) compOverlay.style.width = `${state.sliderPos}%`;
    if (compHandle) compHandle.style.left = `${state.sliderPos}%`;
  }

  if (sliderContainer) {
    function handleSliderMove(e) {
      if (!state.isDraggingSlider) return;
      const rect = sliderContainer.getBoundingClientRect();
      const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
      const x = clientX - rect.left;
      const percent = (x / rect.width) * 100;
      setSliderPosition(percent);
    }

    sliderContainer.addEventListener('mousedown', (e) => {
      state.isDraggingSlider = true;
      handleSliderMove(e);
    });

    window.addEventListener('mousemove', handleSliderMove);
    window.addEventListener('mouseup', () => { state.isDraggingSlider = false; });

    sliderContainer.addEventListener('touchstart', (e) => {
      state.isDraggingSlider = true;
      handleSliderMove(e);
    });

    window.addEventListener('touchmove', handleSliderMove);
    window.addEventListener('touchend', () => { state.isDraggingSlider = false; });
  }


  // =========================================================================
  // 7. 13-Band Multispectral Explorer
  // =========================================================================
  async function loadBandExplorer() {
    try {
      const res = await fetch('/api/bands');
      if (!res.ok) return;
      const data = await res.json();
      if (data.status !== 'success' || !data.bands) return;

      state.bandsList = data.bands;
      renderBandPills();
      selectBand(state.activeBandIndex);
    } catch (err) {
      console.warn('Could not load multispectral bands:', err);
    }
  }

  function renderBandPills() {
    if (!bandPillsBar) return;
    bandPillsBar.innerHTML = '';

    state.bandsList.forEach((band, idx) => {
      const pill = document.createElement('div');
      pill.className = `band-pill ${idx === state.activeBandIndex ? 'active' : ''}`;
      pill.textContent = band.band_code;
      pill.title = `${band.band_code}: ${band.description} (${band.wavelength})`;
      pill.addEventListener('click', () => {
        selectBand(idx);
      });
      bandPillsBar.appendChild(pill);
    });
  }

  function selectBand(idx) {
    state.activeBandIndex = idx;
    const band = state.bandsList[idx];
    if (!band) return;

    document.querySelectorAll('.band-pill').forEach((p, i) => {
      p.classList.toggle('active', i === idx);
    });

    if (bandPreviewImg) bandPreviewImg.src = band.image;
    if (bandSpecTitle) bandSpecTitle.textContent = `${band.band_code} — ${band.description}`;
    if (bandSpecDesc) bandSpecDesc.textContent = `Spectral channel: ${band.description} at ${band.wavelength}.`;
    if (bandSpecWl) bandSpecWl.textContent = band.wavelength;
    if (bandSpecGsd) bandSpecGsd.textContent = band.gsd;
    if (bandSpecMin) bandSpecMin.textContent = band.min;
    if (bandSpecMax) bandSpecMax.textContent = band.max;
    if (bandSpecMean) bandSpecMean.textContent = band.mean;
  }


  // =========================================================================
  // 8. Evaluation Metrics Visualizer
  // =========================================================================
  function updateMetricsUI(metrics) {
    if (!metrics) {
      if (metricsUnavailNotice) metricsUnavailNotice.style.display = 'block';
      if (metricsKpiGrid) metricsKpiGrid.style.display = 'none';
      if (perBandErrorSection) perBandErrorSection.style.display = 'none';
      return;
    }

    if (metricsUnavailNotice) metricsUnavailNotice.style.display = 'none';
    if (metricsKpiGrid) metricsKpiGrid.style.display = 'grid';
    if (perBandErrorSection) perBandErrorSection.style.display = 'block';

    if (kpiMae) kpiMae.textContent = metrics.mae;
    if (kpiRmse) kpiRmse.textContent = metrics.rmse;
    if (kpiPsnr) kpiPsnr.textContent = `${metrics.rgb_psnr} dB`;
    if (kpiSsim) kpiSsim.textContent = metrics.rgb_ssim;

    if (perBandTableBody && metrics.per_band_mae) {
      perBandTableBody.innerHTML = '';
      const bandWls = {
        'B1': '443 nm', 'B2': '490 nm', 'B3': '560 nm', 'B4': '665 nm',
        'B5': '705 nm', 'B6': '740 nm', 'B7': '783 nm', 'B8': '842 nm',
        'B8A': '865 nm', 'B9': '945 nm', 'B10': '1375 nm', 'B11': '1610 nm', 'B12': '2190 nm'
      };

      const bandNames = {
        'B1': 'Coastal Aerosol', 'B2': 'Blue', 'B3': 'Green', 'B4': 'Red',
        'B5': 'Red Edge 1', 'B6': 'Red Edge 2', 'B7': 'Red Edge 3', 'B8': 'Broad NIR',
        'B8A': 'Narrow NIR', 'B9': 'Water Vapor', 'B10': 'Cirrus (Tropospheric)', 'B11': 'SWIR 1', 'B12': 'SWIR 2'
      };

      Object.keys(metrics.per_band_mae).forEach(bandKey => {
        const row = document.createElement('tr');
        const bMae = metrics.per_band_mae[bandKey];
        const bRmse = metrics.per_band_rmse ? metrics.per_band_rmse[bandKey] : '--';

        row.innerHTML = `
          <td><strong>${bandKey}</strong></td>
          <td>${bandNames[bandKey] || 'Sentinel-2 Band'}</td>
          <td class="mono">${bandWls[bandKey] || '--'}</td>
          <td class="mono" style="color: ${bandKey === 'B10' ? '#38bdf8' : 'var(--text-main)'}; font-weight: 600;">${bMae}</td>
          <td class="mono">${bRmse}</td>
        `;
        perBandTableBody.appendChild(row);
      });
    }
  }


  // =========================================================================
  // 9. Initial Load on Page Mount
  // =========================================================================
  async function initApp() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      if (data.status === 'ready' && data.has_opt) {
        headerDatasetLabel.textContent = data.dataset_title;
        statusDot.className = 'status-indicator ready';
      } else {
        // Auto-inspect and load default local real patch
        if (btnInspectLocal) {
          btnInspectLocal.click();
        }
      }
    } catch (err) {
      console.warn('Backend not responding yet:', err);
    }
  }

  initApp();

});

