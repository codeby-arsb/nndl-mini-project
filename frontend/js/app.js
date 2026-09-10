/**
 * ColorAI — Main Application Controller (v1.1)
 * Enhanced with HD Guided Fusion, Real-Time Color Tuning, and In-App Checkpoint Loading
 */
document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewBox = document.getElementById('previewBox');
  const previewThumb = document.getElementById('previewThumb');
  const previewFilename = document.getElementById('previewFilename');
  const previewSize = document.getElementById('previewSize');
  const previewDimensions = document.getElementById('previewDimensions');
  const btnReplace = document.getElementById('btnReplace');
  const btnRemove = document.getElementById('btnRemove');
  const btnColorize = document.getElementById('btnColorize');
  const errorBanner = document.getElementById('errorBanner');
  const errorMessage = document.getElementById('errorMessage');

  // Processing Elements
  const processingContainer = document.getElementById('processingContainer');
  const processingPhase = document.getElementById('processingPhase');
  const pipelineSteps = document.querySelectorAll('.pipeline-step');

  // Results Elements
  const resultsSection = document.getElementById('resultsSection');
  const btnViewSlider = document.getElementById('btnViewSlider');
  const btnViewGrid = document.getElementById('btnViewGrid');
  const sliderWrapper = document.getElementById('sliderWrapper');
  const sideBySideGrid = document.getElementById('sideBySideGrid');
  const gridOrigImg = document.getElementById('gridOrigImg');
  const gridColorizedImg = document.getElementById('gridColorizedImg');

  // Channel Inspector Elements
  const inspectL = document.getElementById('inspectL');
  const inspectA = document.getElementById('inspectA');
  const inspectB = document.getElementById('inspectB');
  const inspectRGB = document.getElementById('inspectRGB');

  // Toolbar & Stats
  const statInferenceTime = document.getElementById('statInferenceTime');
  const statResolution = document.getElementById('statResolution');
  const statDevice = document.getElementById('statDevice');
  const btnDownloadColorized = document.getElementById('btnDownloadColorized');
  const btnDownloadComparison = document.getElementById('btnDownloadComparison');
  const btnReset = document.getElementById('btnReset');

  // Color Calibration & Fine-Tuning Controls
  const inputSaturation = document.getElementById('inputSaturation');
  const valSaturation = document.getElementById('valSaturation');
  const inputTint = document.getElementById('inputTint');
  const valTint = document.getElementById('valTint');
  const inputWarmth = document.getElementById('inputWarmth');
  const valWarmth = document.getElementById('valWarmth');
  const inputDenoise = document.getElementById('inputDenoise');
  const valDenoise = document.getElementById('valDenoise');
  const checkHdMode = document.getElementById('checkHdMode');
  const checkAutoBalance = document.getElementById('checkAutoBalance');
  const btnResetCalibration = document.getElementById('btnResetCalibration');

  // Checkpoint Modal Elements
  const btnOpenCheckpointModal = document.getElementById('btnOpenCheckpointModal');
  const checkpointModal = document.getElementById('checkpointModal');
  const btnCloseCheckpointModal = document.getElementById('btnCloseCheckpointModal');
  const chkDropzone = document.getElementById('chkDropzone');
  const chkFileInput = document.getElementById('chkFileInput');
  const chkStatus = document.getElementById('chkStatus');
  const checkpointBtnText = document.getElementById('checkpointBtnText');

  // Header & System Specs
  const headerStatusText = document.getElementById('headerStatusText');
  const specParams = document.getElementById('specParams');
  const specDevice = document.getElementById('specDevice');
  const specCheckpoint = document.getElementById('specCheckpoint');

  // State
  let currentFile = null;
  let currentSampleId = null;
  let currentResult = null;
  let isProcessing = false;
  let phaseInterval = null;
  let tuningTimeout = null;

  // Initialize Comparison Slider
  const sliderElement = document.getElementById('comparisonSlider');
  const slider = new ImageComparisonSlider(sliderElement);

  // Initialize Sample Gallery
  const sampleGallery = new SampleGallery({
    onSelect: (sample) => {
      loadSample(sample);
    }
  });

  // Fetch System Info on startup
  fetchSystemInfo();

  /* ==========================================================================
     Drag & Drop and File Selection
     ========================================================================== */
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  btnReplace.addEventListener('click', () => fileInput.click());
  btnRemove.addEventListener('click', resetSelection);

  function handleFileSelection(file) {
    clearError();
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
      showError('Please select a valid image file (JPG, PNG, WEBP, or BMP).');
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      showError('Image size exceeds 15MB. Please choose a smaller file.');
      return;
    }

    currentFile = file;
    currentSampleId = null;
    sampleGallery.clearSelection();

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      const img = new Image();
      img.onload = () => {
        previewThumb.src = dataUrl;
        previewFilename.textContent = file.name;
        previewSize.textContent = formatBytes(file.size);
        previewDimensions.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;

        dropzone.style.display = 'none';
        previewBox.style.display = 'block';
        btnColorize.disabled = false;
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }

  function loadSample(sample) {
    clearError();
    currentFile = null;
    currentSampleId = sample.id;

    previewThumb.src = sample.url;
    previewFilename.textContent = `${sample.title} (Demo Sample)`;
    previewSize.textContent = 'Curated Test';
    previewDimensions.textContent = 'Auto (256 × 256)';

    dropzone.style.display = 'none';
    previewBox.style.display = 'block';
    btnColorize.disabled = false;
  }

  function resetSelection() {
    currentFile = null;
    currentSampleId = null;
    fileInput.value = '';
    sampleGallery.clearSelection();

    dropzone.style.display = 'block';
    previewBox.style.display = 'none';
    btnColorize.disabled = true;
    clearError();
  }

  /* ==========================================================================
     Inference Execution
     ========================================================================== */
  btnColorize.addEventListener('click', () => {
    if (isProcessing) return;
    runColorization();
  });

  // Engine Selector
  const engineRadios = document.querySelectorAll('input[name="modelEngine"]');
  const optPretrained = document.getElementById('optPretrained');
  const optCustom = document.getElementById('optCustom');

  engineRadios.forEach((radio) => {
    radio.addEventListener('change', () => {
      if (optPretrained && optCustom) {
        optPretrained.classList.toggle('active', radio.value === 'pretrained');
        optCustom.classList.toggle('active', radio.value === 'custom');
      }
      if (currentResult) {
        runColorization(true);
      }
    });
  });

  function getSelectedEngine() {
    const checked = document.querySelector('input[name="modelEngine"]:checked');
    return checked ? checked.value : 'pretrained';
  }

  function getTuningParams() {
    return {
      engine: getSelectedEngine(),
      mode: checkHdMode.checked ? 'hd' : 'model',
      saturation: parseFloat(inputSaturation.value) / 100.0,
      tint: parseFloat(inputTint.value),
      warmth: parseFloat(inputWarmth.value),
      denoise: parseInt(inputDenoise.value, 10),
      auto_balance: checkAutoBalance.checked
    };
  }

  async function runColorization(isFineTuning = false) {
    isProcessing = true;
    btnColorize.disabled = true;
    clearError();

    if (!isFineTuning) {
      resultsSection.style.display = 'none';
      processingContainer.style.display = 'block';
      startProcessingAnimation();
    }

    const params = getTuningParams();

    try {
      let response;
      if (currentFile) {
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('engine', params.engine);
        formData.append('mode', params.mode);
        formData.append('saturation', params.saturation);
        formData.append('tint', params.tint);
        formData.append('warmth', params.warmth);
        formData.append('denoise', params.denoise);
        formData.append('auto_balance', params.auto_balance);

        response = await fetch('/api/colorize', {
          method: 'POST',
          body: formData
        });
      } else if (currentSampleId) {
        response = await fetch('/api/colorize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample_id: currentSampleId,
            ...params
          })
        });
      } else {
        throw new Error('No image or sample selected.');
      }

      const result = await response.json();
      if (!response.ok || !result.success) {
        throw new Error(result.error || 'Failed to colorize image.');
      }

      // Success!
      currentResult = result;
      displayResults(result, isFineTuning);
    } catch (err) {
      showError(err.message || 'Colorization failed. Please try again.');
    } finally {
      if (!isFineTuning) {
        stopProcessingAnimation();
        processingContainer.style.display = 'none';
      }
      btnColorize.disabled = false;
      isProcessing = false;
    }
  }

  function displayResults(result, isFineTuning = false) {
    // 1. Comparison Slider
    slider.setImages(result.grayscale_image, result.colorized_image);

    // 2. Side-by-Side Grid
    gridOrigImg.src = result.grayscale_image;
    gridColorizedImg.src = result.colorized_image;

    // 3. Channel Inspector
    if (result.channels) {
      inspectL.src = result.channels.l_channel_b64;
      inspectA.src = result.channels.a_channel_b64;
      inspectB.src = result.channels.b_channel_b64;
      inspectRGB.src = result.colorized_image;
    }

    // 4. Stats & Metrics
    if (result.metadata) {
      statInferenceTime.textContent = `${result.metadata.inference_time_ms} ms`;
      statResolution.textContent = `${result.metadata.processed_dimensions[0]} × ${result.metadata.processed_dimensions[1]} px`;
      statDevice.textContent = result.metadata.device_name || result.metadata.device || 'CPU';
    }

    // 5. Show Results & Scroll
    resultsSection.style.display = 'block';
    if (!isFineTuning) {
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  /* ==========================================================================
     Real-Time Color Tuning & Calibration Listeners
     ========================================================================== */
  function scheduleFineTuning() {
    if (!currentResult) return;
    if (tuningTimeout) clearTimeout(tuningTimeout);
    tuningTimeout = setTimeout(() => {
      runColorization(true);
    }, 250);
  }

  inputSaturation.addEventListener('input', () => {
    valSaturation.textContent = `${inputSaturation.value}%`;
    scheduleFineTuning();
  });

  inputTint.addEventListener('input', () => {
    const val = parseInt(inputTint.value, 10);
    valTint.textContent = val > 0 ? `+${val} (Magenta)` : val < 0 ? `${val} (Green)` : '0';
    scheduleFineTuning();
  });

  inputWarmth.addEventListener('input', () => {
    const val = parseInt(inputWarmth.value, 10);
    valWarmth.textContent = val > 0 ? `+${val} (Warm)` : val < 0 ? `${val} (Cool)` : '0';
    scheduleFineTuning();
  });

  inputDenoise.addEventListener('input', () => {
    const val = parseInt(inputDenoise.value, 10);
    valDenoise.textContent = val === 0 ? 'Off' : `Level ${val}`;
    scheduleFineTuning();
  });

  checkHdMode.addEventListener('change', scheduleFineTuning);
  checkAutoBalance.addEventListener('change', scheduleFineTuning);

  btnResetCalibration.addEventListener('click', () => {
    inputSaturation.value = 100;
    valSaturation.textContent = '100%';
    inputTint.value = 0;
    valTint.textContent = '0';
    inputWarmth.value = 0;
    valWarmth.textContent = '0';
    inputDenoise.value = 0;
    valDenoise.textContent = 'Off';
    checkHdMode.checked = true;
    checkAutoBalance.checked = true;
    scheduleFineTuning();
  });

  /* ==========================================================================
     Checkpoint Upload Modal
     ========================================================================== */
  btnOpenCheckpointModal.addEventListener('click', () => {
    checkpointModal.style.display = 'flex';
  });

  btnCloseCheckpointModal.addEventListener('click', () => {
    checkpointModal.style.display = 'none';
  });

  chkDropzone.addEventListener('click', () => chkFileInput.click());

  chkFileInput.addEventListener('change', async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      await uploadCheckpointFile(e.target.files[0]);
    }
  });

  async function uploadCheckpointFile(file) {
    chkStatus.style.display = 'block';
    chkStatus.style.background = 'rgba(139, 92, 246, 0.1)';
    chkStatus.style.color = '#c4b5fd';
    chkStatus.textContent = `Uploading & reloading weights from ${file.name}...`;

    const formData = new FormData();
    formData.append('checkpoint', file);

    try {
      const response = await fetch('/api/upload_checkpoint', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to load checkpoint.');
      }

      chkStatus.style.background = 'rgba(16, 185, 129, 0.15)';
      chkStatus.style.color = '#6ee7b7';
      chkStatus.textContent = `Success! Loaded checkpoint from ${file.name} (Epoch: ${data.data.checkpoint_info.epoch || 'N/A'})`;

      // Update Header Pill
      btnOpenCheckpointModal.classList.add('loaded');
      checkpointBtnText.textContent = `Loaded: ${file.name}`;
      if (specCheckpoint) {
        specCheckpoint.textContent = `Loaded (${file.name})`;
      }

      // If an image is currently shown, re-colorize with the new model
      if (currentFile || currentSampleId) {
        setTimeout(() => {
          checkpointModal.style.display = 'none';
          runColorization(false);
        }, 1200);
      }
    } catch (err) {
      chkStatus.style.background = 'rgba(239, 68, 68, 0.15)';
      chkStatus.style.color = '#fca5a5';
      chkStatus.textContent = `Error: ${err.message}`;
    }
  }

  /* ==========================================================================
     View Controls (Slider vs Side-by-Side Grid)
     ========================================================================== */
  btnViewSlider.addEventListener('click', () => {
    btnViewSlider.classList.add('active');
    btnViewGrid.classList.remove('active');
    sliderWrapper.style.display = 'block';
    sideBySideGrid.style.display = 'none';
    slider.syncWidths();
  });

  btnViewGrid.addEventListener('click', () => {
    btnViewGrid.classList.add('active');
    btnViewSlider.classList.remove('active');
    sliderWrapper.style.display = 'none';
    sideBySideGrid.style.display = 'grid';
  });

  /* ==========================================================================
     Downloads & Reset
     ========================================================================== */
  btnDownloadColorized.addEventListener('click', () => {
    if (!currentResult) return;
    downloadDataUrl(currentResult.colorized_image, 'colorized_image.png');
  });

  btnDownloadComparison.addEventListener('click', () => {
    if (!currentResult) return;
    downloadDataUrl(currentResult.comparison_image, 'colorization_comparison.png');
  });

  btnReset.addEventListener('click', () => {
    resetSelection();
    resultsSection.style.display = 'none';
    currentResult = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  function downloadDataUrl(dataUrl, filename) {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  /* ==========================================================================
     Processing Animation Stages
     ========================================================================== */
  const phases = [
    'Decoding RGB image and resizing to 256×256 tensor...',
    'Converting to CIE Lab representation & extracting L* lightness...',
    'Passing L* through 6-stage U-Net encoder & bottleneck...',
    'Synthesizing a* & b* chrominance channels via skip connections...',
    'Performing HD Guided Luminance Fusion & sRGB Gamut Conversion...'
  ];

  function startProcessingAnimation() {
    let index = 0;
    processingPhase.textContent = phases[0];
    updateSteps(0);

    phaseInterval = setInterval(() => {
      index = (index + 1) % phases.length;
      processingPhase.textContent = phases[index];
      updateSteps(index);
    }, 400);
  }

  function stopProcessingAnimation() {
    if (phaseInterval) clearInterval(phaseInterval);
    pipelineSteps.forEach((s) => s.classList.remove('active'));
  }

  function updateSteps(activeIndex) {
    pipelineSteps.forEach((step, idx) => {
      if (idx === activeIndex) {
        step.classList.add('active');
      } else {
        step.classList.remove('active');
      }
    });
  }

  /* ==========================================================================
     System Info Fetcher
     ========================================================================== */
  async function fetchSystemInfo() {
    try {
      const response = await fetch('/api/info');
      const json = await response.json();
      if (json.success && json.data) {
        const d = json.data;
        if (headerStatusText) {
          headerStatusText.textContent = `U-Net Ready (${d.device_name || d.device})`;
        }
        if (specParams) {
          specParams.textContent = d.parameters_formatted || '29,244,034';
        }
        if (specDevice) {
          specDevice.textContent = `${d.device_name} (${d.device})`;
        }
        if (d.checkpoint_loaded) {
          btnOpenCheckpointModal.classList.add('loaded');
          checkpointBtnText.textContent = `Loaded (${d.checkpoint_info.filename || 'best.pth'})`;
          if (specCheckpoint) {
            specCheckpoint.textContent = `Loaded (${d.checkpoint_info.filename || 'best.pth'})`;
          }
        } else {
          checkpointBtnText.textContent = 'Load .pth Weights';
          if (specCheckpoint) {
            specCheckpoint.textContent = 'Architecture Weights Ready (Upload .pth)';
          }
        }
      }
    } catch (err) {
      console.warn('[ColorAI] Could not fetch system info:', err);
    }
  }

  /* ==========================================================================
     Helpers
     ========================================================================== */
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.style.display = 'flex';
  }

  function clearError() {
    errorMessage.textContent = '';
    errorBanner.style.display = 'none';
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
});
