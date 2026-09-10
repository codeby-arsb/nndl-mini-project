/**
 * ColorAI — Sample Image Gallery Loader
 */
class SampleGallery {
  constructor(options = {}) {
    this.container = document.getElementById('samplesGrid');
    this.onSelect = options.onSelect || (() => {});
    this.samples = [];
    this.selectedSample = null;
    this.init();
  }

  async init() {
    if (!this.container) return;
    try {
      const response = await fetch('/api/samples');
      const data = await response.json();
      if (data.success && data.samples.length > 0) {
        this.samples = data.samples;
        this.render();
      }
    } catch (err) {
      console.warn('[SampleGallery] Failed to fetch sample demo images:', err);
    }
  }

  render() {
    this.container.innerHTML = '';
    this.samples.forEach((sample) => {
      const card = document.createElement('div');
      card.className = 'sample-card';
      card.dataset.sampleId = sample.id;
      card.innerHTML = `
        <img class="sample-thumb" src="${sample.url}" alt="${sample.title}" loading="lazy" />
        <div class="sample-name">${sample.title}</div>
      `;

      card.addEventListener('click', () => {
        this.selectSample(sample, card);
      });

      this.container.appendChild(card);
    });
  }

  selectSample(sample, cardElement) {
    // Toggle active state
    this.container.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('active'));
    if (cardElement) cardElement.classList.add('active');
    this.selectedSample = sample;

    // Trigger callback
    this.onSelect(sample);
  }

  clearSelection() {
    this.selectedSample = null;
    if (this.container) {
      this.container.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('active'));
    }
  }
}

window.SampleGallery = SampleGallery;
