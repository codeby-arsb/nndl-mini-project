/**
 * ColorAI — Draggable Image Comparison Slider Logic
 */
class ImageComparisonSlider {
  constructor(containerElement) {
    this.container = containerElement;
    this.beforeWrap = this.container.querySelector('.slider-img-before-wrap');
    this.divider = this.container.querySelector('.slider-divider');
    this.handle = this.container.querySelector('.slider-handle');
    this.beforeImg = this.container.querySelector('.slider-img-before');
    this.afterImg = this.container.querySelector('.slider-img-after');

    this.isDragging = false;
    this.currentPercent = 50;

    this.init();
  }

  init() {
    if (!this.container || !this.beforeWrap || !this.divider) return;

    // Mouse Events
    this.container.addEventListener('mousedown', (e) => this.onStart(e));
    window.addEventListener('mousemove', (e) => this.onMove(e));
    window.addEventListener('mouseup', () => this.onEnd());

    // Touch Events
    this.container.addEventListener('touchstart', (e) => this.onStart(e), { passive: true });
    window.addEventListener('touchmove', (e) => this.onMove(e), { passive: false });
    window.addEventListener('touchend', () => this.onEnd());

    // Sync image dimensions when slider resizes
    this.syncWidths();
    window.addEventListener('resize', () => this.syncWidths());
  }

  syncWidths() {
    if (!this.container || !this.beforeImg) return;
    const containerWidth = this.container.getBoundingClientRect().width;
    if (this.beforeImg) {
      this.beforeImg.style.width = `${containerWidth}px`;
    }
  }

  onStart(e) {
    this.isDragging = true;
    this.syncWidths();
    this.updatePosition(e);
  }

  onMove(e) {
    if (!this.isDragging) return;
    if (e.cancelable && e.type === 'touchmove') {
      e.preventDefault();
    }
    this.updatePosition(e);
  }

  onEnd() {
    this.isDragging = false;
  }

  updatePosition(e) {
    const rect = this.container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;

    let posX = clientX - rect.left;
    if (posX < 0) posX = 0;
    if (posX > rect.width) posX = rect.width;

    const percent = Math.max(0, Math.min(100, (posX / rect.width) * 100));
    this.setPercent(percent);
  }

  setPercent(percent) {
    this.currentPercent = percent;
    if (this.beforeWrap) {
      this.beforeWrap.style.width = `${percent}%`;
    }
    if (this.divider) {
      this.divider.style.left = `${percent}%`;
    }
  }

  setImages(beforeSrc, afterSrc) {
    if (this.beforeImg) this.beforeImg.src = beforeSrc;
    if (this.afterImg) this.afterImg.src = afterSrc;
    this.syncWidths();
    this.setPercent(50);
  }
}

window.ImageComparisonSlider = ImageComparisonSlider;
