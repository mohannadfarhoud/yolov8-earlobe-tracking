/** Detect phones/tablets — used for safe defaults (memory + CPU). */
export function isMobileDevice() {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  if (/Android|iPhone|iPad|iPod|Mobile|webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua)) {
    return true;
  }
  return navigator.maxTouchPoints > 1 && window.innerWidth < 1024;
}

/** Camera constraints that keep mobile browsers from running out of memory. */
export function getMobileCameraConstraints() {
  return {
    audio: false,
    video: {
      facingMode: 'user',
      width: { ideal: 480, max: 640 },
      height: { ideal: 640, max: 854 },
      frameRate: { ideal: 15, max: 24 },
    },
  };
}
