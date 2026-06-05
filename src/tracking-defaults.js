/** Default thresholds — keep in sync with config/tracking.json */
export const DEFAULT_TRACKING_CFG = {
  leftClassId: 0,
  rightClassId: 1,
  numClasses: 2,
  earlobeKptIndex: 0,
  boxConfThreshold: 0.5,
  kptConfThreshold: 0.5,
  nmsIou: 0.45,
  inferenceIntervalMs: 33,
  mobileInferenceIntervalMs: 100,
  mobileMaxCaptureSide: 384,
  desktopMaxCaptureSide: 960,
  smoothTauMs: 18,
  mobileSmoothTauMs: 27,
  smoothHysteresis: { on: 0.55, off: 0.45 },
};
