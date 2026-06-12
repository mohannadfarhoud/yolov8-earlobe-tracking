/** Side-ear tracker defaults — 4 keypoints per detection. */
export const DEFAULT_SIDE_EAR_CFG = {
  sideEarClassId: 0,
  numClasses: 1,
  numKpts: 4,
  boxConfThreshold: 0.5,
  kptConfThreshold: 0.5,
  nmsIou: 0.45,
  selectionRule: 'highest_conf',
  defaultFps: 30,
  mobileDefaultFps: 15,
  mobileMaxCaptureSide: 384,
  desktopMaxCaptureSide: 960,
  smoothTauMs: 18,
  mobileSmoothTauMs: 27,
  smoothHysteresis: { on: 0.55, off: 0.45 },
};
