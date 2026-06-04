import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { pixelsToNdc } from './placement.js';

/**
 * MVP: orthographic screen-space earring at earlobe pixel (fixed Z).
 */
export class EarringScene {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {string} modelUrl
   */
  constructor(canvas, modelUrl) {
    this.canvas = canvas;
    this.modelUrl = modelUrl;
    this.mesh = null;
    this.hookOffset = new THREE.Vector3(0, 0.05, 0);

    this.scene = new THREE.Scene();
    const aspect = 1;
    const h = 1;
    this.camera = new THREE.OrthographicCamera(-h * aspect, h * aspect, h, -h, 0.1, 10);
    this.camera.position.z = 2;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: false,
    });
    this.renderer.setClearColor(0x000000, 0);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(0.5, 1, 1);
    this.scene.add(dir);

    this.loader = new GLTFLoader();
  }

  async load() {
    try {
      const gltf = await this.loader.loadAsync(this.modelUrl);
      this.mesh = gltf.scene;
      this.mesh.visible = false;
      this.scene.add(this.mesh);
    } catch (e) {
      console.warn('No earring model; using placeholder sphere.', e);
      const geo = new THREE.SphereGeometry(0.08, 16, 16);
      const mat = new THREE.MeshStandardMaterial({ color: 0xffd700, metalness: 0.8, roughness: 0.2 });
      this.mesh = new THREE.Mesh(geo, mat);
      this.mesh.visible = false;
      this.scene.add(this.mesh);
    }
  }

  resize(width, height) {
    this.renderer.setSize(width, height, false);
    const aspect = width / height;
    const h = 1;
    this.camera.left = -h * aspect;
    this.camera.right = h * aspect;
    this.camera.top = h;
    this.camera.bottom = -h;
    this.camera.updateProjectionMatrix();
  }

  /**
   * @param {{ visible: boolean, x: number, y: number, scale: number }} state - screen pixels
   */
  render(state) {
    if (!this.mesh) return;
    this.mesh.visible = state.visible;
    if (!state.visible) {
      this.renderer.render(this.scene, this.camera);
      return;
    }

    const { width, height } = this.canvas;
    const ndc = pixelsToNdc(state.x, state.y, width, height);
    this.mesh.position.set(ndc.x, ndc.y, 0);
    this.mesh.scale.setScalar(state.scale * 0.25);
    this.mesh.position.sub(this.hookOffset.clone().multiplyScalar(state.scale * 0.25));

    this.renderer.render(this.scene, this.camera);
  }
}
