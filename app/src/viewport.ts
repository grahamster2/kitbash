/**
 * three.js viewport for a single generated part.
 *
 * Meshes off the generator are untextured and often not watertight, so the
 * setup is tuned for reading *shape*: an environment map for form, a key light
 * for edges, double-sided materials so holes don't turn into black voids, and a
 * grid to anchor scale.
 */
import {
  ACESFilmicToneMapping,
  AmbientLight,
  Box3,
  Color,
  DirectionalLight,
  DoubleSide,
  GridHelper,
  Group,
  Mesh,
  MeshStandardMaterial,
  PMREMGenerator,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  Sphere,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export interface MeshStats {
  triangles: number;
  size: [number, number, number];
}

export class Viewport {
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera: PerspectiveCamera;
  private controls: OrbitControls;
  private grid: GridHelper;
  private loader = new GLTFLoader();
  private current: Group | null = null;

  constructor(private container: HTMLElement) {
    this.renderer = new WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.95;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new Color(0x14161a);

    // RoomEnvironment ships with three, so a decent IBL costs no extra asset
    // download and no extra dependency.
    const pmrem = new PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    const key = new DirectionalLight(0xffffff, 1.5);
    key.position.set(3, 5, 4);
    this.scene.add(key);
    const rim = new DirectionalLight(0x88aaff, 0.7);
    rim.position.set(-4, 2, -3);
    this.scene.add(rim);
    this.scene.add(new AmbientLight(0xffffff, 0.15));

    this.grid = new GridHelper(2, 20, 0x4a5568, 0x2a2f38);
    this.scene.add(this.grid);

    this.camera = new PerspectiveCamera(45, 1, 0.01, 1000);
    this.camera.position.set(1.6, 1.2, 1.9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  clear() {
    if (!this.current) return;
    this.scene.remove(this.current);
    this.current.traverse((o) => {
      if (o instanceof Mesh) {
        o.geometry.dispose();
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => m.dispose());
      }
    });
    this.current = null;
  }

  /** Parses GLB bytes and swaps them in as the only visible model. */
  async load(glb: ArrayBuffer): Promise<MeshStats> {
    const gltf = await this.loader.parseAsync(glb, "");
    this.clear();

    const root = gltf.scene;
    let triangles = 0;
    root.traverse((o) => {
      if (!(o instanceof Mesh)) return;
      const geo = o.geometry;
      triangles += (geo.index ? geo.index.count : geo.attributes.position.count) / 3;
      const mat = o.material as MeshStandardMaterial;
      // Generated shells are frequently open; single-sided rendering reads as
      // missing geometry rather than as a hole.
      mat.side = DoubleSide;
      // Shape-only meshes export as pure white, which blows out under IBL and
      // hides exactly the surface detail you are trying to judge.
      if (!mat.map && mat.color.getHex() === 0xffffff) mat.color.setHex(0x8d95a5);
      mat.roughness = Math.min(mat.roughness ?? 1, 0.6);
      mat.metalness = 0;
      mat.envMapIntensity = 0.9;
    });

    const box = new Box3().setFromObject(root);
    const size = box.getSize(new Vector3());
    const center = box.getCenter(new Vector3());
    // Sit the part on the grid and centre it, so every load frames the same way
    // regardless of where the generator put the origin.
    root.position.set(-center.x, -box.min.y, -center.z);

    this.scene.add(root);
    this.current = root;
    this.frame(box);

    return { triangles: Math.round(triangles), size: [size.x, size.y, size.z] };
  }

  private frame(box: Box3) {
    const sphere = box.getBoundingSphere(new Sphere());
    const radius = Math.max(sphere.radius, 0.001);
    const height = box.max.y - box.min.y;

    const target = new Vector3(0, height / 2, 0);
    const distance = (radius / Math.sin((this.camera.fov * Math.PI) / 360)) * 1.35;
    const dir = new Vector3(0.9, 0.55, 1).normalize();
    this.camera.position.copy(dir.multiplyScalar(distance).add(target));
    this.camera.near = distance / 100;
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();

    this.controls.target.copy(target);
    this.controls.minDistance = radius * 0.2;
    this.controls.maxDistance = distance * 8;
    this.controls.update();

    this.resizeGrid(radius);
  }

  /** Grid divisions stay ~1 cell per 10% of the model so it reads as a ruler. */
  private resizeGrid(radius: number) {
    const span = Math.max(2 * radius * 2.5, 0.1);
    this.scene.remove(this.grid);
    this.grid.dispose();
    this.grid = new GridHelper(span, 20, 0x4a5568, 0x2a2f38);
    this.scene.add(this.grid);
  }

  setWireframe(on: boolean) {
    this.current?.traverse((o) => {
      if (o instanceof Mesh) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => ((m as MeshStandardMaterial).wireframe = on));
      }
    });
  }

  setAutoRotate(on: boolean) {
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = 1.2;
  }
}
