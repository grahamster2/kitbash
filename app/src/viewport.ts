/**
 * three.js viewport for a generated part or an assembled multi-part scene.
 *
 * Two kinds of surface arrive here and they must not be lit the same way.
 *
 * A **generated** part carries a back-projected albedo, and that albedo *is the
 * reference photograph* — it already contains the light and shadow the photo
 * was taken under. Shading it again double-counts: it blew the gallery assets
 * to white and rendered the same meshes near-black through the preview
 * endpoint, and four of the best assets in the ten-subject set were nearly
 * written off because of it. So a mapped generated part renders unlit, exactly
 * as `gallery_src.js` does.
 *
 * A **scripted** part is the opposite case. Its colour is a PBR factor, or a
 * procedural tiling texture with no light baked in at all, and without shading
 * a wall and the floor it meets are one flat silhouette. Those get lit.
 *
 * Both need `DoubleSide`: generated shells are open and not watertight, and
 * winding is inconsistent across the two generators, so backface culling reads
 * as missing geometry. Meshes also arrive with `POSITION` and at most
 * `TEXCOORD_0` — there is **no NORMAL attribute anywhere** — so anything lit
 * has to be flat-shaded off screen-space derivatives.
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
  Material,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PMREMGenerator,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  Sphere,
  Texture,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export interface PartInfo {
  name: string;
  triangles: number;
}

export interface MeshStats {
  triangles: number;
  size: [number, number, number];
  /** One entry per named top-level node — for an assembled scene, per part. */
  parts: PartInfo[];
}

/**
 * How each named part was built, when the caller knows. `generate` is the only
 * value that changes anything: it is what marks an albedo as photographic.
 * Without it the material's own name is the fallback tell — `texturing.py`
 * stamps `kitbash_backprojected`, `materials.py` stamps `kitbash_<family>`.
 */
export type PartModes = Map<string, "generate" | "script" | "mirror">;

// Assembled scenes arrive with the server's own material, single parts with a
// substituted grey, so a coloured tint would read differently per file and a
// glow in the part's own colour vanishes on an already-saturated one. A neutral
// lift washes any base colour toward white and is visible on all of them.
const HIGHLIGHT_LIFT = 0.22;

export class Viewport {
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera: PerspectiveCamera;
  private controls: OrbitControls;
  private grid: GridHelper;
  private loader = new GLTFLoader();
  private current: Group | null = null;
  private parts = new Map<string, Object3D>();

  constructor(private container: HTMLElement) {
    this.renderer = new WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.9;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new Color(0x14161a);

    // RoomEnvironment ships with three, so a decent IBL costs no extra asset
    // download and no extra dependency.
    const pmrem = new PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    // Wound back from the rig this had when every mesh arrived untextured and
    // a substituted mid-grey needed all the light it could get. Scripted parts
    // now carry their own albedo — measured at 0.72 mean sRGB on the stone
    // family — and headroom above it is worth more than brightness: at the old
    // intensities a stone wall renders at 217 of 255 and its surface relief is
    // the first thing to go. Enough key remains for the faces of a box to read
    // apart, which is the whole job of the lit path.
    const key = new DirectionalLight(0xffffff, 1.1);
    key.position.set(3, 5, 4);
    this.scene.add(key);
    const rim = new DirectionalLight(0x88aaff, 0.4);
    rim.position.set(-4, 2, -3);
    this.scene.add(rim);
    this.scene.add(new AmbientLight(0xffffff, 0.12));

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
    // A texture is shared by every material built from the same glTF image, so
    // it is disposed once by identity rather than once per material.
    const maps = new Set<Texture>();
    this.current.traverse((o) => {
      if (o instanceof Mesh) {
        o.geometry.dispose();
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        for (const m of mats) {
          const map = (m as MeshBasicMaterial).map;
          if (map) maps.add(map);
          m.dispose();
        }
      }
    });
    for (const map of maps) map.dispose();
    this.current = null;
    this.parts.clear();
  }

  /**
   * Parses GLB bytes and swaps them in as the only visible model.
   *
   * `modes` says how each named part was built. It is what decides whether a
   * part's albedo is a photograph to be shown as-is or a material to be lit;
   * see the module header. Omit it and each material's own name is used
   * instead, which is right for a single job straight off the generator and a
   * guess for anything assembled elsewhere.
   */
  async load(glb: ArrayBuffer, modes?: PartModes): Promise<MeshStats> {
    const gltf = await this.loader.parseAsync(glb, "");
    this.clear();

    const root = gltf.scene;
    let triangles = 0;

    // assemble.py writes one named top-level node per part; that node name is
    // the handle the rest of the pipeline (and Roblox) uses for the part.
    const parts: PartInfo[] = [];
    for (const child of root.children) {
      if (!child.name) continue;
      let tris = 0;
      child.traverse((o) => {
        if (!(o instanceof Mesh)) return;
        const geo = o.geometry;
        tris += (geo.index ? geo.index.count : geo.attributes.position.count) / 3;
      });
      if (!tris) continue;
      this.parts.set(child.name, child);
      parts.push({ name: child.name, triangles: Math.round(tris) });
    }

    // Walk per part rather than over the whole tree, so each mesh is shaded
    // according to how *its* part was built.
    const shade = (node: Object3D, mode: string | undefined) => {
      node.traverse((o) => {
        if (!(o instanceof Mesh)) return;
        const geo = o.geometry;
        triangles += (geo.index ? geo.index.count : geo.attributes.position.count) / 3;
        o.material = this.materialFor(o.material as Material, geo.attributes.normal !== undefined, mode);
      });
    };
    // A single job's mesh is one node the server named `geometry_0`, not the
    // part name the caller knows it by, so one mode against one part is taken
    // to be about that part whatever either is called.
    const only =
      this.parts.size === 1 && modes?.size === 1 ? [...modes.values()][0] : undefined;
    for (const [name, node] of this.parts) shade(node, modes?.get(name) ?? only);
    // Anything not under a named top-level node — an unnamed single mesh —
    // still has to be shaded, and has only its material name to go on.
    for (const child of root.children) if (!this.parts.has(child.name)) shade(child, undefined);

    const box = new Box3().setFromObject(root);
    const size = box.getSize(new Vector3());
    const center = box.getCenter(new Vector3());
    // Sit the part on the grid and centre it, so every load frames the same way
    // regardless of where the generator put the origin.
    root.position.set(-center.x, -box.min.y, -center.z);

    this.scene.add(root);
    this.current = root;
    // `box` predates the recentring above, so the post-offset target is derived
    // rather than read back off it.
    this.frame(box, new Vector3(0, size.y / 2, 0));

    return { triangles: Math.round(triangles), size: [size.x, size.y, size.z], parts };
  }

  /**
   * The one decision this file exists to get right.
   *
   * A photographic albedo is shown unlit — no lights, no tone mapping, the
   * pixels the camera actually recorded. Everything else is lit, flat-shaded
   * when the mesh brought no normals, so its form reads at all.
   */
  private materialFor(source: Material, hasNormals: boolean, mode?: string): Material {
    const src = source as MeshStandardMaterial;
    const map = src.map ?? null;
    // GLTFLoader gets this right for a baseColorTexture; setting it again costs
    // nothing and covers a map that arrived any other way. Without it the
    // albedo renders washed out — the classic sRGB-as-linear mistake.
    if (map) map.colorSpace = SRGBColorSpace;

    const photographic =
      map !== null &&
      (mode === "generate" ||
        // Standing alone, a mesh's material name is the tell: texturing.py
        // stamps `kitbash_backprojected`. Inside an assembled scene it is not —
        // materials.py keeps the albedo but renames the material to its family
        // — which is why `modes` exists.
        (mode === undefined && /backproject/i.test(src.name)));

    if (photographic) {
      return new MeshBasicMaterial({ map, side: DoubleSide, toneMapped: false });
    }

    const mat = new MeshStandardMaterial({
      map,
      color: src.color ? src.color.clone() : new Color(0xffffff),
      side: DoubleSide,
      // No mesh from either generator or from primitives.py carries a NORMAL
      // attribute, so without this every lit surface samples a zero normal and
      // the part renders black.
      flatShading: !hasNormals,
      roughness: Math.min(src.roughness ?? 1, 0.9),
      metalness: src.metalness ?? 0,
      envMapIntensity: 0.6,
    });
    // Shape-only meshes export as pure white, which blows out under IBL and
    // hides exactly the surface detail you are trying to judge.
    if (!map && mat.color.getHex() === 0xffffff) mat.color.setHex(0x8d95a5);
    return mat;
  }

  /** Hides every other part and reframes on this one; `null` restores all. */
  isolate(name: string | null) {
    if (name !== null && !this.parts.has(name)) return;
    // Box3 updates only the node's own matrix, not its ancestors', so without
    // this an isolate called before the first render frames the part where it
    // sat *before* load() recentred the model.
    this.current?.updateMatrixWorld(true);
    for (const [key, node] of this.parts) node.visible = name === null || key === name;
    if (!this.current) return;
    const node = name === null ? this.current : this.parts.get(name)!;
    const box = new Box3().setFromObject(node);
    this.frame(box, box.getCenter(new Vector3()));
  }

  /** Lifts one part without hiding the rest, for hover feedback off the list. */
  highlight(name: string | null) {
    for (const [key, node] of this.parts) {
      const on = key === name;
      node.traverse((o) => {
        if (!(o instanceof Mesh)) return;
        const mat = o.material as Material;
        if (mat instanceof MeshStandardMaterial) {
          mat.emissive.setScalar(on ? HIGHLIGHT_LIFT : 0);
        } else if (mat instanceof MeshBasicMaterial) {
          // An unlit part has no emissive channel. Its colour is a multiplier
          // over the albedo, so pushing it past white brightens the picture.
          mat.color.setScalar(on ? 1 + HIGHLIGHT_LIFT : 1);
        }
      });
    }
  }

  private frame(box: Box3, target: Vector3) {
    const sphere = box.getBoundingSphere(new Sphere());
    const radius = Math.max(sphere.radius, 0.001);
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
        mats.forEach((m) => ((m as MeshStandardMaterial | MeshBasicMaterial).wireframe = on));
      }
    });
  }

  setAutoRotate(on: boolean) {
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = 1.2;
  }
}
