import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const host = document.getElementById("stage");
const listEl = document.getElementById("subjects");
const readout = document.getElementById("readout");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.NoToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// No lights, deliberately. See the material swap in show().

const grid = new THREE.GridHelper(4, 20, 0x2b3a2f, 0x1d2622);
grid.material.transparent = true;
grid.material.opacity = 0.45;
scene.add(grid);

const holder = new THREE.Group();
scene.add(holder);

let spin = true;
let current = null;
const loader = new GLTFLoader();

function b64ToBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function show(entry, row) {
  if (current) {
    holder.remove(current);
    current.traverse((o) => {
      if (!o.isMesh) return;
      o.geometry.dispose();
      if (o.material.map) o.material.map.dispose();
      o.material.dispose();
    });
    current = null;
  }
  for (const el of listEl.children) el.classList.remove("on");
  row.classList.add("on");
  readout.textContent = "loading…";

  loader.parse(b64ToBuffer(window.__MODELS__[entry.key]), "", (gltf) => {
    const model = gltf.scene;
    model.traverse((o) => {
      if (!o.isMesh) return;
      // Unlit, on purpose. The albedo is a back-projected photograph and
      // already carries the reference's own light and shadow. Shading it a
      // second time double-counts: it blew these to white here, and rendered
      // them near-black in the preview endpoint. The reference contact sheet
      // is unlit-albedo for exactly this reason.
      const map = o.material.map;
      // Generated shells are not watertight, so backface culling shreds them.
      o.material = new THREE.MeshBasicMaterial({
        map, side: THREE.DoubleSide, toneMapped: false,
      });
      if (map) map.colorSpace = THREE.SRGBColorSpace;
    });

    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()).length();
    const centre = box.getCenter(new THREE.Vector3());
    model.position.sub(centre);
    const s = 2.2 / size;
    model.scale.setScalar(s);
    model.position.multiplyScalar(s);
    holder.add(model);
    current = model;

    grid.position.y = new THREE.Box3().setFromObject(model).min.y - 0.02;
    holder.rotation.y = 0;
    camera.position.set(1.9, 1.0, 2.2);
    controls.target.set(0, 0, 0);
    controls.update();

    readout.textContent =
      `${entry.name} · ${entry.tris.toLocaleString()} tris · ${entry.seconds}s · IoU ${entry.iou}`;
  });
}

for (const entry of window.__CATALOGUE__) {
  const row = document.createElement("button");
  row.className = `row ${entry.verdict}`;
  row.innerHTML = `<span class="sname">${entry.name}</span>
    <span class="smeta">${entry.seconds}s</span>`;
  row.addEventListener("click", () => show(entry, row));
  listEl.appendChild(row);
}
show(window.__CATALOGUE__[0], listEl.children[0]);

const spinBtn = document.getElementById("spin");
spinBtn.addEventListener("click", () => {
  spin = !spin;
  spinBtn.setAttribute("aria-pressed", String(spin));
  spinBtn.textContent = spin ? "Pause" : "Rotate";
});
if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
  spin = false;
  spinBtn.textContent = "Rotate";
}

function resize() {
  const r = host.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

renderer.setAnimationLoop(() => {
  if (spin) holder.rotation.y += 0.0035;
  controls.update();
  renderer.render(scene, camera);
});
