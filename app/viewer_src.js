import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

const host = document.getElementById("stage");
const listEl = document.getElementById("parts");
const readout = document.getElementById("readout");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(3, 5, 4);
scene.add(key);
const rim = new THREE.DirectionalLight(0x9fc4ff, 1.0);
rim.position.set(-4, 2, -3);
scene.add(rim);
scene.add(new THREE.AmbientLight(0xffffff, 0.35));

const grid = new THREE.GridHelper(4, 24, 0x2a3644, 0x1b232e);
grid.material.transparent = true;
grid.material.opacity = 0.5;
scene.add(grid);

const root = new THREE.Group();
scene.add(root);

let parts = [];
let isolated = null;
let spin = true;
let totalTris = 0;

// Which parts were generated. Everything else came out of the primitive
// library — the distinction is the whole point of the build.
const GENERATED = new Set(window.__GENERATED__ || []);

function tris(mesh) {
  const g = mesh.geometry;
  return g.index ? g.index.count / 3 : g.attributes.position.count / 3;
}

function b64ToBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

new GLTFLoader().parse(b64ToBuffer(window.__MODEL_B64__), "", (gltf) => {
  const model = gltf.scene;
  model.traverse((o) => {
    if (!o.isMesh) return;
    // Every primitive shares one cached material, so tinting one would tint all.
    o.material = o.material.clone();
    parts.push(o);
    totalTris += tris(o);
  });

  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  model.position.sub(center);
  const s = 2.4 / size;
  model.scale.setScalar(s);
  model.position.multiplyScalar(s);
  root.add(model);
  grid.position.y = new THREE.Box3().setFromObject(model).min.y - 0.02;

  camera.position.set(2.0, 1.15, 2.4);
  controls.update();
  buildList();
  setReadout();
  resize();
});

function setReadout(mesh) {
  readout.textContent = mesh
    ? `${mesh.name.replace(/_/g, " ")} — ${Math.round(tris(mesh)).toLocaleString()} triangles`
    : `${parts.length} parts · ${Math.round(totalTris).toLocaleString()} triangles`;
}

function buildList() {
  const groups = [
    { label: "Generated", kind: "gen",
      items: parts.filter((p) => GENERATED.has(p.name)) },
    { label: "Scripted", kind: "scr",
      items: parts.filter((p) => !GENERATED.has(p.name)) },
  ];
  for (const g of groups) {
    if (!g.items.length) continue;
    const head = document.createElement("div");
    head.className = "group";
    head.innerHTML = `<span class="glabel ${g.kind}">${g.label}</span>
      <span class="gnote">${g.items.length} parts</span>`;
    listEl.appendChild(head);
    for (const mesh of g.items) {
      const row = document.createElement("button");
      row.className = `row ${g.kind}`;
      row.innerHTML = `<span>${mesh.name.replace(/_/g, " ")}</span>
        <span class="ptris">${Math.round(tris(mesh)).toLocaleString()}</span>`;
      row.addEventListener("pointerenter", () => {
        if (!isolated) mesh.material.emissive.setHex(0x3a2f14);
      });
      row.addEventListener("pointerleave", () => {
        if (!isolated) mesh.material.emissive.setHex(0x000000);
      });
      row.addEventListener("click", () => toggle(mesh));
      listEl.appendChild(row);
      mesh.userData.row = row;
    }
  }
}

function toggle(mesh) {
  const off = isolated === mesh;
  isolated = off ? null : mesh;
  for (const p of parts) {
    p.visible = !isolated || p === isolated;
    p.material.emissive.setHex(0x000000);
    p.userData.row.classList.toggle("on", !off && p === isolated);
  }
  setReadout(isolated);
}

document.getElementById("showall").addEventListener("click", () => {
  isolated = null;
  for (const p of parts) {
    p.visible = true;
    p.material.emissive.setHex(0x000000);
    p.userData.row.classList.remove("on");
  }
  setReadout();
});

const spinBtn = document.getElementById("spin");
spinBtn.addEventListener("click", () => {
  spin = !spin;
  spinBtn.setAttribute("aria-pressed", String(spin));
  spinBtn.textContent = spin ? "Pause" : "Rotate";
});

function resize() {
  const r = host.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
  spin = false;
  spinBtn.textContent = "Rotate";
  spinBtn.setAttribute("aria-pressed", "false");
}

renderer.setAnimationLoop(() => {
  if (spin) root.rotation.y += 0.0026;
  controls.update();
  renderer.render(scene, camera);
});
