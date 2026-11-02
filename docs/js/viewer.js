/**
 * Minimal in-browser MuJoCo viewer for the BRT site.
 *
 * Physics runs in the official MuJoCo WebAssembly bindings (@mujoco/mujoco), rendering in three.js.
 * Both are resolved through the import map declared on each demo page, so this file needs no build step.
 *
 * Usage (declarative): any element with a `data-mujoco-viewer` attribute is mounted automatically.
 *   <div data-mujoco-viewer
 *        data-model="models/brittle_star.xml"            MJCF file; it and the mesh/texture/hfield/skin files it
 *                                                          references (relative to it, honouring the compiler's
 *                                                          meshdir/texturedir/assetdir) are loaded into MuJoCo's VFS
 *        data-controller="js/controllers/brittle_star.js" optional ES module exporting control(...)
 *        data-track-body="BrittleStarMorphology/central_disk" optional; default: first free-floating body
 *        data-camera="0.8,-0.8,0.6"                       optional camera offset from the orbit target (metres)
 *        data-target="0.25,0,0.08"                         optional fixed orbit target (world, metres); default: follow
 *                                                          the tracked body horizontally
 *        data-loop="6"                                    optional: reset every N simulated seconds
 *   ></div>
 *
 * Controller module contract (see docs/README.md):
 *   export function control(t, model, data, actuatorNames) -> ArrayLike<number> | void
 *     Called once per control step, before its physics steps. Return `model.nu` controls, or write `data.ctrl`
 *     directly and return nothing. The control step is the model's custom numeric `brt_control_timestep` (written by
 *     scripts/export_web_models.py from the environment's config), so controls are held like `env.step` holds an
 *     action; without it, the controller runs before every mj_step.
 *   export function init(model, data, actuatorNames) -> void   (optional; called after load and every reset)
 *
 * Rendered: geoms of groups 0-2 (all primitive types and meshes, with crease-angle normals) and spatial tendons of
 * groups 0-2. Tendons driven by an actuator are tinted from their rgba towards ACTIVE_TENDON_COLOR by the square root of
 * the actuator force relative to the largest force the actuator can produce.
 *
 * MuJoCo is z-up; the three.js scene uses MuJoCo world coordinates directly (camera.up = +z).
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

/** The MuJoCo module is ~11 MB of WASM, so it is loaded once and shared by every viewer on the page. */
let mujocoPromise = null;
function loadMujoco() {
  if (!mujocoPromise) {
    mujocoPromise = import("@mujoco/mujoco").then((module) => module.default());
  }
  return mujocoPromise;
}

// mjtGeom values; hfield (1) and primitives beyond mesh are not rendered.
const GEOM = { PLANE: 0, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };
const VISIBLE_GROUPS = 3; // MuJoCo's default viewer shows geom and tendon groups 0, 1 and 2.
const INFINITE_PLANE_HALF_SIZE = 50; // MuJoCo planes with size 0 are infinite; draw them this large.
const MAX_FRAME_DT = 1 / 15; // Clamp wall-clock gaps (tab switches, breakpoints) so physics does not jump.
const PHYSICS_BUDGET_MS = 12; // Max physics time per animation frame; beyond that the sim runs slower than real time.
const DEFAULT_RGBA = [0.5, 0.5, 0.5, 1]; // MuJoCo's default geom rgba: if unchanged, the material colour wins.
const TEXROLE_RGB = 1;
const NTEXROLE = 10;
const TEXTURE_2D = 0;
const CONTROL_TIMESTEP_NUMERIC = "brt_control_timestep";
const TRN_TENDON = 3; // mjtTrn
const BIAS_NONE = 0; // mjtBias
const WRAP_PULLEY = -2; // wrap_obj value that separates the branches of a tendon with pulleys.
const ACTIVE_TENDON_COLOR = [0.9, 0.16, 0.12]; // sRGB colour of a tendon pulling with its actuator's maximum force.
/** Faces of a mesh that meet at a smaller angle are shaded smoothly (see buildMeshGeometry). */
const MESH_CREASE_ANGLE = Math.PI / 6;
const FLOOR_SHADOW_EXTENT = 2.5; // m; half-size of the shadow frustum around the orbit target in models with a floor.
/** Model elements whose `file` attribute names an asset, and the compiler attribute that sets their directory. */
const ASSET_DIRECTORIES = { mesh: "meshdir", skin: "meshdir", hfield: "meshdir", texture: "texturedir" };

function parseVector(text, fallback) {
  if (!text) return fallback;
  const values = text.split(",").map(Number);
  return values.length === 3 && values.every(Number.isFinite) ? new THREE.Vector3(...values) : fallback;
}

/** MuJoCo colours are authored in sRGB; three.js works in linear space. */
function srgbColor(r, g, b) {
  return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}

export class MuJoCoViewer {
  constructor(container) {
    this.container = container;
    this.modelUrl = container.dataset.model;
    this.controllerUrl = container.dataset.controller || null;
    this.trackBodyName = container.dataset.trackBody || null;
    this.cameraOffset = parseVector(container.dataset.camera, new THREE.Vector3(1.0, -1.0, 0.7));
    this.fixedTarget = parseVector(container.dataset.target, null);
    this.loopSeconds = Number(container.dataset.loop) || 0;

    this.mujoco = null;
    this.model = null;
    this.data = null;
    this.controller = null;
    this.actuatorNames = [];
    this.bodyGroups = [];
    this.tendons = null;
    this.paused = false;
    this.visible = true;
    this.realtimeFactor = 1;
    this.lastFrameTime = null;
    this.trackedBody = 0;
    this.physicsStepsPerControl = 1;
    this.physicsStep = 0;

    this.buildDom();
    container.viewer = this; // Debug/test hook: lets scripts inspect `viewer.data.time` etc.
  }

  // ---------------------------------------------------------------- DOM

  buildDom() {
    const c = this.container;
    c.classList.add("viewer");
    c.innerHTML = `
      <div class="viewer-stage">
        <div class="viewer-status" role="status" aria-live="polite">Loading MuJoCo&hellip;</div>
      </div>
      <div class="viewer-toolbar">
        <button type="button" class="viewer-pause" disabled aria-pressed="false">Pause</button>
        <button type="button" class="viewer-reset" disabled>Reset</button>
        <span class="viewer-readout" aria-live="off">t = 0.00 s</span>
      </div>`;
    this.stage = c.querySelector(".viewer-stage");
    this.status = c.querySelector(".viewer-status");
    this.pauseButton = c.querySelector(".viewer-pause");
    this.resetButton = c.querySelector(".viewer-reset");
    this.readout = c.querySelector(".viewer-readout");
    this.pauseButton.addEventListener("click", () => this.setPaused(!this.paused));
    this.resetButton.addEventListener("click", () => this.reset());
  }

  setStatus(message, isError = false) {
    this.status.textContent = message;
    this.status.hidden = !message;
    this.status.classList.toggle("is-error", isError);
  }

  fail(message, error) {
    if (error) console.error(error);
    this.setPaused(true);
    this.container.dataset.state = "error";
    this.setStatus(message, true);
  }

  setPaused(paused) {
    this.paused = paused;
    this.pauseButton.textContent = paused ? "Play" : "Pause";
    this.pauseButton.setAttribute("aria-pressed", String(paused));
    if (this.container.dataset.state !== "error" && this.data) {
      this.container.dataset.state = paused ? "paused" : "running";
    }
    this.lastFrameTime = null;
  }

  // ---------------------------------------------------------------- loading

  async start() {
    this.container.dataset.state = "loading";
    if (!this.modelUrl) return this.fail("No model configured for this viewer.");

    let xml;
    try {
      const response = await fetch(this.modelUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      xml = new Uint8Array(await response.arrayBuffer());
    } catch (error) {
      return this.fail(`The model "${this.modelUrl}" is not available yet. The interactive demo will appear here once it is published.`, error);
    }

    try {
      this.mujoco = await loadMujoco();
    } catch (error) {
      return this.fail("Could not load the MuJoCo WebAssembly module. Check your connection or try a recent browser.", error);
    }

    let assets;
    try {
      assets = await this.fetchAssets(xml);
    } catch (error) {
      return this.fail(`The files of the model "${this.modelUrl}" could not be loaded.`, error);
    }

    try {
      this.loadModel(xml, assets);
    } catch (error) {
      return this.fail(`MuJoCo could not compile "${this.modelUrl}": ${error?.message ?? error}`, error);
    }

    if (this.controllerUrl) {
      try {
        this.controller = await import(new URL(this.controllerUrl, document.baseURI).href);
        if (typeof this.controller.control !== "function") throw new Error("missing export `control`");
      } catch (error) {
        this.controller = null;
        console.warn(`Controller "${this.controllerUrl}" unavailable; actuators stay at zero.`, error);
      }
    }

    try {
      this.initRenderer();
    } catch (error) {
      return this.fail("WebGL is not available in this browser, so the simulation cannot be shown.", error);
    }
    this.buildScene();
    this.reset();
    this.setStatus("");
    this.pauseButton.disabled = false;
    this.resetButton.disabled = false;
    this.setPaused(false);
    this.observeVisibility();
    this.renderer.setAnimationLoop((now) => this.frame(now));
  }

  /**
   * Fetches the asset files the XML references (relative to the XML, as MuJoCo resolves them); returns them by that
   * relative path. Builtin textures and inline meshes have no `file` and need nothing.
   */
  async fetchAssets(xmlBytes) {
    const doc = new DOMParser().parseFromString(new TextDecoder().decode(xmlBytes), "application/xml");
    const compiler = doc.querySelector("compiler");
    const paths = new Set();
    for (const [tag, dirAttribute] of Object.entries(ASSET_DIRECTORIES)) {
      const dir = compiler?.getAttribute(dirAttribute) ?? compiler?.getAttribute("assetdir") ?? "";
      for (const element of doc.querySelectorAll(`asset > ${tag}[file]`)) {
        const file = element.getAttribute("file");
        paths.add(dir && !file.startsWith("/") ? `${dir.replace(/\/+$/, "")}/${file}` : file);
      }
    }
    const modelUrl = new URL(this.modelUrl, document.baseURI);
    const entries = await Promise.all(
      [...paths].map(async (path) => {
        const response = await fetch(new URL(path, modelUrl));
        if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
        return [path, new Uint8Array(await response.arrayBuffer())];
      }),
    );
    return new Map(entries);
  }

  /** Compiles the XML in MuJoCo's virtual filesystem, next to its asset files. */
  loadModel(xmlBytes, assets) {
    const mj = this.mujoco;
    const fileName = this.modelUrl.split("/").pop();
    const vfs = new mj.MjVFS();
    try {
      vfs.addBuffer(fileName, xmlBytes);
      for (const [path, bytes] of assets) vfs.addBuffer(path, bytes);
      this.model = mj.MjModel.from_xml_path(fileName, vfs);
    } finally {
      vfs.delete();
    }
    this.data = new mj.MjData(this.model);
    const objActuator = mj.mjtObj.mjOBJ_ACTUATOR.value;
    this.actuatorNames = Array.from({ length: this.model.nu }, (_, i) => mj.mj_id2name(this.model, objActuator, i));
    this.trackedBody = this.findTrackedBody();
    this.physicsStepsPerControl = this.findPhysicsStepsPerControl();
  }

  findPhysicsStepsPerControl() {
    const mj = this.mujoco;
    const m = this.model;
    const id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_NUMERIC.value, CONTROL_TIMESTEP_NUMERIC);
    if (id < 0) return 1;
    return Math.max(1, Math.round(m.numeric_data[m.numeric_adr[id]] / m.opt.timestep));
  }

  findTrackedBody() {
    const mj = this.mujoco;
    const m = this.model;
    if (this.trackBodyName) {
      const id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY.value, this.trackBodyName);
      if (id >= 0) return id;
      console.warn(`Body "${this.trackBodyName}" not found; tracking the first free body instead.`);
    }
    const jointBody = m.jnt_bodyid;
    const jointType = m.jnt_type;
    for (let j = 0; j < m.njnt; j++) {
      if (jointType[j] === mj.mjtJoint.mjJNT_FREE.value) return jointBody[j];
    }
    return m.nbody > 1 ? 1 : 0;
  }

  // ---------------------------------------------------------------- rendering setup

  initRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.domElement.setAttribute("aria-label", "Interactive MuJoCo simulation. Drag to orbit, scroll to zoom.");
    this.renderer.domElement.setAttribute("role", "img");
    this.stage.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    if (this.hasFloor()) this.controls.maxPolarAngle = Math.PI * 0.495; // Keep the camera above the ground plane.

    this.applyTheme();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.applyTheme());
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.resize();
  }

  applyTheme() {
    const css = getComputedStyle(this.container);
    this.scene.background = new THREE.Color(css.getPropertyValue("--viewer-bg").trim() || "#e8edf2");
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.stage;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  hasFloor() {
    const m = this.model;
    for (let g = 0; g < m.ngeom; g++) if (m.geom_type[g] === GEOM.PLANE) return true;
    return false;
  }

  observeVisibility() {
    new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
      this.lastFrameTime = null;
    }).observe(this.container);
  }

  buildScene() {
    const m = this.model;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7280, 1.6));
    this.sun = new THREE.DirectionalLight(0xffffff, 2.2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    // Without a floor (e.g. a fixed-base model), the frustum fits the model for sharper shadows.
    const extent = this.hasFloor() ? FLOOR_SHADOW_EXTENT : Math.min(FLOOR_SHADOW_EXTENT, m.stat.extent);
    Object.assign(this.sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near: 0.1, far: 40 });
    this.sun.shadow.bias = -0.0005;
    this.scene.add(this.sun, this.sun.target);

    this.bodyGroups = Array.from({ length: m.nbody }, () => {
      const group = new THREE.Group();
      this.scene.add(group);
      return group;
    });
    const materials = new Map();
    const meshGeometries = new Map();
    for (let g = 0; g < m.ngeom; g++) {
      const mesh = this.buildGeom(g, materials, meshGeometries);
      if (mesh) this.bodyGroups[m.geom_bodyid[g]].add(mesh);
    }
    this.buildTendons();
  }

  buildGeom(g, materials, meshGeometries) {
    const m = this.model;
    if (m.geom_group[g] >= VISIBLE_GROUPS) return null;
    const type = m.geom_type[g];
    const [sx, sy, sz] = m.geom_size.subarray(3 * g, 3 * g + 3);
    let geometry;
    switch (type) {
      case GEOM.PLANE: {
        const hx = sx > 0 ? sx : INFINITE_PLANE_HALF_SIZE;
        const hy = sy > 0 ? sy : INFINITE_PLANE_HALF_SIZE;
        geometry = new THREE.PlaneGeometry(2 * hx, 2 * hy);
        break;
      }
      case GEOM.SPHERE:
        geometry = new THREE.SphereGeometry(sx, 24, 16);
        break;
      case GEOM.ELLIPSOID:
        geometry = new THREE.SphereGeometry(1, 24, 16).scale(sx, sy, sz);
        break;
      case GEOM.CAPSULE: // three.js capsules and cylinders run along +y, MuJoCo's along +z.
        geometry = new THREE.CapsuleGeometry(sx, 2 * sy, 6, 16).rotateX(Math.PI / 2);
        break;
      case GEOM.CYLINDER:
        geometry = new THREE.CylinderGeometry(sx, sx, 2 * sy, 24).rotateX(Math.PI / 2);
        break;
      case GEOM.BOX:
        geometry = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
        break;
      case GEOM.MESH: {
        const meshId = m.geom_dataid[g];
        if (!meshGeometries.has(meshId)) meshGeometries.set(meshId, this.buildMeshGeometry(meshId));
        geometry = meshGeometries.get(meshId);
        break;
      }
      default:
        return null;
    }
    const material = this.materialFor(g, materials);
    if (!material) return null;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.fromArray(m.geom_pos, 3 * g);
    const q = m.geom_quat.subarray(4 * g, 4 * g + 4);
    mesh.quaternion.set(q[1], q[2], q[3], q[0]);
    mesh.receiveShadow = true;
    mesh.castShadow = type !== GEOM.PLANE && !material.transparent;
    return mesh;
  }

  /**
   * Mesh vertices are stored in the geom frame after compilation, so no extra transform is needed. Normals are
   * recomputed with a crease angle: MuJoCo stores one normal per vertex, which on CAD meshes (long thin triangles that
   * meet at sharp edges) smears dark streaks over flat faces. Faces that meet at less than MESH_CREASE_ANGLE are
   * shaded smoothly, sharper edges stay sharp. Geometries are shared by all geoms of a mesh.
   */
  buildMeshGeometry(meshId) {
    const m = this.model;
    const vert = m.mesh_vert.subarray(3 * m.mesh_vertadr[meshId], 3 * (m.mesh_vertadr[meshId] + m.mesh_vertnum[meshId]));
    const faceStart = 3 * m.mesh_faceadr[meshId];
    const numCorners = 3 * m.mesh_facenum[meshId];
    const face = m.mesh_face.subarray(faceStart, faceStart + numCorners);
    const positions = new Float32Array(3 * numCorners);
    for (let c = 0; c < numCorners; c++) {
      positions.set(vert.subarray(3 * face[c], 3 * face[c] + 3), 3 * c);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return toCreasedNormals(geometry, MESH_CREASE_ANGLE);
  }

  /** One instanced cylinder per straight tendon segment; poses and colours are set in syncTendons. */
  buildTendons() {
    const m = this.model;
    const tendons = [];
    for (let t = 0; t < m.ntendon; t++) {
      const alpha = m.tendon_rgba[4 * t + 3];
      if (m.tendon_group[t] >= VISIBLE_GROUPS || alpha <= 0) continue;
      tendons.push({
        id: t,
        radius: m.tendon_width[t],
        color: srgbColor(...m.tendon_rgba.subarray(4 * t, 4 * t + 3)),
        actuator: -1,
        maxForce: Infinity,
      });
    }
    const byId = new Map(tendons.map((tendon) => [tendon.id, tendon]));
    for (let a = 0; a < m.nu; a++) {
      const tendon = m.actuator_trntype[a] === TRN_TENDON ? byId.get(m.actuator_trnid[2 * a]) : undefined;
      if (!tendon) continue;
      tendon.actuator = a;
      tendon.maxForce = this.maxActuatorForce(a);
    }
    // mj_forward stores up to 2 points per wrap object (entry and exit of wrapping geoms) plus the end sites.
    const capacity = 2 * m.nwrap;
    if (!tendons.length || !capacity) return;
    const mesh = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(1, 1, 1, 8, 1, true),
      new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0 }),
      capacity,
    );
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false; // Its bounding sphere would be that of the unit cylinder.
    mesh.count = 0;
    this.scene.add(mesh);
    this.tendons = { list: tendons, mesh, capacity };
  }

  /** Largest force (in actuator_force units) the actuator can produce, from its gain, bias and force/control ranges. */
  maxActuatorForce(a) {
    const m = this.model;
    const [low, high] = m.actuator_ctrlrange.subarray(2 * a, 2 * a + 2);
    const gain = Math.abs(m.actuator_gainprm[10 * a]);
    // Without a bias the force is gain * ctrl; with a (position-like) bias it is at most gain * the control span.
    let force = m.actuator_biastype[a] === BIAS_NONE ? gain * Math.max(Math.abs(low), Math.abs(high)) : gain * (high - low);
    if (m.actuator_forcelimited[a]) {
      const [fLow, fHigh] = m.actuator_forcerange.subarray(2 * a, 2 * a + 2);
      force = Math.min(force, Math.max(Math.abs(fLow), Math.abs(fHigh)));
    }
    return force > 0 && Number.isFinite(force) ? force : Infinity;
  }

  materialFor(g, cache) {
    const m = this.model;
    const matId = m.geom_matid[g];
    const geomRgba = Array.from(m.geom_rgba.subarray(4 * g, 4 * g + 4));
    const useMaterial = matId >= 0 && geomRgba.every((v, i) => Math.abs(v - DEFAULT_RGBA[i]) < 1e-6);
    const rgba = useMaterial ? Array.from(m.mat_rgba.subarray(4 * matId, 4 * matId + 4)) : geomRgba;
    if (rgba[3] <= 0) return null;

    const texId = matId >= 0 ? m.mat_texid[matId * NTEXROLE + TEXROLE_RGB] : -1;
    const isPlane = m.geom_type[g] === GEOM.PLANE;
    const repeat = texId >= 0 && isPlane ? this.planeTextureRepeat(g, matId) : null;
    const key = `${rgba.join()}|${texId}|${repeat ? `${repeat.x},${repeat.y}` : ""}`;
    if (cache.has(key)) return cache.get(key);

    const material = new THREE.MeshStandardMaterial({
      color: srgbColor(rgba[0], rgba[1], rgba[2]),
      opacity: rgba[3],
      transparent: rgba[3] < 1,
      depthWrite: rgba[3] >= 1,
      roughness: matId >= 0 ? 1 - 0.6 * m.mat_reflectance[matId] : 0.75,
      metalness: 0.0,
    });
    // Textures are only drawn on planes (the arenas' floors); other geoms use their flat colour.
    if (repeat) {
      const texture = this.buildTexture(texId);
      if (texture) {
        texture.repeat.copy(repeat);
        material.map = texture;
      }
    }
    cache.set(key, material);
    return material;
  }

  /** With `texuniform`, MuJoCo repeats a plane texture `texrepeat` times per metre instead of once per plane. */
  planeTextureRepeat(g, matId) {
    const m = this.model;
    const rx = m.mat_texrepeat[2 * matId];
    const ry = m.mat_texrepeat[2 * matId + 1];
    if (!m.mat_texuniform[matId]) return new THREE.Vector2(rx, ry);
    const hx = m.geom_size[3 * g] || INFINITE_PLANE_HALF_SIZE;
    const hy = m.geom_size[3 * g + 1] || INFINITE_PLANE_HALF_SIZE;
    return new THREE.Vector2(2 * rx * hx, 2 * ry * hy);
  }

  buildTexture(texId) {
    const m = this.model;
    if (m.tex_type[texId] !== TEXTURE_2D) return null;
    const width = m.tex_width[texId];
    const height = m.tex_height[texId];
    const channels = m.tex_nchannel[texId];
    const source = m.tex_data.subarray(m.tex_adr[texId], m.tex_adr[texId] + width * height * channels);
    const rgba = new Uint8Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      for (let c = 0; c < 3; c++) rgba[4 * i + c] = source[channels * i + Math.min(c, channels - 1)];
      rgba[4 * i + 3] = channels === 4 ? source[4 * i + 3] : 255;
    }
    const texture = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.magFilter = THREE.NearestFilter; // Keeps checkerboard edges crisp.
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = true;
    texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    texture.needsUpdate = true;
    return texture;
  }

  // ---------------------------------------------------------------- simulation

  reset() {
    const mj = this.mujoco;
    mj.mj_resetData(this.model, this.data);
    mj.mj_forward(this.model, this.data);
    this.physicsStep = 0;
    try {
      this.controller?.init?.(this.model, this.data, this.actuatorNames);
    } catch (error) {
      this.disableController(error);
    }
    this.syncBodies();
    const target = this.fixedTarget ?? this.trackedPosition();
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(this.cameraOffset);
    this.controls.update();
    this.updateSun();
    this.lastFrameTime = null;
    this.render();
  }

  disableController(error) {
    console.error("Controller failed and was disabled; actuators stay at their last values.", error);
    this.controller = null;
  }

  applyControl() {
    if (!this.controller) return;
    try {
      const u = this.controller.control(this.data.time, this.model, this.data, this.actuatorNames);
      if (u) this.data.ctrl.set(u);
    } catch (error) {
      this.disableController(error);
    }
  }

  /** Advances physics to keep pace with the wall clock, within a per-frame compute budget. */
  stepRealtime(frameDt) {
    const mj = this.mujoco;
    const targetTime = this.data.time + frameDt;
    const start = performance.now();
    while (this.data.time < targetTime && performance.now() - start < PHYSICS_BUDGET_MS) {
      if (this.physicsStep % this.physicsStepsPerControl === 0) this.applyControl();
      mj.mj_step(this.model, this.data);
      this.physicsStep += 1;
    }
    const achieved = frameDt > 0 ? 1 - Math.max(0, targetTime - this.data.time) / frameDt : 1;
    this.realtimeFactor = 0.9 * this.realtimeFactor + 0.1 * Math.min(1, achieved);
    if (this.loopSeconds && this.data.time >= this.loopSeconds) this.reset();
  }

  frame(now) {
    if (!this.visible) return;
    const dt = this.lastFrameTime === null ? 0 : Math.min((now - this.lastFrameTime) / 1000, MAX_FRAME_DT);
    this.lastFrameTime = now;
    if (!this.paused && dt > 0) {
      const before = this.trackedPosition();
      this.stepRealtime(dt);
      this.syncBodies();
      if (!this.fixedTarget) this.followTrackedBody(before);
    }
    this.controls.update();
    this.render();
    this.updateReadout();
  }

  syncBodies() {
    const xpos = this.data.xpos;
    const xquat = this.data.xquat;
    for (let b = 0; b < this.bodyGroups.length; b++) {
      const group = this.bodyGroups[b];
      group.position.set(xpos[3 * b], xpos[3 * b + 1], xpos[3 * b + 2]);
      group.quaternion.set(xquat[4 * b + 1], xquat[4 * b + 2], xquat[4 * b + 3], xquat[4 * b]);
    }
    if (this.tendons) this.syncTendons();
  }

  /** Draws every straight segment between consecutive wrap points (as MuJoCo does, skipping pulley breaks). */
  syncTendons() {
    const { list, mesh, capacity } = this.tendons;
    const d = this.data;
    const wrapAdr = d.ten_wrapadr;
    const wrapNum = d.ten_wrapnum;
    const wrapObj = d.wrap_obj;
    const wrapXpos = d.wrap_xpos;
    const force = d.actuator_force;
    const start = new THREE.Vector3();
    const end = new THREE.Vector3();
    const matrix = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const color = new THREE.Color();
    const active = srgbColor(...ACTIVE_TENDON_COLOR);
    const yAxis = new THREE.Vector3(0, 1, 0);
    const direction = new THREE.Vector3();
    let count = 0;
    for (const tendon of list) {
      // The square root makes small forces visible too.
      const level = tendon.actuator >= 0 ? Math.sqrt(Math.min(1, Math.abs(force[tendon.actuator]) / tendon.maxForce)) : 0;
      color.copy(tendon.color).lerp(active, level);
      const adr = wrapAdr[tendon.id];
      for (let j = adr; j < adr + wrapNum[tendon.id] - 1 && count < capacity; j++) {
        if (wrapObj[j] === WRAP_PULLEY || wrapObj[j + 1] === WRAP_PULLEY) continue;
        start.fromArray(wrapXpos, 3 * j);
        end.fromArray(wrapXpos, 3 * (j + 1));
        const length = start.distanceTo(end);
        if (length < 1e-9) continue;
        quaternion.setFromUnitVectors(yAxis, direction.subVectors(end, start).divideScalar(length));
        scale.set(tendon.radius, length, tendon.radius);
        matrix.compose(start.add(end).multiplyScalar(0.5), quaternion, scale);
        mesh.setMatrixAt(count, matrix);
        mesh.setColorAt(count, color);
        count += 1;
      }
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  trackedPosition() {
    return this.bodyGroups[this.trackedBody]?.position.clone() ?? new THREE.Vector3();
  }

  /** Pans camera and orbit target horizontally with the animal; vertical motion (jumps) stays visible. */
  followTrackedBody(before) {
    const delta = this.trackedPosition().sub(before);
    delta.z = 0;
    this.camera.position.add(delta);
    this.controls.target.add(delta);
    this.updateSun();
  }

  /** Keeps the shadow-casting light (and its small shadow frustum) centred on the orbit target. */
  updateSun() {
    const target = this.controls.target;
    this.sun.position.set(target.x - 2, target.y - 3, target.z + 6);
    this.sun.target.position.copy(target);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  updateReadout() {
    const speed = this.realtimeFactor < 0.95 ? ` · ${this.realtimeFactor.toFixed(2)}× real time` : "";
    this.readout.textContent = `t = ${this.data.time.toFixed(2)} s${speed}`;
  }
}

for (const element of document.querySelectorAll("[data-mujoco-viewer]")) {
  new MuJoCoViewer(element).start();
}
