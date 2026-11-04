# BRT website

This folder is the BRT project site, served by GitHub Pages. It is plain static HTML, CSS and ES modules with no build
step. Each animal has a demo page that runs its MuJoCo model in the browser.

- `index.html`: landing page (goals, environments, quickstart, contributing).
- `brittle-star.html`, `jumping-spider.html`, `seahorse.html`, `swimmer.html`: one demo page per animal.
- `css/style.css`: all styles. Colours are CSS variables, with a dark palette under `prefers-color-scheme: dark`.
- `js/viewer.js`: the generic MuJoCo viewer.
- `js/controllers/<animal>.js`: open-loop controllers that drive the demos.
- `models/<animal>.xml`: MJCF files loaded by the demos, with their mesh files (if any) in `models/<animal>/`.

The site is deployed by `.github/workflows/pages.yml` on every push to `main` that touches `docs/`, and can also be run
by hand from the Actions tab. In the repository settings, under Pages, set the source to "GitHub Actions".

## How the demos work

Physics runs in the official MuJoCo WebAssembly bindings,
[`@mujoco/mujoco`](https://www.npmjs.com/package/@mujoco/mujoco), maintained by Google DeepMind in the
[mujoco repository](https://github.com/google-deepmind/mujoco/tree/main/wasm). We use the single-threaded build
because the multi-threaded one needs cross-origin isolation headers, which GitHub Pages cannot send. Rendering uses
[three.js](https://threejs.org). Both load from jsDelivr through the import map in each demo page, pinned to exact
versions:

| Package          | Version |
|------------------|---------|
| `@mujoco/mujoco` | 3.17.0  |
| `three`          | 0.186.1 |

Keep `@mujoco/mujoco` at the same version as the `mujoco` Python package that exports the models, so that the MJCF
schema matches. To upgrade, change the version in the import map of every demo page.

The viewer (`js/viewer.js`) mounts on every element that has a `data-mujoco-viewer` attribute:

```html
<div data-mujoco-viewer
     data-model="models/brittle_star.xml"
     data-controller="js/controllers/brittle_star.js"
     data-track-body="BrittleStarMorphology/central_disk"
     data-camera="0.9,-1.1,0.8"
     data-loop="20"></div>
```

| Attribute         | Meaning                                                                                      |
|-------------------|----------------------------------------------------------------------------------------------|
| `data-model`      | Required. MJCF file, written into MuJoCo's in-memory filesystem with the files it references and compiled there. |
| `data-controller` | Optional. ES module that drives the actuators (see below). Without it, all controls stay 0.  |
| `data-track-body` | Optional. Body the camera follows horizontally. Defaults to the first body with a free joint. |
| `data-camera`     | Optional. Initial camera offset from the orbit target, in metres (`x,y,z`, MuJoCo is z-up).   |
| `data-target`     | Optional. Fixed orbit target in world coordinates (metres). Without it, the camera follows the tracked body. |
| `data-loop`       | Optional. Reset the simulation every N simulated seconds.                                    |

The viewer steps `mj_step` in real time. It calls the controller once per control step and holds its output in
between, like `env.step` holds an action. The control step comes from the model's custom numeric
`brt_control_timestep` (seconds), which the export script sets to the environment's `config.control_timestep`; models
without it call the controller before every `mj_step`. If the browser cannot keep up, the simulation slows down and the toolbar shows
the achieved real-time factor.

Before compiling, the viewer reads the XML for the `file` attributes of `<mesh>`, `<skin>`, `<hfield>` and `<texture>`
assets, resolves them as MuJoCo does (relative to the XML, under the compiler's `meshdir`, `texturedir` or `assetdir`),
fetches them next to the XML and writes them to the in-memory filesystem under the same relative paths (with
`FS.writeFile`; `MjVFS.addBuffer` copies byte by byte and takes seconds for a few MB of meshes).

It renders the geom types plane, sphere, capsule, ellipsoid, cylinder, box and mesh from geom groups 0 to 2. Meshes use
the compiled `mesh_vert`/`mesh_face` with normals recomputed at a 30° crease angle (MuJoCo's one normal per vertex
smears dark streaks over the flat faces of CAD meshes), and all geoms of a mesh share one GPU geometry. Colours come from `geom_rgba`, or from the
material when the geom keeps the default rgba. 2D textures are drawn on planes only, for example the checkerboard
floors. Everything else uses its flat colour. Spatial tendons of groups 0 to 2 are drawn as cylinders of radius
`tendon_width` between consecutive wrap points (`wrap_xpos`), in one instanced mesh. A tendon driven by an actuator is
tinted from its `tendon_rgba` towards red by the square root of its actuator force relative to the largest force that
actuator can produce (from its gain, control range and force range). Sites, height fields and skyboxes are not drawn.
In models without a floor plane the camera can orbit below the model, and the shadow frustum fits the model's extent.
Physics stops while the viewer is scrolled out of view.

If the model is missing, does not compile, or WebGL or WASM is unavailable, the viewer shows a message instead of a
canvas. A controller that throws is disabled and the error goes to the console. For tests and debugging, the mounted
element exposes `element.viewer` (with `.model`, `.data`, `.paused`) and its `data-state` attribute is `loading`,
`running`, `paused` or `error`.

### Controller modules

```js
// Optional: called after loading and after every reset.
export function init(model, data, actuatorNames) {}

// Required: called once per control step, before its physics steps. `t` is data.time in seconds.
// Return model.nu controls (an array or typed array), or write data.ctrl yourself and return nothing.
export function control(t, model, data, actuatorNames) {
  return new Float64Array(model.nu);
}
```

`model` and `data` are the `MjModel`/`MjData` objects of the WASM bindings. Their arrays are live typed-array views,
for example `model.actuator_ctrlrange` (flattened `nu x 2`) or `data.qpos`. `actuatorNames[i]` is the full name of
actuator `i`, including the morphology prefix. Match actuators by name substring instead of by index or full name, so
a controller keeps working when the prefix, the number of arms or legs, or the actuator order changes. The demo
controllers are open loop:

- `brittle_star.js`: a rowing gait (the five-armed version of `biorobot/brittle_star/examples/rowing.py`), matching
  `/arm_(\d+)_segment_(\d+)_(in_plane|out_of_plane)/` and scaling its targets to each actuator's `ctrlrange`. It moves
  the disk about 2 m in 10 s.
- `jumping_spider.js`: a port of `biorobot/jumping_spider/examples/scripted_jump.py`, matching
  `leg_<side><pair>_<joint>` and `abdomen_<joint>` names. Its poses are joint angles in radians; keep them in sync with
  the Python script. The spider lands about 0.47 m forward after a 0.4 s flight.
- `seahorse.js`: muscle contraction levels set by the page's sliders. It exports a `settings` object (HM dextral and
  sinistral, MVM, contract/relax cycle), which `seahorse.html` imports from the same module URL and updates on input.
  As `contraction` in `biorobot/seahorse/examples/common.py`, a level in [0, 1] maps each actuator from the upper end
  of its `ctrlrange` (relaxed) to the lower end (fully contracted). It matches `/hm_beam_...(dextral|sinistral)` and
  `/mvm_beam_` names.
- `swimmer.js`: a port of `biorobot/swimmer/examples/van_der_pol_cpg.py`, a chain of Van der Pol oscillators that it
  warms up for 40 cycles after every reset and integrates with RK2 at the 2 ms control timestep. It matches
  `segment_<i>_joint` names: joint i follows oscillator i. Like `seahorse.js`, it exports a `settings` object (CPG
  frequency, servo amplitude, reversed wave) that `swimmer.html` updates from its sliders. With the defaults it moves
  the head 1.45 m head first in 10 s.

Because the viewer uses the environment's control timestep and the exported models are the environments' own, the
demos reproduce the Python rollouts (`mjc` backend): the brittle star ends within about 1 cm of it after 10 s, the
spider lands at the same spot, and the seahorse tail's tip ends at the same position (to 0.1 mm after 3 s) for the same constant
contraction.

## Preview locally

```sh
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>. Open the pages over HTTP, not as `file://` URLs: ES modules and `fetch` do not work
from the file system. The first visit downloads about 11 MB of WebAssembly, which the browser then caches. The seahorse page also downloads
3.4 MB of meshes and a 428 KB XML (21 KB gzipped).

## Regenerating the models

The files in `models/` are MJCF. Textures are MuJoCo builtins (`flat`, `checker`, ...). Mesh files are stored next
to the XML in `models/<animal>/` (its `meshdir`), in MuJoCo's binary `.msh` format. They are exported from the Python
package with `scripts/export_web_models.py`, which builds a registered environment with `biorobot.make` (default
morphology and arena, plus the config overrides listed below), takes the exact spec that environment compiles
(options, task sensors and all), moves the free-joint body (if any) to the environment's reset pose, converts its STL
meshes to `.msh` (vertices deduplicated as MuJoCo does when it loads an STL, so the compiled meshes are identical,
at 38% of the size), adds the `brt_control_timestep` numeric, and writes `spec.to_xml()`. Before writing, it checks
that the files reproduce the compiled meshes, `env.reset` and 50 random-action `env.step`s:

| Model                | Environment                          | Overrides             |
|----------------------|--------------------------------------|-----------------------|
| `brittle_star.xml`   | `brittle_star/undirected_locomotion` |                       |
| `jumping_spider.xml` | `jumping_spider/long_jump`           |                       |
| `seahorse.xml`       | `seahorse/tail`                      |                       |
| `swimmer.xml`        | `swimmer/head_first_swimming`        |                       |

```sh
python scripts/export_web_models.py          # regenerate docs/models/
python scripts/export_web_models.py --check  # exit code 1 if docs/models/ is stale
```

Regenerate the models whenever a morphology, arena or environment default changes, and keep `@mujoco/mujoco` at the
version of the `mujoco` Python package used for the export.

Requirements for an exported model:

- One XML file, plus mesh files under `models/<animal>/` that the XML references by relative path. The
  `--check` mode also compares these files and reports stray ones.
- Physics options (timestep, density, viscosity, ...) set as in the environment, because the browser runs the XML
  as it is.
- The animal starts at its reset pose in the default `qpos0` (the free-joint body's `pos` and `quat`).
- Actuator names keep the substrings the controller matches on.
- Keep it small: the files are fetched on every page view (now 87 KB, 116 KB, 428 KB plus 3.4 MB of seahorse
  meshes, and 9 KB).

## Adding an animal

1. Add the animal's demo environment to `DEMOS` in `scripts/export_web_models.py` and run it.
2. Write `docs/js/controllers/<animal>.js` with the controller interface above.
3. Copy `brittle-star.html` to `<animal-name>.html`. Update the title, the text, the `data-*` attributes and the nav
   links, and keep the import map. The import map must come before anything that loads `viewer.js` (no
   `<link rel="modulepreload">` above it): otherwise a cached `viewer.js` can be resolved before the map, and
   `import "three"` fails.
4. Add a card under "Environments" in `index.html`, and the new page to the nav of the other demo pages.
5. Preview locally and check the browser console for errors.
