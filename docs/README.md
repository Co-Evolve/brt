# BRT website

This folder is the BRT project site, served by GitHub Pages. It is plain static HTML, CSS and ES modules with no build
step. Each animal has a demo page that runs its MuJoCo model in the browser.

- `index.html`: landing page (goals, environments, quickstart, contributing).
- `brittle-star.html`, `jumping-spider.html`: one demo page per animal.
- `css/style.css`: all styles. Colours are CSS variables, with a dark palette under `prefers-color-scheme: dark`.
- `js/viewer.js`: the generic MuJoCo viewer.
- `js/controllers/<animal>.js`: open-loop controllers that drive the demos.
- `models/<animal>.xml`: self-contained MJCF files loaded by the demos.
- `assets/`: images. `banner.jpg` is a 1600 px copy of the repository banner (`assets/banner.png`), shown on the page
  and used as the social preview (`og:image`, an absolute URL because crawlers ignore relative ones).

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
| `data-model`      | Required. MJCF file, written into MuJoCo's virtual filesystem and compiled there.             |
| `data-controller` | Optional. ES module that drives the actuators (see below). Without it, all controls stay 0.  |
| `data-track-body` | Optional. Body the camera follows horizontally. Defaults to the first body with a free joint. |
| `data-camera`     | Optional. Initial camera offset from the orbit target, in metres (`x,y,z`, MuJoCo is z-up).   |
| `data-target`     | Optional. Fixed orbit target in world coordinates (metres). Without it, the camera follows the tracked body. |
| `data-loop`       | Optional. Reset the simulation every N simulated seconds.                                    |

The viewer steps `mj_step` in real time. It calls the controller once per control step and holds its output in
between, like `env.step` holds an action. The control step comes from the model's custom numeric
`brt_control_timestep` (seconds), which the export script sets to the environment's `config.control_timestep`; models
without it call the controller before every `mj_step`. If the browser cannot keep up, the simulation slows down and the toolbar shows
the achieved real-time factor. It renders the geom types plane, sphere, capsule, ellipsoid, cylinder, box and mesh
from geom groups 0 to 2. Colours come from `geom_rgba`, or from the material when the geom keeps the default rgba. 2D
textures are drawn on planes only, for example the checkerboard floors. Everything else uses its flat colour. Sites,
tendons, height fields and skyboxes are not drawn. Physics stops while the viewer is scrolled out of view.

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
a controller keeps working when the prefix, the number of arms or legs, or the actuator order changes. Both demo
controllers are open loop and assume position actuators (the default morphologies):

- `brittle_star.js`: a rowing gait (the five-armed version of `biorobot/brittle_star/examples/rowing.py`), matching
  `/arm_(\d+)_segment_(\d+)_(in_plane|out_of_plane)/` and scaling its targets to each actuator's `ctrlrange`. It moves
  the disk about 2 m in 10 s.
- `jumping_spider.js`: a port of `biorobot/jumping_spider/examples/scripted_jump.py`, matching
  `leg_<side><pair>_<joint>` and `abdomen_<joint>` names. Its poses are joint angles in radians; keep them in sync with
  the Python script. The spider lands about 0.47 m forward after a 0.4 s flight.

Because the viewer uses the environment's control timestep and the exported models are the environments' own, both
demos reproduce the Python rollouts (`mjc` backend): the brittle star ends within about 1 cm of it after 10 s, and the
spider lands at the same spot.

## Preview locally

```sh
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>. Open the pages over HTTP, not as `file://` URLs: ES modules and `fetch` do not work
from the file system. The first visit downloads about 11 MB of WebAssembly, which the browser then caches.

## Regenerating the models

The files in `models/` are self-contained MJCF. There are no external files: textures are MuJoCo builtins (`flat`,
`checker`, ...), and meshes, if any, are inlined in the XML. They are exported from the Python package with
`scripts/export_web_models.py`, which builds a registered environment with `biorobot.make` (default morphology and
arena), takes the exact spec that environment compiles (options, task sensors and all), moves the free-joint body to
the environment's reset pose, adds the `brt_control_timestep` numeric, and writes `spec.to_xml()`. Before writing, it
checks that the XML reproduces `env.reset` and 50 random-action `env.step`s:

| Model                | Environment                          |
|----------------------|--------------------------------------|
| `brittle_star.xml`   | `brittle_star/undirected_locomotion` |
| `jumping_spider.xml` | `jumping_spider/long_jump`           |

```sh
python scripts/export_web_models.py          # regenerate docs/models/
python scripts/export_web_models.py --check  # exit code 1 if docs/models/ is stale
```

Regenerate the models whenever a morphology, arena or environment default changes, and keep `@mujoco/mujoco` at the
version of the `mujoco` Python package used for the export.

Requirements for an exported model:

- One file, loadable with `mujoco.MjModel.from_xml_path` without any other files.
- Physics options (timestep, density, viscosity, ...) set as in the environment, because the browser runs the XML
  as it is.
- The animal starts at its reset pose in the default `qpos0` (the free-joint body's `pos` and `quat`).
- Actuator names keep the substrings the controller matches on.
- Keep it small: the files are fetched on every page view (now 87 and 116 KB).

## Adding an animal

1. Add the animal's demo environment to `DEMOS` in `scripts/export_web_models.py` and run it.
2. Write `docs/js/controllers/<animal>.js` with the controller interface above.
3. Copy `brittle-star.html` to `<animal-name>.html`. Update the title, the text, the `data-*` attributes and the nav
   links, and keep the import map.
4. Add a card under "Environments" in `index.html`, and the new page to the nav of the other demo pages.
5. Preview locally and check the browser console for errors.
