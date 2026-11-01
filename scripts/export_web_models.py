"""Export the MJCF models of the website demos (``docs/models/<animal>.xml``) from the registered environments.

Each model is the exact spec an environment compiles (arena and morphology composed by the environment, with its
physics options and task sensors applied), with these changes for the browser viewer:

* The free-joint body (if any) starts at the environment's reset pose, so ``mj_resetData`` reproduces ``env.reset``.
* A custom numeric ``brt_control_timestep`` holds the environment's control timestep (s). The viewer holds each
  control for that long, like ``env.step`` does, so the open-loop demos behave as they do in Python.
* Mesh files (STL) are converted to MuJoCo's binary ``.msh`` format (deduplicated vertices, ~2.6x smaller than binary
  STL, same compiled mesh) and written to ``docs/models/<animal>/``, the XML's ``meshdir`` (stray files there are
  removed). The viewer loads them into
  MuJoCo's virtual filesystem next to the XML.

The script checks that the written files reproduce the environment (same initial state, compiled meshes and a matching
rollout), and ``--check`` only verifies that the committed files are up to date (exit code 1 if not).

    python scripts/export_web_models.py [--check]
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import mujoco
import numpy as np

import biorobot
from biorobot.core.env import BRTEnv

MODELS_DIR = Path(__file__).resolve().parents[1] / "docs" / "models"
#: Demo file name -> registered environment that defines its model, and its config overrides. The seahorse demo
#: enables gravity (off by default, as in the original model): without it, nothing straightens the tail when its
#: muscles relax, so the page's contract/relax cycle would leave it curled.
DEMOS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "brittle_star.xml": ("brittle_star/undirected_locomotion", {}),
    "jumping_spider.xml": ("jumping_spider/long_jump", {}),
    "seahorse.xml": ("seahorse/tail", {"enable_gravity": True}),
}
CONTROL_TIMESTEP_NUMERIC = "brt_control_timestep"
#: Control steps compared between the environment and the exported XML, with random actions.
NUM_VALIDATION_STEPS = 50


@contextlib.contextmanager
def _captured_specs() -> Iterator[List[mujoco.MjSpec]]:
    """Record the spec every environment is built from; ``BRTEnv.__init__`` applies the options to it in place."""
    specs: List[mujoco.MjSpec] = []
    original_init = BRTEnv.__init__

    def init(self: BRTEnv, spec: mujoco.MjSpec, *args, **kwargs) -> None:
        specs.append(spec)
        original_init(self, spec, *args, **kwargs)

    BRTEnv.__init__ = init
    try:
        yield specs
    finally:
        BRTEnv.__init__ = original_init


def stl_to_msh(stl: bytes) -> bytes:
    """Convert a binary STL file to MuJoCo's binary ``.msh`` format.

    Vertices are deduplicated in order of first occurrence, as MuJoCo does when it loads an STL file, so both files
    compile to the same mesh. ``.msh`` layout: int32 ``nvertex, nnormal, ntexcoord, nface``, then float32 vertices
    (nvertex x 3) and int32 faces (nface x 3); no normals or texture coordinates.
    """
    (num_faces,) = np.frombuffer(stl, dtype="<u4", count=1, offset=80)
    triangle = np.dtype(
        [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    if len(stl) != 84 + num_faces * triangle.itemsize:
        raise ValueError("Only binary STL files are supported.")
    corners = np.frombuffer(stl, dtype=triangle, count=num_faces, offset=84)[
        "vertices"
    ].reshape(-1, 3)
    unique, first, inverse = np.unique(
        corners, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    vertices = unique[order]
    faces = rank[inverse.ravel()].reshape(-1, 3)
    header = np.array([len(vertices), 0, 0, len(faces)], dtype="<i4")
    return (
        header.tobytes()
        + vertices.astype("<f4").tobytes()
        + faces.astype("<i4").tobytes()
    )


def _externalize_meshes(spec: mujoco.MjSpec, mesh_dir: str) -> Dict[str, bytes]:
    """Point the spec's file meshes to ``.msh`` copies in ``mesh_dir`` (relative to the XML); returns their contents
    by path relative to the XML."""
    files = {}
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        source = Path(mesh.file)
        if source.suffix.lower() != ".stl":
            raise ValueError(
                f"Mesh '{mesh.name}': only STL files can be exported, got {source.name}."
            )
        mesh.file = f"{source.stem}.msh"
        files[f"{mesh_dir}/{mesh.file}"] = stl_to_msh(source.read_bytes())
    if files:
        spec.meshdir = mesh_dir
        spec.assets = files
    return files


def export_model(
    file_name: str, env_name: str, config_overrides: Dict[str, Any]
) -> Tuple[str, Dict[str, bytes]]:
    """The XML of the demo ``file_name`` and its asset files (by path relative to ``MODELS_DIR``)."""
    with _captured_specs() as specs:
        env = biorobot.make(env_name, **config_overrides)
    (spec,) = specs
    initial_qpos = np.asarray(env.reset(0).data.qpos)

    model = env.mj_model
    unchanged = np.ones(model.nq, dtype=bool)
    for free_joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        adr = int(model.jnt_qposadr[free_joint])
        unchanged[adr : adr + 7] = False
        root = spec.body(model.body(model.jnt_bodyid[free_joint]).name)
        root.pos = initial_qpos[adr : adr + 3]
        root.quat = initial_qpos[adr + 3 : adr + 7]
    if not np.allclose(initial_qpos[unchanged], model.qpos0[unchanged]):
        raise ValueError(
            f"{env_name}: the reset changes joints other than free joints; export a keyframe instead."
        )

    assets = _externalize_meshes(spec, mesh_dir=Path(file_name).stem)
    spec.add_numeric(
        name=CONTROL_TIMESTEP_NUMERIC, data=[env.config.control_timestep], size=1
    )
    spec.compile()
    xml = spec.to_xml()
    _validate(env, xml, assets)
    return xml, assets


def _validate(env: BRTEnv, xml: str, assets: Dict[str, bytes]) -> None:
    """The files must reproduce the compiled meshes, ``env.reset`` and ``env.step`` (up to the precision of the XML's
    printed numbers)."""
    model = mujoco.MjModel.from_xml_string(xml, assets)
    np.testing.assert_array_equal(model.mesh_vert, env.mj_model.mesh_vert)
    np.testing.assert_array_equal(model.mesh_face, env.mj_model.mesh_face)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    state = env.reset(0)
    np.testing.assert_allclose(data.qpos, state.data.qpos, atol=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(NUM_VALIDATION_STEPS):
        action = rng.uniform(env.action_low, env.action_high)
        state = env.step(state, action)
        data.ctrl[:] = action
        mujoco.mj_step(model, data, nstep=env.config.num_physics_steps_per_control_step)
    np.testing.assert_allclose(data.xpos, state.data.xpos, atol=1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; fail if a committed model differs from a fresh export.",
    )
    args = parser.parse_args()

    stale = []
    for file_name, (env_name, config_overrides) in DEMOS.items():
        path = MODELS_DIR / file_name
        xml, assets = export_model(file_name, env_name, config_overrides)
        # Files in the model's asset directory, ignoring hidden ones such as macOS's .DS_Store.
        existing = {
            str(p.relative_to(MODELS_DIR))
            for p in (MODELS_DIR / path.stem).glob("[!.]*")
            if p.is_file()
        }
        if args.check:
            current = path.read_text() if path.exists() else ""
            if current != xml:
                stale.append(file_name)
                sys.stdout.writelines(
                    list(
                        difflib.unified_diff(
                            current.splitlines(keepends=True),
                            xml.splitlines(keepends=True),
                            fromfile=str(path),
                            tofile=f"export of {env_name}",
                            n=1,
                        )
                    )[:40]
                )
            for name in sorted(existing | set(assets)):
                asset_path = MODELS_DIR / name
                if (
                    name not in assets
                    or not asset_path.exists()
                    or asset_path.read_bytes() != assets[name]
                ):
                    stale.append(name)
        else:
            path.write_text(xml)
            for name, content in assets.items():
                (MODELS_DIR / name).parent.mkdir(exist_ok=True)
                (MODELS_DIR / name).write_bytes(content)
            for name in existing - set(assets):
                (MODELS_DIR / name).unlink()
            size = len(xml) + sum(len(content) for content in assets.values())
            print(
                f"Wrote {path} and {len(assets)} mesh files ({size / 1024:.0f} KB) from '{env_name}'."
            )
    if stale:
        sys.exit(
            f"Stale web models: {', '.join(stale)}. Run python scripts/export_web_models.py"
        )


if __name__ == "__main__":
    main()
