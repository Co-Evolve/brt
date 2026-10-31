"""Export the MJCF models of the website demos (``docs/models/<animal>.xml``) from the registered environments.

Each model is the exact spec an environment compiles (arena and morphology composed by the environment, with its
physics options and task sensors applied), with two additions for the browser viewer:

* The free-joint body starts at the environment's reset pose, so ``mj_resetData`` reproduces ``env.reset``.
* A custom numeric ``brt_control_timestep`` holds the environment's control timestep (s). The viewer holds each
  control for that long, like ``env.step`` does, so the open-loop demos behave as they do in Python.

The script checks that the written XML reproduces the environment (same initial state and a matching rollout), and
``--check`` only verifies that the committed files are up to date (exit code 1 if not).

    python scripts/export_web_models.py [--check]
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import sys
from pathlib import Path
from typing import Iterator, List

import mujoco
import numpy as np

import biorobot
from biorobot.core.env import BRTEnv

MODELS_DIR = Path(__file__).resolve().parents[1] / "docs" / "models"
#: Demo file name -> registered environment that defines its model.
DEMOS = {
    "brittle_star.xml": "brittle_star/undirected_locomotion",
    "jumping_spider.xml": "jumping_spider/long_jump",
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


def export_xml(env_name: str) -> str:
    with _captured_specs() as specs:
        env = biorobot.make(env_name)
    (spec,) = specs
    initial_qpos = np.asarray(env.reset(0).data.qpos)

    model = env.mj_model
    (free_joint,) = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    adr = int(model.jnt_qposadr[free_joint])
    others = np.ones(model.nq, dtype=bool)
    others[adr : adr + 7] = False
    if not np.allclose(initial_qpos[others], model.qpos0[others]):
        raise ValueError(
            f"{env_name}: the reset changes joints other than the free joint; export a keyframe instead."
        )
    root = spec.body(model.body(model.jnt_bodyid[free_joint]).name)
    root.pos = initial_qpos[adr : adr + 3]
    root.quat = initial_qpos[adr + 3 : adr + 7]

    spec.add_numeric(
        name=CONTROL_TIMESTEP_NUMERIC, data=[env.config.control_timestep], size=1
    )
    spec.compile()
    xml = spec.to_xml()
    _validate(env, xml)
    return xml


def _validate(env: BRTEnv, xml: str) -> None:
    """The XML must reproduce ``env.reset`` and ``env.step`` (up to the precision of the XML's printed numbers)."""
    model = mujoco.MjModel.from_xml_string(xml)
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
    for file_name, env_name in DEMOS.items():
        path = MODELS_DIR / file_name
        xml = export_xml(env_name)
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
        else:
            path.write_text(xml)
            print(f"Wrote {path} ({len(xml) / 1024:.0f} KB) from '{env_name}'.")
    if stale:
        sys.exit(
            f"Stale web models: {', '.join(stale)}. Run python scripts/export_web_models.py"
        )


if __name__ == "__main__":
    main()
