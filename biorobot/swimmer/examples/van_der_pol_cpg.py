"""Open-loop swimming with a central pattern generator (CPG): a chain of coupled Van der Pol oscillators.

Every joint ``i`` (head to tail) follows oscillator ``i + 1`` of a chain of ``num_segments`` oscillators,

    x_i'' = mu (1 - x_i^2) x_i' - omega^2 x_i + w_fwd (x_{i-1} - x_i) + w_bwd (x_{i+1} - x_i),   x_0 = x_{N+1} = 0,

integrated with RK2 at the physics timestep and scaled by a linear amplitude envelope along the body. The chain first
runs on its own for 40 cycles, so the swimmer is driven by a CPG on its limit cycle from the start.

    python -m biorobot.swimmer.examples.van_der_pol_cpg --backend mjc --render human
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from biorobot import make
from biorobot.swimmer import default_swimmer_morphology_specification


@dataclass
class VanDerPolChain:
    """A chain of ``num_oscillators`` Van der Pol oscillators with nearest-neighbour coupling and fixed (zero) ends."""

    num_oscillators: int
    mu: float = 1.5
    omega: float = 8.0
    w_fwd: float = 2.0
    w_bwd: float = 0.5

    def __post_init__(self) -> None:
        # Staggered initial state: a uniform one synchronises the chain.
        indices = np.arange(1, self.num_oscillators + 1)
        self.x = 0.1 * np.cos(2 * np.pi * indices / self.num_oscillators)
        self.y = np.zeros(self.num_oscillators)

    def _derivatives(self, x: np.ndarray, y: np.ndarray):
        left, right = np.r_[0.0, x[:-1]], np.r_[x[1:], 0.0]
        dy = (
            self.mu * (1 - x * x) * y
            - self.omega**2 * x
            + self.w_fwd * (left - x)
            + self.w_bwd * (right - x)
        )
        return y, dy

    def step(self, dt: float) -> None:
        """One RK2 (midpoint) step."""
        k1x, k1y = self._derivatives(self.x, self.y)
        k2x, k2y = self._derivatives(self.x + 0.5 * dt * k1x, self.y + 0.5 * dt * k1y)
        self.x, self.y = self.x + dt * k2x, self.y + dt * k2y

    def warm_up(self, dt: float, num_cycles: float = 40, min_time: float = 5.0) -> None:
        duration = max(min_time, num_cycles * 2 * np.pi / self.omega)
        for _ in range(int(round(duration / dt))):
            self.step(dt)


def amplitude_envelope(
    num_segments: int, amplitude_scale: float, amp_taper: float
) -> np.ndarray:
    """Servo amplitude per joint, changing linearly along the body (``amp_taper > 0``: larger towards the tail)."""
    segment = np.arange(2, num_segments + 1, dtype=float)
    return amplitude_scale * (
        1 + amp_taper * (segment - (num_segments + 1) / 2) / (num_segments / 2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", default="mjc", choices=["mjc", "warp"])
    parser.add_argument(
        "--render", default="human", choices=["human", "rgb_array", "none"]
    )
    parser.add_argument("--num-segments", type=int, default=8)
    args = parser.parse_args()

    num_segments = args.num_segments
    env = make(
        "swimmer/head_first_swimming",
        backend=args.backend,
        morphology_specification=default_swimmer_morphology_specification(
            num_segments=num_segments
        ),
        render_mode=None if args.render == "none" else args.render,
    )
    # A stronger forward coupling (w_fwd > w_bwd) makes the body wave travel from head to tail, which drives the
    # swimmer head first, as in anguilliform fish; swap them and it swims tail first.
    cpg = VanDerPolChain(num_segments, mu=1.5, omega=8.0, w_fwd=2.0, w_bwd=0.5)
    dt = env.mj_model.opt.timestep
    cpg.warm_up(dt)
    amplitude = amplitude_envelope(num_segments, amplitude_scale=0.4, amp_taper=0.0)

    if env.backend.name == "mjc":
        state, step = env.reset(0), env.step
    else:
        import jax

        state, step = jax.jit(env.reset)(jax.random.PRNGKey(0)), jax.jit(env.step)
    while not bool(state.truncated):
        state = step(state, amplitude * cpg.x[1:])
        for _ in range(env.config.num_physics_steps_per_control_step):
            cpg.step(dt)
        if env.config.render_mode is not None:
            env.render(state)

    duration = float(state.info["time"])
    print(
        f"After {duration:.1f} s: head-first distance {float(state.info['head_first_distance']):+.3f} m "
        f"({float(state.info['head_first_distance']) / duration:+.3f} m/s), actuator work "
        f"{float(state.info['actuator_work']):.2f} J."
    )
    env.close()


if __name__ == "__main__":
    main()
