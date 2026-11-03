"""
vdp_network.py — chain of coupled Van der Pol oscillators.

    ẍᵢ = μ(1 − xᵢ²)ẋᵢ − ω²xᵢ + w_fwd(xᵢ₋₁ − xᵢ) + w_bwd(xᵢ₊₁ − xᵢ)

with Dirichlet ends: x₀ ≡ 0 and x_{N+1} ≡ 0. Integrated with RK2 (midpoint).
"""

import numpy as np


def derivatives(
    x: np.ndarray, y: np.ndarray, mu: float, omega: float, w_fwd: float, w_bwd: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dx/dt, dy/dt) for the whole chain."""
    x_left = np.empty_like(x)
    x_left[0] = 0.0  # x_0 boundary
    x_left[1:] = x[:-1]
    x_right = np.empty_like(x)
    x_right[-1] = 0.0  # x_{N+1} boundary
    x_right[:-1] = x[1:]

    dx = y
    dy = (
        mu * (1.0 - x * x) * y
        - omega * omega * x
        + w_fwd * (x_left - x)
        + w_bwd * (x_right - x)
    )
    return dx, dy


def rk2_step(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    mu: float,
    omega: float,
    w_fwd: float,
    w_bwd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One RK2 (midpoint) step."""
    k1x, k1y = derivatives(x, y, mu, omega, w_fwd, w_bwd)
    xm = x + 0.5 * dt * k1x
    ym = y + 0.5 * dt * k1y
    k2x, k2y = derivatives(xm, ym, mu, omega, w_fwd, w_bwd)
    return x + dt * k2x, y + dt * k2y


def default_x0(N: int) -> np.ndarray:
    """Staggered initial xᵢ — breaks symmetry so the chain doesn't synchronise."""
    return 0.1 * np.cos(2.0 * np.pi * np.arange(1, N + 1) / N)


def warmup_vdp(
    N: int,
    mu: float,
    omega: float,
    w_fwd: float,
    w_bwd: float,
    dt: float,
    warmup_s: float = 5.0,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the chain in isolation until it sits on its limit cycle.

    Returns (x, y) state at the end of the warm-up window.
    """
    x = default_x0(N) if x0 is None else np.asarray(x0, dtype=float).copy()
    y = np.zeros(N)
    for _ in range(int(round(warmup_s / dt))):
        x, y = rk2_step(x, y, dt, mu, omega, w_fwd, w_bwd)
    return x, y


def simulate_vdp(
    N: int,
    mu: float,
    omega: float,
    w_fwd: float,
    w_bwd: float,
    dt: float,
    T: float,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the chain.

    Returns:
        x_trace: shape (n_steps, N), x state per step.
        t_trace: shape (n_steps,), time per step.

    Default x0: staggered xᵢ(0) = 0.1·cos(2πi/N), ẋᵢ(0) = 0 (1-indexed i).
    Required to break symmetry — uniform initial state synchronises.
    """
    x = default_x0(N) if x0 is None else np.asarray(x0, dtype=float).copy()
    y = np.zeros(N)

    n_steps = int(round(T / dt))
    x_trace = np.empty((n_steps, N))
    t_trace = np.arange(n_steps) * dt

    for step in range(n_steps):
        x_trace[step] = x
        x, y = rk2_step(x, y, dt, mu, omega, w_fwd, w_bwd)
    return x_trace, t_trace
