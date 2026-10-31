"""Small math helpers that work with both numpy and jax.numpy (pass the namespace as ``xp``)."""

from __future__ import annotations

from typing import Any


def quat_to_euler(quat: Any, xp: Any) -> Any:
    """MuJoCo ``(w, x, y, z)`` quaternion to extrinsic xyz Euler angles (roll, pitch, yaw) in radians."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    roll = xp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = xp.arcsin(xp.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return xp.stack([roll, pitch, yaw])


def yaw_to_quat(yaw: Any, xp: Any) -> Any:
    """Quaternion ``(w, x, y, z)`` of a rotation of ``yaw`` radians around the world z-axis."""
    zero = xp.zeros_like(yaw)
    return xp.stack([xp.cos(yaw / 2), zero, zero, xp.sin(yaw / 2)])
