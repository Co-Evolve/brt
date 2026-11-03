"""
model.py — MuJoCo XML for the abstract N-segment capsule-chain swimmer.

Copied from MuJoCo-demo ``src/swimmer.py`` (commit f7ad026). The XML template,
segment-length rule and every physical constant are unchanged. Removed: the
C. elegans preset (``WORM_MORPHOLOGY``), the on-disk XML cache and the JSON
simulation records.

The body lies in the xy-plane with hinges around z; gravity is off; fluid
forces come from MuJoCo's ellipsoid fluid model (``fluidshape="ellipsoid"``,
viscosity 0.1 Pa s, density 1000 kg/m^3). The head segment ``seg0`` carries the
free joint and sits at the origin; the chain extends toward +x.
"""

DEFAULT_RADIUS = 0.015


def segment_lengths(N: int, taper: float, L0: float = 0.1) -> list[float]:
    """Per-segment lengths (m), linearly interpolated head -> tail."""
    return [L0 * (1 + (taper - 1) * i / (N - 1)) for i in range(N)]


def body_length(N: int, taper: float, L0: float = 0.1) -> float:
    """Total body length (m) for a given morphology."""
    return float(sum(segment_lengths(N, taper, L0)))


def make_swimmer_xml(
    N: int, taper: float, stiffness: float, radius: float = DEFAULT_RADIUS
) -> str:
    """Return a MuJoCo XML string for an N-segment tapered swimmer.

    Args:
        N: number of capsule segments (>= 2).
        taper: tail-segment length / head-segment length. 1.0 = uniform.
        stiffness: passive torsional spring at each hinge (Nm/rad).
        radius: capsule radius (m).
    """
    L0 = 0.1  # head-segment length (m)
    lengths = segment_lengths(N, taper, L0)

    # Build nested <body> XML. Each child body sits at the tail end of its
    # parent; the hinge at the child's origin is what bends the chain.
    open_lines = []
    for i, L in enumerate(lengths):
        pad = "  " * (2 + i)
        if i == 0:
            open_lines.append(
                f'{pad}<body name="seg0" pos="0 0 0">\n'
                f"{pad}  <freejoint/>\n"
                f'{pad}  <camera name="track" pos="0 0 2.5" xyaxes="1 0 0 0 1 0" '
                f'mode="trackcom" fovy="60"/>\n'
                f'{pad}  <geom name="g0" fromto="0 0 0 {L:.4f} 0 0"/>\n'
            )
        else:
            prev_L = lengths[i - 1]
            open_lines.append(
                f'{pad}<body name="seg{i}" pos="{prev_L:.4f} 0 0">\n'
                f'{pad}  <joint name="j{i}" stiffness="{stiffness:.6f}"/>\n'
                f'{pad}  <geom name="g{i}" fromto="0 0 0 {L:.4f} 0 0"/>\n'
            )
    close_lines = ["  " * (2 + i) + "</body>\n" for i in reversed(range(N))]
    body_section = "".join(open_lines + close_lines)

    actuators = "\n".join(
        f'    <position name="a{i}" joint="j{i}" kp="2.0"/>' for i in range(1, N)
    )

    return f"""<mujoco model="swimmer">
  <option timestep="0.002" gravity="0 0 0" viscosity="0.1" density="1000" integrator="Euler"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"
             width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
    <material name="grid" texture="grid" texrepeat="4 4" texuniform="true" reflectance="0.2"/>
  </asset>

  <default>
    <geom type="capsule" size="{radius}" rgba="0.55 0.65 0.9 1" density="600" fluidshape="ellipsoid"/>
    <joint type="hinge" axis="0 0 1" damping="0.005" limited="true" range="-1.5 1.5"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="grid"
          pos="0 0 -0.3" contype="0" conaffinity="0"/>
{body_section}  </worldbody>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""
