# The Brittle Star Environment

> “The whole of the brittle stars are curious and restless beings. They can never remain in the same attitude for the
> tenth part of a second, but are constantly twisting their long arms, as if they were indeed the serpents with which
> Medusa’s head was surrounded.” -- J. G. Wood, 1898

The Brittle Star Robot environment within the Bio-Inspired Robotics Benchmark (BRB) presents an intriguing case study,
embodying the intersection of natural biological systems and robotic engineering. To introduce this environment, it's
essential to first understand the biological entity that inspired it: the brittle star.

![](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/assets/brittle_stars_species.jpg?raw=true)
(Image src: https://doi.org/10.1016/j.jsb.2020.107481)

## The Brittle Star

Brittle stars, belonging to the class Ophiuroidea, are remarkable echinoderms distinguished by their distinctively
slender arms and central disk. They inhabit various marine environments, showcasing a fascinating array of behaviors and
adaptations that have captivated biologists for decades. One of the most striking features of brittle stars is their
locomotion. Unlike their close relatives, the starfish, brittle stars move with a unique, elegant undulating motion of
their arms. This locomotion is not only efficient but also highly adaptable, allowing them to navigate complex
underwater terrains with agility and precision.

In addition to locomotion, brittle stars use their arms to hold onto substrates, coiling their flexible arms tightly
around structurally complex objects such as kelp, sponges, or corals, with spines located along the arms aiding in the
anchoring process.

From an evolutionary standpoint, brittle stars represent a significant lineage, offering insights into the adaptability
and resilience of marine life. Their ability to regenerate lost limbs and their sensory capabilities, which include
responding to light without distinct eyes, are just a few examples of their remarkable biological traits. These
characteristics make them an excellent subject for bio-inspired robotics, as they provide a template for designing
robots that are both flexible and resilient, capable of navigating and adapting to diverse environments.

### Morphology

![](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/assets/brittle_stars_arm_anatomy.jpg?raw=true)
(Image src: https://doi.org/10.1016/j.jsb.2020.107481)

At the core of a brittle star's anatomy is the central disk, which varies in size from as small as 3mm to as large as
50mm. This disk is not just a structural component but also serves as a crucial hub for the organism's nervous system
and, in some species, reproductive organs.

Brittle stars possess five arms that are symmetrically arranged in a pentaradial fashion around the central disk. This
pentaradial symmetry is a hallmark of echinoderms and is crucial for the brittle star's locomotion and interaction with
its environment. These arms typically span about two to twenty times the diameter of the central disk. In terms of
mobility, the arms of brittle stars are capable of both in-plane and out-of-plane bending.

Structurally, each arm consists of a chain of segments, each one consisting out of a vertebra and 4 muscle groups and
completely enclosed by plates. This skeletal structure provides both support and flexibility, enabling the arms to move
fluidly while maintaining structural integrity.

![](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/assets/brittle_stars_muscles.jpg?raw=true)
(Image src: https://doi.org/10.3389/fnbot.2019.00104)

Lastly, the arms of brittle stars are adorned with soft spines. These spines play a crucial role in anchoring and
interacting with the environment. They provide stability and traction on various surfaces, aiding in locomotion and
foraging activities.

For an interesting read w.r.t. brittle star
morphology: [The structural origins of brittle star arm kinematics: An integrated tomographic, additive manufacturing, and parametric modeling-based approach](https://www.sciencedirect.com/science/article/pii/S1047847720300447)

### Nervous system

![](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/assets/brittle_stars_nervous_system.jpg?raw=true)
(Image src: http://dx.doi.org/10.3389/fnbot.2019.00066)

The nervous system of the brittle star is a fascinating aspect of its biology, characterized by its decentralized
nature. This decentralized nervous system is quite different from the central nervous systems found in vertebrates and
even some invertebrates. In brittle stars, the nervous system is primarily composed of a nerve ring that encircles the
central disk, from which branches extend into each arm.

The nerve ring acts as a central hub, integrating sensory inputs and coordinating the movements of the arms. Unlike
centralized nervous systems where a brain or a similar central organ exerts control, the nerve ring in brittle stars
does not function as a central command center. Instead, it serves more as a conduit or a relay station, facilitating
communication across the organism's body.

From this nerve ring, nerve branches run along each arm, allowing for localized control and sensory perception within
each limb. This structure is crucial for the brittle star's survival, as it enables each arm to react independently to
stimuli. For instance, if one arm encounters a predator or a harmful stimulus, it can respond immediately and
appropriately, without the need for centralized processing.

### Behavioural studies

Several modes of locomotion have been observed in brittle star species, with one species often exhibiting multiple
modes. A common mode is the ['breast stroke' or 'rowing'](https://www.youtube.com/watch?v=X8UEST-flCM), where the
brittle star moves forward led by an arm, uses two lateral arms as rowers, and drags the remaining two arms passively.
This mode is documented in various studies. Another mode, known as 'paddling' or 'reverse rowing,' involves dragging the
rearmost arm while the other four arms actively row. These modes, which involve bilaterally coordinated movements, allow
the brittle star to crawl in a specific direction. However, as the role of each arm changes with the direction of
movement, brittle stars do not maintain fixed front-back and left-right axes.

The following biological works provide valuable insights into the behavior and control mechanisms of brittle stars,
which provides a source of inspiration for their artificial counterpart. Here's a summary of each paper's findings:

- **Flexible coordination of flexible limbs: decentralized control scheme for inter- and intra-limb coordination in
  brittle stars’ locomotion**
    - Brittle stars display distinct roles for their arms: forelimbs, hindlimbs, and a center limb. This specialization
      allows for effective locomotion.
    - When all arms are shortened, brittle stars maintain their inter-arm coordination pattern, similar to when they are
      intact.
    - If some arms are removed, they still move effectively by coordinating the degrees of freedom in the remaining
      arms.
    - In the presence of obstacles, arms either push against objects to assist propulsion or avoid objects that hinder
      propulsion.
    - An arm completely detached from its disc is incapable of coordinated locomotion, indicating the importance of the
      proximal ends of the arms in the central disk for locomotion.
- **The function of the ophiuroid nerve ring: how a decentralized nervous system controls coordinated locomotion**
    - Nerve cut experiments reveal that the longest axons in the circumoral nerve ring do not extend beyond a fifth of
      its total length, suggesting limited direct connections between distant nodes.
    - Coordinated locomotion persists even with a single nerve ring cut, implying bidirectional connections between
      adjacent arms.
    - Arms adjacent to a nerve cut are less likely to lead in locomotion, suggesting that leading arms are those with
      better information propagation capabilities.
- **Decentralized Control Mechanism for Determination of Moving Direction in Brittle Stars With Penta-Radially Symmetric
  body**
    - When the nerve ring is cut in one place, movement tends to be in the opposite direction of the cut, and
      coordination is maintained except for arms adjacent to the cut.
    - With two cuts in the nerve ring, only neurally connected arms coordinate, while disconnected ones do not.
    - Complete severance of nerve ring connections between arms results in a lack of coordination and thus no
      locomotion.
    - If the nerve ring is cut on both sides of the same arm, the neurally isolated arm does not coordinate with others,
      highlighting the role of sensory input in coordination.
- **A brittle star-like robot capable of immediately adapting to unexpected physical damage**
    - Amputation experiments show varying roles for arms: center limb for orientation, forelimbs for propulsion, and
      minimal role for hindlimbs.
    - If arms are amputated, the robot adapts by reassigning roles, like using remaining arms for reverse rowing or as a
      trailing arm.
    - With three arms removed and no adjacent arms remaining, the direction-orienting arm swings to pull the body
      forward, while other arms play minimal roles in propulsion.
    - If only one arm remains, it swings left and right to pull the body forward.
    - When ground contacts of the arms are deprived, the subjects lose coordination in their arms.

### What makes it interesting for robotics?

The brittle star presents an intriguing case study for robotics due to several key characteristics:

1. Adaptive Locomotion in Unpredictable and Unstructured Environments: Brittle stars excel at navigating through
   unpredictable and unstructured environments. Their ability to adapt their locomotion patterns in response to varying
   terrain and obstacles makes them an excellent model for developing robots that can operate effectively in complex and
   dynamic settings.
2. Prehensile Arms Enabling Manipulation: The arms of brittle stars are not only locomotive appendages but also
   prehensile, meaning they have the ability to grasp and manipulate objects, such as prey. This feature is highly
   valuable in robotics, offering insights into designing robotic limbs that can perform intricate manipulation tasks.
3. Robustness to Damage: Thanks to their modular and redundant design, brittle stars are highly robust to damage. Each
   arm can operate independently, which means that the loss or impairment of one arm does not significantly hinder the
   overall functionality. This characteristic is crucial for creating resilient robotic systems that can maintain
   operation even when parts of them are damaged.
4. Integration of Strength and Flexibility in Arms: The arms of a brittle star embody a unique combination of strength
   and flexibility. This integration is key to their versatile functionality, allowing for both powerful movements and
   delicate handling. Replicating this in robotics could lead to the development of robotic limbs that are both strong
   and dexterous, capable of performing a wide range of tasks.

## In-silico brittle star morphology

The in-silico model of the brittle star morphology provides an abstract virtual twin. As discussed above, every
morphology in the BRB is parameterized via [FPRS](https://github.com/Co-Evolve/fprs) and the brittle star's morphology
specification can be
found [here](https://github.com/Co-Evolve/brb/blob/new-framework/brb/brittle_star/mjcf/morphology/specification/specification.py).
Default parameter values can be
found [here](https://github.com/Co-Evolve/brb/blob/new-framework/brb/brittle_star/mjcf/morphology/specification/default.py).
In our case, the most important configuration parameters are the number of arms, and the number of segments that each
arm has.

![](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/assets/brittle_stars_in_silico.png?raw=true)
Two example in-silico brittle star morphologies.

Every segment has two degrees of freedom (DoF), one for in-plane motion and one for out-of-plane motion.

In terms of actuation, the morphology specification can be used to select either position based control, and torque
based control.
The morphology specification can also be used to use tendon-based transmission (but only with torque based control). In
this case, four tendons will be added, similar to the muscle architecture of the brittle star as shown above.

In terms of sensing, the following sensors are implemented. These sensors define the base set of observations that every
brittle star environment returns as observations (further discussed below).

- Proprioception
    - Joint positions (two per segment, in-plane and out-of-plane, in radians)
    - Joint velocities (two per segment, in-plane and out-of-plane, in radians / second)
    - Joint actuator force (i.e. the total actuator force acting on a joint, in Newton meters) (two per segment)
    - Actuator force (the scalar actuator force, in Newtons) (four per segment in case of tendon transmission, otherwise
      two)
    - Tendon position (in case tendon transmission is used, four per segment, in meters)
    - Tendon velocity (in case tendon transmission is used, four per segment, in meters / second)
    - Central disk's position (w.r.t. world frame)
    - Central disk's rotation (w.r.t. world frame, in radians)
    - Central disk's velocity (w.r.t. world frame, in m/s)
    - Central disk's angular velocity (w.r.t. world frame, in radians/s)
- Exteroception
    - Contact (X continuous values per segment, in Newtons) (X is defined in the morphology specification)

In terms of actuation, the following actuators are implemented (two per segment, one for the in-plane DoF and one for
the out-of-plane DoF). The brittle star's morphology specification defines which if either position-based or
torque-based control is used.
The actuator force limits are scaled by the segment radii. Consequently, the maximum force applied by an actuator
decreases along the arm.

## Environment variants

The brittle star environment comes with three locomotion-orientated tasks that provide a curriculum of increasing
difficulty:

1. Undirected locomotion (simple): Move the brittle star away as far as possible from its starting position.
    - Reward per timestep: $distance\\_current\\_timestep - distance\\_previous\\_timestep$. The reward will thus be
      positive if the distance from its starting position has increased in the current timestep, and negative if this
      distance has decreased.

2. Directed locomotion (intermediate): Move the brittle star towards target.
    - Reward per timestep: $distance\\_previous\\_timestep - distance\\_current\\_timestep$. The reward will thus be
      positive if the distance to the target has decreased in the current timestep, and negative if this distance has
      increased.
    - Requires an aquarium with `attach_target=True`.
    - Target position can be specified using the `target_position` argument of
      the `reset` function.
      If no target position is given, a target will be spawned randomly on a circle with a radius given by the
      `target_distance` argument of
      the [BrittleStarDirectedLocomotionEnvironmentConfiguration](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/environment/directed_locomotion/shared.py).
    - Additional observations:
        - Unit direction on the horizontal XY plane from the central disk to the target.
        - The distance in XY plane from the central disk to the target.
3. Light escape (hard): Move the brittle star towards darker spots in the environment.
    - Reward per timestep: $light\\_income\\_previous\\_timestep - light\\_income\\_current\\_timestep$. The reward will
      thus
      be positive if the measured light income has decreased in the current timestep, and negative if the light income
      has increased. The light income at a given timestep is calculated as a weighted average over all body geoms (
      weight scales with the surface area of the geom).
    - The light escape environment configuration accepts an additional argument `random_initial_rotation`. This sets a
      random z-axis rotation of the brittle star upon environment resets.
    - Requires an aquarium with `sand_ground_color=True`.
    - Additional observations:
        - The amount of light each segment takes in.

All brittle star environments support visualising segment contacts (i.e., coloring a segment's capsule red upon contact)
via the `color_contacts` argument of the environment configuration.

![](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star/assets/brittle_star_environments.png?raw=true)
From left to right: the undirected locomotion, the targeted locomotion (target is the red sphere), and the light
escape (with light noise) environments.
