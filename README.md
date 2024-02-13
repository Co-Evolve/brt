<h1>
  <a href="#"><img alt="moojoco-banner" src=https://github.com/Co-Evolve/brt/blob/main/assets/banner.png?raw=true" width="100%"/></a>
</h1>

The **Bio-Inspired Robotics Testbed (BRT)** represents a significant stride in the convergence of robotics and
experimental biology, offering a unique platform that leverages the MuJoCo physics simulator to create a collection of
bio-inspired robotics simulation environments. This initiative is more than just a technical tool; it serves as a
crucible for cross-disciplinary dialogue and innovation, uniting two fields that, while distinct, share a common goal of
understanding and emulating complex biological systems. In the realm of robotics, this testbed addresses a pivotal
challenge: the design and control of bio-inspired robots. These robots, inspired by the intricacies of biological
organisms, present novel hurdles that contemporary optimization methodologies struggle to surmount. By providing a
benchmark specifically tailored to these challenges, the initiative not only facilitates the validation of existing
methodologies but also catalyzes the development of new approaches. This is crucial for advancing the field, as it
pushes the boundaries of what we can achieve in robotic design and control, drawing inspiration from the most
sophisticated systems found in nature. On the biological front, the application of robotics serves as a powerful
comparative method. It offers a more controlled environment for conducting comparative studies, essential for dissecting
complex behaviors, ecological interactions, and the evolutionary histories of organisms. This perspective is invaluable,
as it transcends traditional observational methods, allowing for more controlled analyses. Overall, the Bio-Inspired
Robotics Testbed is a testament to the symbiotic potential of robotics and biology. It not only advances our technical
capabilities in creating machines that mimic life but also deepens our understanding of the biological phenomena that
inspire these creations. This testbed is a stepping stone towards a future where the interplay between robotics and
biology yields innovations that are as profound as they are transformative.

BRT distinguishes itself with its capability to effortlessly simulate various
morphological variations of bio-inspired robots. This flexibility is largely enabled by the use of the [Framework for
Parameterized Robot Specifications (FPRS)](https://github.com/Co-Evolve/fprs), a framework that centralizes the
adaptable definition of a robot's
morphology. In FPRS, a 'Specification' acts as a comprehensive bundle, encapsulating all the modifiable parameters that
define the robot's form and structure. This approach to morphology parameterization not only simplifies the modification
process but also enhances the reproducibility and comparability of different morphological setups within the BRT

The implementation of environments within the BRT adheres to framework defined in
the [moojoco](https://github.com/Co-Evolve/moojoco) package.
This framework defines an environment as a combination of a parameterized morphology, as defined by FPRS, and a
configurable
arena. Both the morphology and the arena in this context are essentially generators for MJCF (MuJoCo XML) files,
allowing for dynamic and customizable environment creation. This design philosophy ensures that each environment within
the BRT is both flexible and specific to the needs of the experiment at hand.

In terms of simulation capabilities, the BRT offers environments in both native MuJoCo and MuJoCo XLA (MJX) formats.
This dual availability caters to diverse computational needs and preferences, ensuring that researchers can select the
simulation environment that best suits their specific requirements, whether it's the high-fidelity physics of native
MuJoCo or the accelerated computation offered by MuJoCo XLA.

In summary, the BRT's unique approach to morphology parameterization through FPRS, adherence to the moojoco
framework for environment implementation, and availability in both MuJoCo and MJX formats, collectively contribute to
its status as a versatile and powerful tool for bio-inspired robotics research. This tool not only facilitates a deeper
exploration of robotic morphologies but also bridges the gap between theoretical research and practical application in
the field of bio-inspired robotics.

## Current environments

* [Toy Example](https://github.com/Co-Evolve/brt/blob/main/biorobot/toy_example)
* [Brittle star](https://github.com/Co-Evolve/brt/blob/main/biorobot/brittle_star)

## Contributions

Interested in having a model developed? Please fill in this [form](https://forms.gle/h1xgnZA39h2qhX8k9).

## Examples

Preliminary Jupyter Notebook examples can be found [here](https://github.com/Co-Evolve/SEL3-2024).

## Installation

```pip install biorobot```
