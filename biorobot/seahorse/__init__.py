"""The seahorse tail: a muscle-architecture playground without a task (Marzougui et al. 2025).

Create the environment with ``biorobot.make("seahorse/tail", backend="mjc")`` (or ``"mjx"``/``"warp"``) and configure
the muscles through ``morphology_specification=...``.
"""

from biorobot.seahorse.arena import SeahorseArena, SeahorseArenaConfiguration
from biorobot.seahorse.environment import (
    SeahorseTailEnv,
    SeahorseTailEnvConfig,
    default_tail_morphology_specification,
)
from biorobot.seahorse.morphology import SeahorseMorphology
from biorobot.seahorse.specification import (
    HMMTendonRoutingSpecification,
    JointSpecification,
    MeshSpecification,
    SeahorseHMMTendonActuationSpecification,
    SeahorseMorphologySpecification,
    SeahorseMVMTendonActuationSpecification,
    SeahorsePlateSpecification,
    SeahorseSegmentSpecification,
    SeahorseTendonActuationSpecification,
    SeahorseVertebraeSpecification,
    SeahorseVertebralStrutSpecification,
    default_hm_routing_specification,
    default_seahorse_morphology_specification,
)

__all__ = [
    "SeahorseArena",
    "SeahorseArenaConfiguration",
    "SeahorseTailEnv",
    "SeahorseTailEnvConfig",
    "default_tail_morphology_specification",
    "SeahorseMorphology",
    "HMMTendonRoutingSpecification",
    "JointSpecification",
    "MeshSpecification",
    "SeahorseHMMTendonActuationSpecification",
    "SeahorseMorphologySpecification",
    "SeahorseMVMTendonActuationSpecification",
    "SeahorsePlateSpecification",
    "SeahorseSegmentSpecification",
    "SeahorseTendonActuationSpecification",
    "SeahorseVertebraeSpecification",
    "SeahorseVertebralStrutSpecification",
    "default_hm_routing_specification",
    "default_seahorse_morphology_specification",
]
