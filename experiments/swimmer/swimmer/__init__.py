"""Swimmer worked example: capsule-chain swimmer, coupled Van der Pol CPG,
parameter sweep and two MAP-Elites searches (see README.md)."""

from .episode import DESCRIPTOR_NAMES, PARAM_NAMES, evaluate, simulate
from .model import body_length, make_swimmer_xml, segment_lengths

__all__ = [
    "DESCRIPTOR_NAMES",
    "PARAM_NAMES",
    "body_length",
    "evaluate",
    "make_swimmer_xml",
    "segment_lengths",
    "simulate",
]
