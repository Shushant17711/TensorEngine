"""DequantEngine: automated tensor-network baseline generator for PennyLane QNNs.

See README.md for the project spec and NOTES_WEEK1.md for the scope decision this package
implements.
"""

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.parser import CircuitTopology, parse_qnode
from dequant_engine.sweep import SweepResult, sweep_bond_dimension

__all__ = [
    "CircuitTopology",
    "MPSSurrogate",
    "SweepResult",
    "parse_qnode",
    "sweep_bond_dimension",
]
