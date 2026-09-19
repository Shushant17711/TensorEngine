## Title
`default.tensor` (method="mps") ignores the device's `wires=` ordering — physical MPS
chain order is fixed by numeric label order regardless

## Environment
- PennyLane 0.45.1
- quimb 1.15.0
- Python 3.12.14, Linux x86_64

## Summary

When constructing a `default.tensor` device with `method="mps"`, passing a custom order
in the `wires=` argument (e.g. `wires=[0, 7, 1, 6, 2, 5, 3, 4]`) has **no effect** on the
bond-dimension truncation behavior — results are bit-for-bit identical to using the
natural order (`wires=8`). The only way to actually change which qubits are physically
adjacent in the underlying MPS chain is to relabel the wires used *inside the circuit
itself* (e.g. via `qml.map_wires`), not via the device constructor.

This is surprising given the device's own docstring describes `wires` as accepting
"unique labels for the wires ... in the desired order," which reads as if it controls
qubit placement in the simulated device the way it does for e.g. `default.qubit`'s
wire-to-index mapping in other tensor-network-adjacent contexts. If this is expected
behavior for the current implementation, it would be worth calling out explicitly in the
docs — right now a user has no signal that reordering via the device constructor silently
does nothing (no error or warning), which can produce a systematically inflated
`max_bond_dim` requirement for a circuit whose actual entanglement structure would need
much smaller `max_bond_dim` under a different physical wire order.

## Minimal reproduction

```python
import pennylane as qml
import numpy as np


def circuit(w):
    for i in range(8):
        qml.RY(w[i], wires=i)
    # four independent, near-maximally-entangled pairs -- long range under natural order
    qml.CNOT(wires=[0, 7])
    qml.CNOT(wires=[1, 6])
    qml.CNOT(wires=[2, 5])
    qml.CNOT(wires=[3, 4])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(7))


rng = np.random.default_rng(0)
w = rng.uniform(0, 2 * np.pi, size=8)

dev_exact = qml.device("default.qubit", wires=8)
exact = float(qml.QNode(circuit, dev_exact)(w))

# (a) natural wire order
dev_natural = qml.device("default.tensor", wires=8, method="mps", max_bond_dim=2)
val_natural = float(qml.QNode(circuit, dev_natural)(w))

# (b) custom device wire order interleaving each entangled pair
dev_reordered = qml.device(
    "default.tensor", wires=[0, 7, 1, 6, 2, 5, 3, 4], method="mps", max_bond_dim=2
)
val_reordered = float(qml.QNode(circuit, dev_reordered)(w))

print("exact:     ", exact)
print("natural:   ", val_natural, "diff", abs(val_natural - exact))
print("reordered: ", val_reordered, "diff", abs(val_reordered - exact))  # IDENTICAL to natural

# (c) the only thing that actually works: relabel wires INSIDE the circuit
wire_map = {0: 0, 7: 1, 1: 2, 6: 3, 2: 4, 5: 5, 3: 6, 4: 7}
mapped_circuit = qml.map_wires(circuit, wire_map)
dev_default_order = qml.device("default.tensor", wires=8, method="mps", max_bond_dim=2)
val_mapped = float(qml.QNode(mapped_circuit, dev_default_order)(w))
print("mapped:    ", val_mapped, "diff", abs(val_mapped - exact))  # matches exact
```

**Output:**
```
exact:      -0.128470859145962
natural:    -0.05432... diff 0.0741...
reordered:  -0.05432... diff 0.0741...   <-- identical to natural, wires= had no effect
mapped:     -0.128470859145962 diff ~4e-16   <-- only qml.map_wires actually changes behavior
```

## Expected behavior

Either:
1. The `wires=` order on `default.tensor` should control the physical MPS chain order
   (matching what the docstring's phrasing suggests), or
2. If this is not supported/intended, the docs should say so explicitly, since right now
   there's no error, warning, or documented caveat — a user reordering `wires=` to try to
   reduce required bond dimension will silently get no benefit at all.

Found while building an automated tool that infers a good qubit ordering for a
`default.tensor` MPS surrogate from a circuit's entanglement structure — this behavior
meant an early version of that tool's fix (reordering via the device constructor) had no
effect at all, until switching to `qml.map_wires` on the circuit itself.
