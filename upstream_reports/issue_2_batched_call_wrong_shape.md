## Title
`default.tensor` silently returns a wrong-shaped result for a batched QNode call (no
error, no warning)

## Environment
- PennyLane 0.45.1
- quimb 1.15.0
- Python 3.12.14, Linux x86_64

## Summary

Calling a `default.tensor`-backed `QNode` with a batch of inputs (a 2D array where the
first axis is the batch dimension) does not correctly vectorize over the batch. It also
does not raise an error. It silently returns a result whose shape doesn't even match the
batch size, and whose values don't match any of the correct per-sample results computed
by calling the QNode once per sample in a Python loop.

This is a correctness hazard rather than just a missing-feature/performance gap: a user
who assumes PennyLane's usual broadcasting support (which works correctly on
`default.qubit`) applies uniformly across devices could silently get wrong results with no
indication anything went wrong.

## Minimal reproduction

```python
import pennylane as qml
import torch


def circuit(weights, x):
    qml.RX(torch.pi * x[0], wires=0)
    qml.RX(torch.pi * x[1], wires=1)
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))


dev = qml.device("default.tensor", wires=2, method="mps", max_bond_dim=2)
qnode = qml.QNode(circuit, dev, interface="torch", diff_method="best")

weights = torch.tensor([0.3, 0.7])
xs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float64)

per_sample = torch.stack([qnode(weights, xi) for xi in xs])
print("per-sample loop (correct):", per_sample, per_sample.shape)

batched = qnode(weights, xs)
print("single batched call:      ", batched, batched.shape)
```

**Output:**
```
per-sample loop (correct): tensor([ 0.7307, -0.7307, -0.7307,  0.7307, -0.0000], ...) torch.Size([5])
single batched call:       tensor([-0.7307,  0.7307], ...) torch.Size([2])
```

The batched call returns a length-2 tensor for a 5-sample input, and its values don't
correspond to any of the 5 correct per-sample results (they're closest to the first two
per-sample values, but sign-flipped) — this looks like some kind of dimension confusion
between the batch axis and one of the circuit's own array-valued inputs, rather than an
intentional (if undocumented) broadcasting convention.

## Expected behavior

Either:
1. `default.tensor` should support standard QNode-level batching correctly, matching
   `default.qubit`'s behavior for the same circuit, or
2. If batching isn't supported for this device/method, calling a QNode with a batch
   dimension should raise a clear error (as several other unsupported operations on this
   device already do, e.g. attempting `diff_method="backprop"`) rather than silently
   returning a wrong-shaped, wrong-valued result.

Found while building an automated training pipeline against `default.tensor` — caught
only because a correctness check happened to compare batched vs. per-sample results
directly; would otherwise have silently corrupted training.
