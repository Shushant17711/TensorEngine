# Preregistration — target model battery

**Status: not yet written.** Per the spec (§4.3) and the project timeline (README §8,
Week 3), the target-model list for the main sweep (E2) must be committed here, with a
timestamp, **before running any real sweep** — writing it now, before doing the work to
identify a fair and representative battery, would itself be the cherry-picking risk the
spec warns against.

When this is filled in (Week 3), it will contain:

- 6-10 published small-scale QML/QCNN classifiers selected for reimplementation, with
  paper citations.
- For each: why it was picked, and which "expected outcome" bucket it's predicted to fall
  into (expected-to-dequantize / expected-to-resist / uncertain) — recorded *before*
  running the sweep, so the prediction itself is falsifiable.
- The commit hash/date this list was frozen at.
