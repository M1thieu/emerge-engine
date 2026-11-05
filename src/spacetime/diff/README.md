# diff

**Level:** discretization, for training, not for the world. **Status:** separate from
both paths; built on the explicit solver's transfers.

A differentiable forward simulation with a hand-derived reverse pass, for offline
training of gaits. Nothing the world runs depends on it.

**Derives from:** `spacetime/transfer/` (the adjoint chain). The controllers it trains
are defined here (`config.rs`).
