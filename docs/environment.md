# Environment snapshot

This is the local verification environment at the 2026-08-28 pause. It is not
a lockfile and may not exactly reproduce the package state of every historical
run. Per-run configurations and timing records remain authoritative where they
record more specific values.

| Component | Version |
| --- | --- |
| Operating system | Windows 11, 64-bit |
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime reported by PyTorch | 12.4 |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU |
| tokenizers | 0.22.2 |
| Pydantic | 2.13.4 |
| PyYAML | 6.0.3 |
| NumPy | 2.5.2 |
| pytest | 9.1.1 |

The package metadata declares minimum direct dependencies in
[`pyproject.toml`](../pyproject.toml). A future kill test should freeze a new
environment before registration and record the model revisions, CUDA stack,
drivers, dependency lock, and hardware power settings used for every arm.
