# 0.4.9 checks

The release was reviewed on an Apple Silicon Mac. Source-level tests covered
multi-turn task changes, structure builders, input checks, connections,
continuation and reports. A real Codex conversation prepared and revised a
Quantum ESPRESSO task, which completed on Parallel Supercomputing. Separate
conversations exercised structure edits and reports from retained VASP results.

The packaged app was checked separately: first launch, project creation and
persistence, model-setup screens, the MCP subprocess, engine inputs, plots,
and the native window. Its frozen structure tools built a 192-atom MoS2 cell
and a constrained Cu/CO surface. A modified external ASE library was loaded
successfully from a copied application.

Windows end-to-end testing remains pending and this release contains no new
Windows installer. No test establishes universal compatibility with every
cluster, engine configuration or material; scientific convergence checks
remain specific to the calculation.
