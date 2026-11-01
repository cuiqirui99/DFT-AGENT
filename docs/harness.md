# Scientific guidance

Plans use local VASP notes, structure hints and relevant saved runs. Open
**Scientific guidance** to see method choices, checks and sources. The report
is saved with the plan. Past runs supply advice, never automatic parameter changes.
Each check states when it applies, what to inspect and its source. Missing
moments, spin axes or U/J values return the plan to a question. Geometry hints
remain suggestions. Result checks stay pending until calculation evidence exists.

Bundled memory adds checked guidance, success and failure cases, and repair
examples. Plans and repairs retrieve matching entries and save their references.
Open **Memory** to search records, inspect evidence or import selected local
records. Rejected, retired and unverified lessons are excluded from advice.
[Memory](memory.md).

History comes from the selected **Run folder**. Results are rechecked before use;
changed or rejected outputs cannot supply successful examples. Magnetic moments
and spin axes require compatible methods and site geometry.

For a failed run, open **Repair**, then **Plan repair**. Review the proposed change,
click **Prepare repair**, and submit the new run after checking its inputs.
The original run stays intact. A repair chain allows at most two attempts.

Repairs can increase electronic or ionic iteration limits, or the Slurm time limit.
They keep the method, structure, k-points and convergence thresholds unchanged.
Unknown failures, uncertain job status, magnetic comparisons and failed PBE spectra
that need the original charge density require manual review. Reconnect first if
the original job has not been collected.

CLI:

```bash
dft-agent plan examples/Si.cif "Relax, then SCF" --provider codex --runs ./runs
dft-agent repair ./runs/failed --provider codex --output repair.json
dft-agent prepare-repair ./runs/failed ./runs/repaired --plan repair.json
dft-agent watch ./runs/repaired
```

This adapts the earlier project's document guidance, scientific reports, scoped
experience and repair planning. It keeps the current executor and its result checks.
It does not import the old private corpus, run unrestricted model tools, or search
the literature automatically. See [sources and method limits](harness-sources.md).

The v0.3.0 checks include 488 passing offline tests and real Codex calls for
revisions, method choices and missing inputs. On Slurm, reviewed electronic,
ionic and time-limit repairs succeeded with unchanged structures, k-points,
methods and thresholds.
See the [memory checks](validation-0.3.1.md), [calculation validation](validation-0.3.0.md) and
[earlier harness record](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/harness.json).
