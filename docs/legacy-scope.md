# Relationship to the earlier platform

DFT Agent is a standalone application for VASP on Slurm.
It follows the earlier project's approach to job tracking and result checks.
It does not depend on that platform, its private datasets or old experiment state.

Scientific guidance, scoped run history and repair planning are adapted from the
earlier platform. The current executor remains in use. See [the workflow](harness.md).

The 2026-10-07 audit of the earlier workspace recorded **911 passed, 93 failed
and 2 errors** out of 1,006 tests. The
[audit record](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/legacy-scope.json) preserves those results and the
source report's SHA-256 hash:

| Unresolved category | Count | Treatment in this release |
| --- | ---: | --- |
| Missing historical evidence or dataset files | 89 | Keep the old tests and records in the earlier workspace; these datasets are not needed by this package |
| Dependency hash checks in advanced DFT campaigns | 5 | Defect/surface and k-point research campaigns are outside the basic workflows |
| Pinned XPS parser interpreter contract | 1 | XPS is outside the release scope |

The affected test files remain in the earlier workspace. Test counts for this
package cover its own tests, with separate checks of SSH transfers, job tracking,
results and installation. Its v0.1.0 cluster calculations have their own input
and output records, described in the [validation report](validation.md).
