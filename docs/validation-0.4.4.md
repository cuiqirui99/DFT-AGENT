# 0.4.4 checks

The release checks cover project storage, model and SSH connections, workflow
planning, input review, execution state, result collection and desktop packaging.
They do not establish that every material, model provider or cluster is supported.

## What was checked

- Automated tests exercise both engines, project isolation, migrations, review
  and submission boundaries, retries, file transfers and result readers.
- A real Codex CLI connection used project tools to import silicon, plan a QE
  relaxation and revise the same task. Planning did not submit a calculation.
  A recovered CLI warning no longer marks a completed answer as failed.
- Password SSH login, a read-only cluster probe and MPI toolchain detection
  were checked on Paracloud.
- A local SSH server requires a password or key followed by a verification code.
  Tests exercise authenticated session reuse by separate worker processes,
  command execution, file transfer, disconnects and revocation. The scheduler
  responses in this test are simulated; no scientific calculation is claimed.
- Desktop checks start the installed application with fresh settings, create and
  reopen a project, call the bundled project tools, render the native window and
  start and stop the local SSH session service using the system keychain.

The [Windows packaging workflow](../.github/workflows/desktop.yml) runs the tests
on Windows, builds the installer and checks a fresh installation. The
[Linux workflow](../.github/workflows/ci.yml) tests Python 3.11–3.13 and installs
the wheel outside the source checkout. Release packages are published only
after these checks pass.

## Scientific evidence

The real VASP → QE, Wannier90 and Python workflow checks from 14 October are
recorded in [Project workspace](project-workspace.md#acceptance-on-14-october-2026).
The broader QE checks are in [0.4.3 validation](../validation/qe-0.4.3/README.md).
The 0.4.4 connection and interface checks do not replace numerical convergence
studies or independently validate new materials.

## Limits

Claude Code and Claude API adapters have automated tests, but no authenticated
Claude call was available for this review. Multi-factor authentication was
checked against the local SSH server, not a production Swedish cluster. PBS
has adapter tests, not live submission acceptance. The macOS package targets
Apple Silicon; the Windows package targets x64.

Older release tags and assets are retained unchanged.
