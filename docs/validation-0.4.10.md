# 0.4.10 checks

This build was prepared on 2026-11-05 and not published on its own; its fixes are
part of 0.5.0. It repackaged 0.4.9 with the fixes below. The 0.4.9 source snapshot
was rebuilt on an Apple Silicon Mac with the same dependency versions and the same
packaging steps.

What changed:

- Claude Code CLI sign-in on macOS: the CLI reads its sign-in from the keychain by
  user name. The app started it without the user name, so it fell back to a stale
  credentials file and failed although the sign-in check had passed. Both CLIs now
  receive the user name.
- One environment list for both CLIs, used by their requests and by their sign-in
  checks: proxy variables in both spellings, CA bundles, and the Windows and Linux
  settings the CLIs need to find their sign-in. API keys, passwords and the variables
  of a parent Claude Code session stay out.
- CLI failures are named by cause: expired or invalid sign-ins, usage, credit or quota
  limits, region blocks, requests that are too long, network, proxy or certificate
  problems, and missing Git Bash on Windows, in the interface language. Unrecognised
  failures are written to the app log with secrets removed.
- Reply language: replies follow the language the user writes in, and the interface
  language when a message shows none (a single word such as "build" was once answered
  in German). Explanations of verified results stay in English, because their safety
  checks read English text.
- The conversation no longer shows the raw tool-call list or the raw structure record.
  Structure warnings and sources appear as short notes, and saved files named in a
  reply can be downloaded.
- Structure results sent to the agent leave out the site list when they would exceed
  30,000 characters, and any tool result above 50,000 characters is replaced by a short
  notice. Claude Code had dropped a 304-atom twisted bilayer's result (78,887
  characters) and stored it outside the app.
- 2D entries from structure databases that are stored as 3D cells, such as JARVIS
  monolayers, are imported as sheets: periodic in-plane, the layer moved whole away
  from the cell boundary, all distances unchanged. Nanotube building accepts them.
- The app keeps its font cache in the settings folder; rebuilding it took about
  10 s at every start and in each agent turn's tool server.
- Interface notes from structure building, input review, cluster setup and search
  have Chinese text; a test fails on any shown sentence without an entry.
- Interface: your own messages sit on the right, files appear as compact chips, the
  logo shows again, and the first-start page, new-project page and settings use a
  readable width.
- Saved structures are named by what they are (formula, atom count and kind, such as
  "CrI3 · 112 atoms · twisted bilayer") in replies, downloads, structure cards and
  build forms, and runs in the side panel by their step and start time. Structure
  folder IDs, run folder names and project paths no longer appear there; links,
  inline code, bare paths and IDs that a model writes are renamed before display.
  Runs under **Attach results** still showed their folder names; 0.5.0 names
  them by step and start time there too.
- Clusters whose Slurm installation provides PBS-style qsub, qstat and qdel next to
  sbatch (Paracloud) were treated as having two schedulers, so software detection asked
  the user to choose one. Such commands now count as Slurm, also for clusters tested
  before; PBS commands found elsewhere still leave the choice open. 0.5.0 replaced
  this folder rule with the model's judgement and a test job; without a connected
  model, choose Slurm in the cluster settings.
- The open conversation is highlighted in the sidebar, and Connections and Settings
  lead back to it; after visiting Connections the conversation could look lost.
- A plain scroll over a 3D structure moves the page; ⌘ or Ctrl + scroll, or a trackpad
  pinch, zooms it. The viewer used to take every scroll that passed over it and zoomed
  a structure until its atoms filled the frame.
- The side panel shows structures and calculations only: the raw file browser, raw
  run records and the panel-size control are gone, and programs' output is folded
  away; the panel stays in view and scrolls on its own, so opening it no longer jumps the
  page to its end. The input review shows the input files, a table of each QE stage's cutoffs and
  mesh, and one resources line instead of raw records.
- The conversation header shows its controls in full and wraps on narrow windows.
- Quitting the app from its menu or with ⌘Q stops its background server, which used to
  keep running.
- Notes set as whole lists are now covered by the Chinese-text check; four had no entry.

Source-level tests: the full suite passed on the final snapshot (5621 passed, 18 skipped, none failed; 5639 collected).
New regression tests cover the environment list, the sign-in check environment,
the failure classes with texts taken from the installed Claude Code 2.1.251 and
Codex CLIs, redaction in the log, the reply-language rule in all three prompts and
the tool server, the chat and structure-card changes, result sizes, the sheet
import and the font cache, the names shown for structures, runs and files, the
scheduler rule, the way back to a conversation, the viewer's scroll and pinch handling,
the side panel and input review, the server stopping after Quit, and the Chinese-text
check for notes set as whole lists.

An independent review of the CLI and language changes found two error patterns
that misnamed failures, a test that would fail on Windows and limit messages that
still read as generic failures; all were fixed with tests. The sign-in status
commands of both CLIs reject the requests' isolation options, so the status line
stays advisory; the connection test sends a real request.

The packaged app was checked as for 0.4.9: ad-hoc signature valid; no first-party
plaintext source files, first-party modules in the bytecode archive only, ASE
outside the archive and replaceable (a copied app imported a modified ASE and
passed the full bundle check); DMG verified, mounted and installed files identical
to the build; the installed app passed the native checks and was unchanged
afterwards.

Live checks with the installed app, started with a minimal environment like a
Finder launch:

- A Claude Code CLI 2.1.251 model connection connected. With the Chinese interface,
  "build" was answered in Chinese; with the English interface, in English.
- A JARVIS CrI3 monolayer was imported as a sheet and rolled into a (6,6) armchair
  nanotube in one turn, without the workaround an earlier build needed.
- With an unreachable proxy set in lowercase, the connection test failed with the
  timeout message, shown in Chinese.
- A second start reused the font cache and showed the project within 3 s. The first
  start of a newly built app takes about a minute while macOS checks it.
- On a copy of a user's project: replies, downloads and structure cards named
  structures by formula, atom count and kind, with no IDs or paths; a plain scroll over
  a 3D structure moved the page (the previous build zoomed the structure instead); the
  open conversation stayed highlighted after visiting Connections; Quit stopped the
  app's server within 3 s; the side panel opened beside the conversation without moving
  the page; the agent saw the user's Paracloud connection as a Slurm cluster.

Windows end-to-end testing remains pending and this build contains no new
Windows installer. On Windows, the step-by-step mode with Claude Code passes a
command line longer than Windows allows; this is to be fixed before the next
Windows installer. No test establishes universal compatibility with every
cluster, engine configuration or material; scientific convergence checks remain
specific to the calculation.
