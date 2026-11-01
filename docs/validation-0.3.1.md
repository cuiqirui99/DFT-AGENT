# v0.3.1 validation

Memory checks completed on 9 October 2026.

| Check | Result |
|---|---|
| Offline tests | 523 passed; no failures or skips |
| Built-in records | 20 verified and 21 inactive; all evidence hashes checked |
| Local import | Actual legacy files previewed without changes: 68 experience records, four failures and 50 lessons |
| Selective import | JSONL, SQLite and current saved runs; repeat imports do not duplicate records |
| Isolation | Rejected and unverified lessons excluded; altered run outputs lose successful facts |
| Installation | Wheel installed outside the checkout; all 117 package files match source, and retrieval works |
| Live model | HSE06 planning retained the requested settings; repair proposed only NELM=1 to 120 |

The initial live repair asked for SCF history that the advice required but the
payload did not provide. The final version supplies current reparsed checks and
uses conditions supported by the executor. The repeated planning and repair
checks passed. Four real Codex calls were made across these two rounds.

Inputs and result receipts for the bundled cases were checked against the
existing [v0.3.0 campaign](validation-0.3.0.md). No new HPC jobs were submitted,
and no new material-accuracy claim is made. Raw VASP outputs are retained locally;
the package contains inputs, summaries and output hashes.

[Machine-readable record](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/memory-0.3.1.json) ·
[Memory sources](memory-sources.md)
