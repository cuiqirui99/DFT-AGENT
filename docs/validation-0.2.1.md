# v0.2.1 validation

Tested on 2026-10-07. This release adds result explanations and saved dialogue.
The solver workflows retain the [v0.2.0 validation](validation-0.2.0.md).

| Check | Result |
|---|---|
| Offline tests | 291 passed |
| Live model explanations | HSE06, SOC, Fe magnetic seeds and NiO +U passed |
| Rejected result | An older HSE band result was rejected despite its cached success status |
| Dialogue | Clarifications, revisions, failed requests, retries and saved follow-ups tested |
| Result changes | Changed outputs or plans invalidate saved explanations |
| Installed wheel | A live follow-up identified the accepted relaxed structure |

Five real Codex calls read retained solver outputs. The answers cited verified
facts, did not invent a band gap, and described Fe's ranking as a comparison of
tested seeds. No new solver jobs were submitted or saved calculations changed.
See the [responses and evidence](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/explanations-0.2.1.json).
The [wheel check](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/installed-explanation-0.2.1.json) ran
outside the source checkout and retained the prior result conversation.

The UI requires an explicit **Explain results** or **Ask** click. Explanations
do not submit jobs. OpenAI and compatible API adapters have offline tests;
live model checks used Codex CLI. These checks do not prove that every model
answer is correct or establish material-specific convergence.
