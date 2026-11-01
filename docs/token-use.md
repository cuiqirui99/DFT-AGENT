# Model use

[Set up your model account](models.md) before making your first request.
An API key grants access; tokens measure input and output. No tokens are bundled
with DFT Agent. Check your provider's dashboard for charges and remaining usage.
OpenAI, Claude, Qwen, Grok, GLM and DeepSeek each use your own provider account.

DFT Agent calls a model when you request a calculation or structure plan,
revision, explanation or repair. Applying structure edits, format conversion,
input checks, Slurm polling, downloads and plots run locally.
Refreshing the app does not repeat a model request. Unknown failures are rejected
locally when no supported repair exists.

## Context

- Keep the original goal and every user correction in the current conversation.
- Remove repeated report metadata from assistant history.
- Send complete indexed site tables; never shorten or round the geometry.
- Retrieve a few verified runs. Large advisory vectors stay in the saved report,
  with explicit references instead of values the model might copy.
- Explain verified facts instead of sending entire OUTCAR or XML files.

Reports and source checks stay on disk. Context compression uses no extra model
call. Task context over 96,000 UTF-8 bytes stops before calling the provider; it is
not silently truncated. History has no message-count cutoff. If the context exceeds
the byte limit, start again with the reviewed settings you want to retain.

API output budgets start at 8,192 tokens and grow for larger site lists. Set
`DFT_AGENT_MAX_OUTPUT_TOKENS` to use a provider-specific budget. Incomplete output
is rejected. Codex uses its CLI output budget; the API setting does not affect it.

## Usage

Each response records `model_usage`: input, output, cached input and reasoning
tokens when the provider reports them, plus latency and character counts.
Missing token counts are **unknown**, not zero. Character counts are not token
estimates. Cached tokens are part of input tokens; reasoning tokens are part of
output tokens. Do not add either twice.

For Claude, displayed input includes fresh tokens, cache reads and cache writes.
For Grok, displayed output includes reasoning tokens, which xAI reports separately
from visible completions. Its output limit is not a spending cap.

The app shows a short usage line. Saved plans, explanations and repairs retain
the record. Failed responses can also carry reported usage; a connection failure
may leave it unknown. No automatic retry hides extra calls.

Static instructions come before changing task data. Cache reuse depends on the
provider and model; the reported cached count shows what actually happened.
Pricing and speed vary, so smaller context alone is not a cost or latency promise.

[Responses usage](https://developers.openai.com/api/reference/resources/responses/methods/create),
[Codex JSON events](https://developers.openai.com/codex/noninteractive), and
[prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
describe the provider fields used here.
