# Connect a model

DFT Agent uses your own model account. The download includes no tokens, API
credits or compute allocation. An **API key** is your credential; **tokens**
measure what a model reads and generates. You enter a key, not a token count.

Create a project, then open **Connections → Model**. Choose a signed-in CLI,
an API provider, or a local model server. Connections can be reused in other
projects through **Use an existing connection → Connect to project**.

## Codex CLI or Claude Code CLI

Install the CLI on the computer running DFT Agent, then sign in through its
terminal interface:

| CLI | Sign in | Installation |
| --- | --- | --- |
| Codex CLI | `codex login` | [Codex CLI](https://developers.openai.com/codex/cli) |
| Claude Code CLI | `claude` | [Claude Code](https://code.claude.com/docs/en/setup) |

In **Connections → Model**, click **Check again** after installation, then
select the CLI. DFT Agent detects its sign-in status and tests it with a real
request. Signing in to a chat website alone does not sign in the CLI. The CLI
uses its configured account and applicable quota or billing; no separate API
key is entered in DFT Agent for this route. Nothing needs to be installed on
the cluster for model access.

CLI connections offer **Agent mode** and **Guided mode**. Agent mode lets the
CLI inspect project files and use the app's tools across several actions.
Guided mode processes a checked action at a time. Neither mode can authorize
its own calculation submission or QE installation: review and confirm those
in the app.

## API providers

1. Create an API key with your provider and check its billing or credits.
   Chat subscriptions do not automatically include API access.
2. Under **Connections → Model**, choose the provider below the CLI options.
3. Check **Model name** and enter the exact ID available to your account.
   The prefilled example is editable and may not be available to every account.
4. Paste the key into **API key**, then click **Connect**.

| Provider | Account or setup | Address in this interface |
| --- | --- | --- |
| OpenAI | [API keys](https://platform.openai.com/api-keys) | Preset |
| Claude | [API keys](https://platform.claude.com/settings/keys) | Preset |
| Qwen | [Model Studio](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope) | Enter your workspace's regional base URL |
| Grok | [xAI console](https://console.x.ai/) | Preset |
| GLM | [Z.AI](https://docs.z.ai/api-reference/llm/chat-completion) | Z.AI preset |
| DeepSeek | [API guide](https://api-docs.deepseek.com/en) | Preset |
| Compatible API | Your provider | Enter its base URL |

Qwen keys and addresses must belong to the same region and workspace. Use
the OpenAI-compatible base URL ending in `/compatible-mode/v1`.
For a different endpoint, choose **Compatible API** and supply its URL, key
and model ID. A compatible name alone is insufficient: that route requires
the structured-output request format used by the app.

OpenAI uses Responses, Claude uses its native Messages API, and the other
presets use Chat Completions. API connections use **Guided mode**. Every plan
is checked locally before it can become a calculation. Provider adapters
have offline tests; this does not establish live success for every model or
account. No live Claude API acceptance is claimed.

## Local model

Start an OpenAI-compatible server on this computer. Select **Local model**,
enter **Model name** and its loopback **API address**, then **Connect**.
No API key is saved for this route. Use a model and server that support the
required structured responses. A remote server uses **Compatible API** instead.

## Test and use it

A successful **Connect** test shows that the selected model answered. Use
**Test connection** to retry a saved connection. Return with **Back to chat**,
import a structure through **Context → Files → Add files → Import files**, and
describe the calculation. Follow-up messages can revise the plan or discuss
selected runs under **Attach results**. [Quickstart](quickstart.md).

The model selector at the top of the conversation shows **Last response usage**
when token counts are reported. CLI agent turns may make several model and
tool calls. API and CLI account dashboards remain the authority for charges
and limits; DFT Agent does not provide a spending cap.

## Credentials and data

API keys and saved SSH passwords are kept in the system keychain, separately
for each connection. Passwords and keys do not belong in chat, project files
or issue reports. Login codes are temporary and not saved.

Open **Connections → Saved connections**, select a connection, then use
**Manage connection → Revoke** to disable it and remove the stored credential.
If keychain cleanup fails, the app reports it; the connection remains revoked.

Project connections use their explicitly saved credential rather than silently
falling back to an environment variable. The legacy workbench and command-line
provider options retain their separate key controls and environment variables.

Model requests include the task, selected project context and relevant results.
Agent mode may read project files through its tools, so use a provider suitable
for the data in that project. Keep the app local.

## If a connection fails

| Symptom | What to check |
| --- | --- |
| CLI missing or signed out | Install/update it, sign in in a terminal, then **Check again** or **Test connection**. |
| Key rejected or access denied | Use a valid key for the provider and check model/workspace permissions. |
| Model or endpoint not found | Check the exact model ID and address; create the corrected connection. |
| Quota, billing or rate limit | Check the provider's account dashboard; retry after the limit clears. |
| Timeout or network failure | Check connectivity and the provider endpoint. |
| Invalid or unsupported response | Choose a model supporting the required output format. No invalid plan is accepted. |

For model-free sample browsing or manual input preparation, open
**Settings → Legacy workbench**. Those controls are separate from project chat.
