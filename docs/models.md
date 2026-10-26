# Connect a model

DFT Agent uses your own model account. The download includes no tokens, API
credits or compute allocation. An **API key** is your credential; **tokens**
measure what a model reads and generates. You enter a key, not a token count.

Create a project, then open **Connections → Model**. Choose a signed-in CLI,
an API provider, or a local model server. Connections can be reused in other
projects through **Use an existing connection → Connect to project**.

## Codex CLI or Claude Code CLI

These are the official command-line tools of OpenAI and Anthropic. They use a
plan you already have, so no API key is needed:

| CLI | Account | Official setup |
| --- | --- | --- |
| Codex CLI | ChatGPT Plus, Pro, Business, Edu or Enterprise | [Codex CLI](https://developers.openai.com/codex/cli) |
| Claude Code CLI | Claude Pro, Max, Team or Enterprise, or an Anthropic Console account | [Claude Code](https://code.claude.com/docs/en/setup) |

**Connections → Model** shows a card for each with three steps: install, sign
in, connect. The card offers only the next step:

1. **Install and sign in** opens a terminal window that runs the official
   installer and then the sign-in. The installers need neither Node.js nor
   administrator rights and put the program in your user folder.
2. The sign-in opens a page in your browser. Sign in there; the terminal says
   when it is done. Signing in to the chat website alone does not sign in the
   CLI.
3. Back in DFT Agent, the card notices the sign-in by itself within a few
   seconds and offers **Connect**, which tests the CLI with a real request.

A CLI that is installed but signed out offers **Sign in** instead.
Use **Stop checking** to stop watching the sign-in status. Nothing is
installed or signed in without that visible terminal window, and the sign-in
never passes through DFT Agent. If the window cannot open, or you prefer to
type yourself, **Do it yourself in a terminal** shows the same commands and
how to open a terminal. On macOS:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

```bash
codex login
```

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

```bash
claude auth login
```

On Windows, open PowerShell and use `irm https://chatgpt.com/codex/install.ps1 | iex`
or `irm https://claude.ai/install.ps1 | iex` instead. An installation that
needs network access the computer does not have, such as a blocked site,
stops with a message; an API key is the alternative then. The CLI uses its own
account and quota or billing. Nothing needs to be installed on the cluster for
model access.

CLI connections offer two modes, **Automatic** and **Step by step**. Automatic
lets the CLI inspect project files and use the app's tools across several
actions. Step by step processes one checked action at a time. Neither mode can authorize
its own calculation submission or QE installation: review and confirm those
in the app.

## API providers

Choose the provider below the CLI cards. Its card walks through three steps:

1. **Get an API key.** The card names the site, links to its key page and
   reminds you that API use is paid separately from chat memberships: add
   credit there first.
2. **Paste the key.** Spaces, line breaks and quotes that copying adds are
   removed.
3. **Connect.** The app checks the provider's model list and tries the known
   Qwen or GLM regions. It selects an available model when the default is
   missing, then tests it with a small structured request. If necessary, it
   tries up to two alternatives. These test requests use your account's tokens.
   Some accounts restrict model discovery; the app then tests the model directly.

Errors explain what to check next, such as the key, permissions, API credit
or network. Use **Change the model** on a saved connection to choose another.

| Provider | Key page | Address |
| --- | --- | --- |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | Preset |
| Claude | [platform.claude.com](https://platform.claude.com/settings/keys) | Preset |
| Qwen | [bailian.console.aliyun.com](https://bailian.console.aliyun.com/?apiKey=1), or [Model Studio](https://modelstudio.console.alibabacloud.com/ap-southeast-1/settings/api-key) outside mainland China | Found from the key |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/api_keys) | Preset |
| GLM | [bigmodel.cn](https://bigmodel.cn/usercenter/proj-mgmt/apikeys), or [z.ai](https://z.ai/manage-apikey/apikey-list) outside mainland China | Found from the key |
| Grok | [console.x.ai](https://console.x.ai/) | Preset |
| Compatible API | Your provider | Enter its address; `/v1` is tried when missing |

A Qwen workspace with its own address can enter it under **Advanced
settings**. **Compatible API** asks for the address, key and model name your
provider gives you, and lists the provider's models when it can. A
compatible name alone is insufficient: that route requires the
structured-output request format used by the app, which the test request
checks.

OpenAI uses Responses, Claude uses its native Messages API, and the other
presets use Chat Completions. API connections work **Step by step**. Every plan
is checked locally before it can become a calculation. A successful test
confirms the selected model and account, not every provider or model.

## Local model

Start an OpenAI-compatible server on this computer, for example Ollama:
install it from ollama.com and download a model with `ollama pull qwen3`.
Select **Local model**: servers of Ollama, LM Studio, llama.cpp and vLLM on
their usual ports are found with their models. Choose one and **Connect**.
For another port, use **Other local server** and enter the loopback API address
and model name. No API key is saved for this route. Small models may plan poorly; the test
request checks that the model gives structured answers. A remote server uses
**Compatible API** instead.

## Test and use it

A successful **Connect** test shows that the selected model answered. Use
**Test connection** to retry a saved connection. Return with **Back to chat**,
import a structure through **Files and runs → Files → Add files → Import files**, and
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
Automatic mode may read project files through its tools, so use a provider suitable
for the data in that project. Keep the app local.

## If a connection fails

| Symptom | What to check |
| --- | --- |
| CLI missing or signed out | Use the card's **Install and sign in** or **Sign in**; a saved connection then needs **Test connection**. |
| Key rejected or access denied | The card says which: copy the key again in full or make a new one with the link, or choose another model. |
| Model or endpoint not found | Check the exact model ID and address; create the corrected connection. |
| Quota, billing or rate limit | Check the provider's account dashboard; retry after the limit clears. |
| Timeout or network failure | Check connectivity and the provider endpoint. |
| Invalid or unsupported response | Choose a model supporting the required output format. No invalid plan is accepted. |

For model-free sample browsing or manual input preparation, open
**Settings → Classic workbench**. Those controls are separate from project chat.
