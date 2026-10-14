# Connect a model

Anyone can install DFT Agent. To run calculations, you need your own Slurm
account and either Quantum ESPRESSO or licensed VASP and POTCAR files. To use the agent, you also need one of
the model accounts below. The download does not include model credits or cluster
access. [Install and open the app](install.md), then choose a route.

Without a model account, you can still view sample results and prepare inputs
in **Manual** mode. The desktop app includes the SDKs; no Python installation
is needed to enter an API key.

An **API key** is a credential for your model account. **Tokens** measure the
text a model reads and generates. You enter a key, not a number of tokens.

## API providers

1. Open your provider's account page below. Create an API key and check billing
   or available credits. A chat subscription does not automatically include API use.
2. In **Model**, select the provider and paste its key into **API key**.
3. Enter an exact **Model name** available to your account. The examples below
   are starting points; model availability can change. Gray example text in
   the field is a hint, not a saved model name: enter the ID explicitly.
4. Leave **API URL** blank for the preset endpoint. Qwen needs your regional URL;
   GLM users on BigModel must replace the Z.AI default.

| Provider | Account and setup | Model example | API URL |
| --- | --- | --- | --- |
| OpenAI | [Create a key](https://platform.openai.com/api-keys) · [Billing](https://platform.openai.com/settings/organization/billing/overview) | `gpt-6.1-sol` | Leave blank |
| Claude | [Create a key](https://platform.claude.com/settings/keys) · [API guide](https://platform.claude.com/docs/en/get-started) | `claude-sonnet-4-6` | `https://api.anthropic.com` |
| Qwen | [Model Studio setup](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope) | `qwen-plus` | Copy your workspace's regional base URL |
| Grok | [xAI console](https://console.x.ai/) | `grok-4.7` | `https://api.x.ai/v1` |
| GLM | [Z.AI API guide](https://docs.z.ai/api-reference/llm/chat-completion) | `glm-5.3` | `https://api.z.ai/api/paas/v4` |
| DeepSeek | [DeepSeek API guide](https://api-docs.deepseek.com/en) | `deepseek-flash` | `https://api.deepseek.com` |

For Qwen, the key and URL must belong to the same region and workspace. Use the
OpenAI-compatible base URL from Model Studio, ending in `/compatible-mode/v1`.
For GLM through [BigModel in China](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction.md),
use `https://open.bigmodel.cn/api/paas/v4` and a BigModel key.
For Claude, create a key scoped to the workspace you will use.

Choose a text model. OpenAI needs Responses with strict JSON-schema output;
Grok uses strict JSON-schema output through Chat Completions. Claude uses its
native Messages API with [structured JSON output](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
Use a model that supports this feature, such as Sonnet 4.6. Qwen, GLM and DeepSeek
use JSON mode. All returned plans are checked locally; invalid or incomplete
responses cannot prepare a run.

Provider adapters have offline request tests. These do not establish live success
for every provider, model or account. Use the Si test below with your own account.

## Codex CLI

1. [Install Codex CLI](https://developers.openai.com/codex/cli) on the computer
   running DFT Agent.
2. In a terminal, run:

   ```bash
   codex login
   codex login status
   ```

3. Choose **Codex CLI** in **Model**. Leave **Model name** blank to use your
   configured model or the CLI default. API URL and API key are not used.

Signing in to ChatGPT in a browser does not sign the CLI in. Codex can use
eligible ChatGPT subscription access or an API key; API-key use is billed by
usage. See [Codex authentication](https://learn.chatgpt.com/docs/auth) for current
account options and limits. No Codex installation is needed on the cluster.

## Compatible API

Choose **Compatible API** and enter the provider's exact **Model name**, **API
URL** and **API key**. Use its API base URL, not its chat website or a full
`/chat/completions` endpoint.

The service must support Chat Completions with strict JSON-schema output and
the request options used by DFT Agent. An OpenAI-compatible label alone does
not guarantee this. Check the provider's documentation and billing dashboard.

## Try one plan

1. Open **New calculation**, choose **Example**, then **Si**.
2. Select **Agent** and enter `Relax this structure with PBE` in **Goal**.
3. Click **Plan**. A plan or a clarification question confirms a model response.
4. To change it, use **Change the plan** and click **Plan** again.

This test uses model tokens but needs no cluster connection and submits no VASP
job. To run the calculation, complete **Cluster setup**, review **Prepare
inputs**, then confirm **Submit calculation**. Follow the [quickstart](quickstart.md).

## Tokens and cost

Model calls happen when you click **Plan**, **Plan structure**, **Explain
results**, **Ask** or **Plan repair**, including follow-up requests. Input tokens
include your goal, structure, conversation and relevant guidance. Output tokens
cover the response, including reasoning where the provider reports it.

Preparing inputs, applying a structure edit, checking jobs, refreshing the page
and downloading results do not call a model. **Manual** calculations need no
model unless you choose a model action such as **Explain results**.

The app shows token counts after a response when available. Check your
[OpenAI usage](https://platform.openai.com/usage), Codex account limits or your
provider's dashboard for charges and remaining access. DFT Agent does not sell
tokens or set a spending cap. [Usage details](token-use.md).

API costs depend on the model's input and output rates and the tokens used.
Check your provider's prices before starting. Begin with the Si example and
review usage before larger requests.

## Keys and data

The key entered in the app stays in memory and is not saved in calculation
files. Tick **Remember on this computer** to store it in the system keychain
(macOS Keychain, Windows Credential Manager or a Linux Secret Service) and
have it filled in at the next start; untick to remove it. The command line
equivalents are `dft-agent key set PROVIDER`, `dft-agent key status` and
`dft-agent key clear PROVIDER`. **Clear API key** clears the field. Each provider has separate fields,
so switching providers does not send a previous provider's key to the new one.
Keep keys on the computer running the app; they are not needed on the cluster.

Instead of pasting a key, you can set the matching environment variable before
starting the app:

| Provider | Key variable |
| --- | --- |
| OpenAI / Compatible API | `DFT_AGENT_API_KEY` or `OPENAI_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| Qwen | `DASHSCOPE_API_KEY` |
| Grok | `XAI_API_KEY` |
| GLM | `ZAI_API_KEY` or `ZHIPUAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

A key entered in the app takes priority. Named providers use only their own
key variables. `DFT_AGENT_MODEL` and `DFT_AGENT_BASE_URL` apply to OpenAI and
Compatible API; enter other providers' settings in their fields or CLI options.

Model requests include the structure and task context needed for the action.
Result explanations use extracted results rather than entire output files.
Use a provider suitable for the data you send.

## If it fails

Use the reported error to choose the next step.

| Message or symptom | Check |
| --- | --- |
| API key rejected | Paste a valid key for the selected provider. |
| Access denied | Check model permissions and account or workspace access. |
| Model or endpoint not found | Check the exact model ID and API URL. |
| Quota or billing error | Check credits, billing and account limits with the provider. |
| Rate limit | Wait before requesting another plan. |
| Request format rejected | Check that the model and endpoint support structured output. |
| Timeout or connection failure | Check the network and API URL, then try again. |
| Provider unavailable | Try again later. |
| Invalid or incomplete output | No plan was accepted. Simplify the request or choose another model. |
| Codex not found or signed out | Install or update the CLI, then run `codex login status` in the app's terminal environment. |

Do not post API keys when reporting an error.
