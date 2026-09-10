# tongflow-api-deepseek

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Routes text
generation through the [DeepSeek](https://platform.deepseek.com) V4 API — an
OpenAI-compatible, text-only model family.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

- **Generate / rewrite text** (`gen-text`) — create or edit copy from a prompt.
- **Combine text** (`combine-text`) — merge multiple text inputs into one.
- **Split long text** (`split-text`) — break a long passage into chunks.
- **Arrange & batch groups** (`arrange-group`) — group and arrange batches.
- **Filter or drop clips** (`drop-video`) — drop unwanted clips by rule.

## Models

Each text node exposes a **model dropdown** with two choices:

| Selection | Notes |
| --- | --- |
| `deepseek-v4-flash` (default) | cheap / fast |
| `deepseek-v4-pro` | strongest |

Thinking is a separate switch: expand the node's **Advanced** section and turn
on **Thinking**. The completion is then **streamed** and the model's reasoning
appears live in an auto-scrolling bubble beside the node; the final answer is
returned as the node output. The switch is declared in the plugin's
`TONGFLOW_SLOT_PARAMS` and read via `tongflow.slots.current_params()`; older
workflows that still carry a `<model>-thinking` id keep thinking on.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | ✅ | Create one at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys). |
| `DEEPSEEK_BASE_URL` | optional | Override the endpoint (defaults to `https://api.deepseek.com`). |

Values are stored locally and take effect without a restart.

## Development

Requires `tongflow>=0.3.3` (the Advanced **Thinking** switch uses
`current_params()`). Install into a clean venv and type-check:

```bash
pip install "tongflow==0.3.3"
pyright entry.py
```
