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

Each text node exposes a **model dropdown** with four choices:

| Selection | Model | Thinking |
| --- | --- | --- |
| `deepseek-v4-flash` (default) | `deepseek-v4-flash` | off |
| `deepseek-v4-pro` | `deepseek-v4-pro` | off |
| `deepseek-v4-flash-thinking` | `deepseek-v4-flash` | on |
| `deepseek-v4-pro-thinking` | `deepseek-v4-pro` | on |

With a `*-thinking` selection the completion is **streamed** and the model's
reasoning appears live in an auto-scrolling bubble beside the node; the final
answer is returned as the node output.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | ✅ | Create one at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys). |
| `DEEPSEEK_BASE_URL` | optional | Override the endpoint (defaults to `https://api.deepseek.com`). |

Values are stored locally and take effect without a restart.

## Development

Requires `tongflow>=0.2.17` (the streaming thinking bubble uses
`progress(..., thinking=True)`). Install into a clean venv and type-check:

```bash
pip install "tongflow==0.2.17"
pyright entry.py
```
