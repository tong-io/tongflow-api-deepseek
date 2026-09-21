# tongflow-api-deepseek

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Routes text
generation and image understanding through the
[DeepSeek](https://platform.deepseek.com) API — an OpenAI-compatible model
family. Since **V4.1 Flash** (10 Sep 2026) the Flash model understands images
natively, so the plugin also serves the two vision → text slots.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

- **Generate / rewrite text** (`gen-text`) — create or edit copy from a prompt.
- **Combine text** (`combine-text`) — merge multiple text inputs into one.
- **Split long text** (`split-text`) — break a long passage into chunks.
- **Image → text** (`image-gen-text`) — ask anything about an image.
- **Describe image** (`image-describe`) — caption an image.
- **Arrange & batch groups** (`arrange-group`) — group and arrange batches.
- **Filter or drop clips** (`drop-video`) — drop unwanted clips by rule.

## Models

Each node exposes a **model dropdown**:

| Selection | Notes |
| --- | --- |
| `deepseek-flash` (default) | V4.1 Flash — cheap / fast, 1M context, sees images |
| `deepseek-v4-pro` | strongest; text-only |

The vision slots list `deepseek-flash` alone, because V4 Pro cannot see images.
The retired `deepseek-v4-flash` / `deepseek-v4-flash-vision-exp` ids saved in
older workflows are mapped to `deepseek-flash` by the plugin, so those graphs
keep running once DeepSeek's own temporary routing is switched off.

Thinking is a separate switch: expand the node's **Advanced** section and turn
on **Thinking**. The completion is then **streamed** and the model's reasoning
appears live in an auto-scrolling bubble beside the node; the final answer is
returned as the node output. The switch is declared in the plugin's
`TONGFLOW_SLOT_PARAMS` and read via `tongflow.slots.current_params()`; older
workflows that still carry a `<model>-thinking` id keep thinking on.

The vision slots add an **Image detail** knob under Advanced: `low` downscales
the image to 512×512 (faster, cheaper), `high` keeps the original size, `auto`
(default) lets DeepSeek choose.

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
