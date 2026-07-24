"""TongFlow DeepSeek plugin — text generation via the DeepSeek V4 API.

DeepSeek V4 is a text-only, OpenAI-compatible model family
(``https://api.deepseek.com``). Two models — ``deepseek-v4-flash`` (cheap/fast)
and ``deepseek-v4-pro`` — each with an optional ``thinking`` mode. The node's
model dropdown exposes the four combinations; ``main()`` reads the selection
from the request envelope's top-level ``model`` field.

When a ``*-thinking`` model is selected the completion is streamed and the
reasoning (``reasoning_content``) is pushed to the node's live thinking bubble
via ``progress(..., thinking=True)``; the final answer (``content``) is returned.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.slots import node_slot
from tongflow.progress import progress
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.drop_video import DropVideoInput, DropVideoOutput
from tongflow.models.arrange_group import ArrangeGroupInput, ArrangeGroupOutput
from tongflow.llm_batch_handlers import arrange_group_output, drop_video_output

# Per-slot model lists surfaced as the node's model dropdown. Must stay a pure
# dict of list-of-string literals — the platform reads it by AST without
# importing this module, so every value has to be inlined (no shared variable).
# First entry per slot = default. The `-thinking` suffix is stripped by
# _resolve_selection() into (base model id, thinking on/off).
TONGFLOW_SLOT_MODELS = {
    "gen-text": [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash-thinking",
        "deepseek-v4-pro-thinking",
    ],
    "split-text": [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash-thinking",
        "deepseek-v4-pro-thinking",
    ],
    "combine-text": [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash-thinking",
        "deepseek-v4-pro-thinking",
    ],
}

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[deepseek] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.deepseek")

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
# Flush the streamed reasoning to the thinking bubble at most this often (chars
# accumulated since the last flush) and cap what we send to the tail so the
# progress message stays bounded.
_THINK_FLUSH_CHARS = 48
_THINK_TAIL_CHARS = 2000

# Model chosen on the node; set by main() from the request envelope. Empty →
# DEFAULT_MODEL with thinking off.
_REQUEST_MODEL: str = ""

_SYSTEM_PROMPT = (
    "You are a versatile text-generation assistant. Follow the user's "
    "instructions strictly and respond in the same language as the user."
)


def _base_url() -> str:
    return (
        os.environ.get("DEEPSEEK_BASE_URL") or ""
    ).strip().rstrip("/") or DEFAULT_BASE_URL


def _require_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Create one at "
            "https://platform.deepseek.com/api_keys and add it in TongFlow "
            "Settings."
        )
    return api_key


def _resolve_selection() -> Tuple[str, bool]:
    """Map the selected dropdown id to (base model id, thinking enabled)."""
    sel = (_REQUEST_MODEL or "").strip() or DEFAULT_MODEL
    if sel.endswith("-thinking"):
        return sel[: -len("-thinking")], True
    return sel, False


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _post(url: str, payload: Dict[str, Any], api_key: str, *, timeout: int = 300):
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(api_key),
        method="POST",
    )
    try:
        return urlopen(req, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error("HTTP %s on %s\nresponse body: %s", e.code, url, err_body)
        raise RuntimeError(
            f"HTTP {e.code} from DeepSeek: {err_body or e.reason}"
        ) from e
    except URLError as e:
        log.error("network error contacting %s: %s", url, e.reason)
        raise RuntimeError(f"Network error: {e.reason}") from e


def _chat(*, user_message: str) -> str:
    """One-shot chat completion. Streams reasoning to the thinking bubble when
    a thinking model is selected; returns the final answer text."""
    api_key = _require_api_key()
    model, thinking = _resolve_selection()
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 1,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    if thinking:
        payload["thinking"] = {"type": "enabled"}
        payload["stream"] = True
        return _chat_streaming(payload, api_key)

    url = f"{_base_url()}/chat/completions"
    log.info("POST %s model=%s", url, model)
    body = _post(url, payload, api_key).read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek response missing choices")
    content = ((choices[0] or {}).get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("DeepSeek response missing message.content")
    return content.strip()


def _iter_sse_data(resp) -> Any:
    """Yield parsed JSON objects from a text/event-stream body."""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            if data == "[DONE]":
                return
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _chat_streaming(payload: Dict[str, Any], api_key: str) -> str:
    url = f"{_base_url()}/chat/completions"
    log.info("POST %s model=%s (stream, thinking)", url, payload.get("model"))
    resp = _post(url, payload, api_key)

    answer_parts: List[str] = []
    reasoning = ""
    since_flush = 0
    for chunk in _iter_sse_data(resp):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        rc = delta.get("reasoning_content")
        if isinstance(rc, str) and rc:
            reasoning += rc
            since_flush += len(rc)
            # Send the accumulated (tail-capped) reasoning each tick; the UI
            # replaces its bubble content, so a dropped tick just skips ahead.
            if since_flush >= _THINK_FLUSH_CHARS:
                progress(reasoning[-_THINK_TAIL_CHARS:], thinking=True)
                since_flush = 0
        content = delta.get("content")
        if isinstance(content, str) and content:
            answer_parts.append(content)
    if reasoning:
        progress(reasoning[-_THINK_TAIL_CHARS:], thinking=True)
    return "".join(answer_parts).strip()


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    user_message = (
        f"{input.userPrompt or ''}\n\nUser input: {input.text}\n\n"
        "Note: output only the requested answer. Do not include any other content."
    )
    answer = _chat(user_message=user_message)
    return GenTextOutput(success=True, text=answer)


@node_slot(NodeSlots.COMBINE_TEXT)
def combine_text(input: CombineTextInput) -> CombineTextOutput:
    joined = "\n\n".join(input.texts)
    user_message = (
        f"{input.userPrompt or ''}\n\nUser input: {joined}\n\n"
        "Note: output only the requested answer. Do not include any other content."
    )
    answer = _chat(user_message=user_message)
    return CombineTextOutput(success=True, text=answer)


def _parse_split_texts(raw: str) -> list[str]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            _, _, s = s.partition("\n")
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    obj = json.loads(s)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        raw_items = obj.get("texts")
        items = raw_items if isinstance(raw_items, list) else None
    else:
        items = None
    if items is None or not all(isinstance(x, str) for x in items):
        raise ValueError("LLM did not return a JSON array of strings")
    cleaned = [x.strip() for x in items if x.strip()]
    if not cleaned:
        raise ValueError("LLM returned an empty split")
    return cleaned


def _build_split_user_message(input: SplitTextInput) -> str:
    instruction = (
        input.userPrompt or ""
    ).strip() or "Split into natural, coherent segments."
    return (
        f"Split the following text into multiple segments according to this instruction:\n"
        f"{instruction}\n\n"
        f"Return ONLY a JSON array of strings — no prose, no markdown, no keys, no code fences. "
        f"Each array element is one segment. Preserve the original wording; do not summarize.\n\n"
        f"TEXT:\n{input.text}"
    )


@node_slot(NodeSlots.SPLIT_TEXT)
def split_text(input: SplitTextInput) -> SplitTextOutput:
    raw = _chat(user_message=_build_split_user_message(input))
    try:
        texts = _parse_split_texts(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return SplitTextOutput(success=False, error=str(e))
    return SplitTextOutput(success=True, texts=texts)


@node_slot(NodeSlots.DROP_VIDEO)
def drop_video(input: DropVideoInput) -> DropVideoOutput:
    result = drop_video_output(input.model_dump())
    return DropVideoOutput.model_construct(**result)


@node_slot(NodeSlots.ARRANGE_GROUP)
def arrange_group(input: ArrangeGroupInput) -> ArrangeGroupOutput:
    result = arrange_group_output(input.model_dump())
    return ArrangeGroupOutput.model_construct(**result)


# Runtime dispatcher. The @node_slot wrapper accepts a raw dict here (it
# deep-constructs the typed BaseModel internally) and dumps the BaseModel
# return to a dict. `Any` reflects the I/O boundary, not the plugin contract.
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.COMBINE_TEXT: combine_text,
    NodeSlots.SPLIT_TEXT: split_text,
    NodeSlots.DROP_VIDEO: drop_video,
    NodeSlots.ARRANGE_GROUP: arrange_group,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    global _REQUEST_MODEL
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""
        _REQUEST_MODEL = (
            str(req.get("model") or "").strip() if isinstance(req, dict) else ""
        )

        handler = _SLOT_HANDLERS.get(slot)
        if handler is None:
            raise RuntimeError(f"unsupported nodeSlot: {slot!r}")
        out = handler(prompt)
    except Exception as e:  # noqa: BLE001 — surfaced as ABI failure
        _write({"success": False, "error": str(e)})
        return 1

    _write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
