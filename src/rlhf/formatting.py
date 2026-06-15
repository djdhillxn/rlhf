from typing import Any
import re


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}

_CHAT_START_RE = re.compile(r"<\|im_start\|>\s*(system|user|assistant)\s*", re.IGNORECASE)
_CHAT_END = "<|im_end|>"


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Some chat datasets store multimodal-ish content lists. Keep text parts.
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(value)


def _parse_embedded_qwen_chat(text: str) -> list[dict[str, str]] | None:
    """Parse strings that already contain Qwen-style chat markers.

    Some HelpSteer3 rows are OpenAI-message-like, but a small subset contains
    already-rendered chat text inside a message `content` field.  If we wrap that
    raw text inside a new user message, the final prompt can accidentally include
    an assistant answer before the generation point, e.g.:

        <|im_start|>user ... <|im_end|><|im_start|>assistant old answer ...

    That poisons SFT/RM/PPO prompts.  This parser recovers the embedded turns so
    strip_trailing_assistant can remove any existing final assistant answer.
    """
    if "<|im_start|>" not in text:
        return None
    matches = list(_CHAT_START_RE.finditer(text))
    if not matches:
        return None
    messages: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        role = ROLE_ALIASES.get(match.group(1).lower(), match.group(1).lower())
        content_start = match.end()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[content_start:next_start]
        end_idx = chunk.find(_CHAT_END)
        if end_idx >= 0:
            chunk = chunk[:end_idx]
        content = chunk.strip()
        if content:
            messages.append({"role": role, "content": content})
        elif role in {"system", "user"}:
            messages.append({"role": role, "content": ""})
    return messages or None


def normalize_messages(context: Any) -> list[dict[str, str]]:
    """Normalize HelpSteer-style context objects into OpenAI/Qwen chat messages.

    Handles the common shapes:
    - list[{"role": ..., "content": ...}]
    - list[{"from": ..., "value": ...}]
    - string prompt
    - dict with context/messages/conversation key
    - strings that already contain Qwen `<|im_start|>` chat markers
    """
    if isinstance(context, dict):
        for key in ("messages", "context", "conversation", "conversations", "turns"):
            if key in context:
                return normalize_messages(context[key])
        # Last-resort single user prompt from dict content-ish fields.
        text = context.get("content") or context.get("prompt") or context.get("value") or str(context)
        text = _stringify_content(text).strip()
        parsed = _parse_embedded_qwen_chat(text)
        if parsed is not None:
            return parsed
        return [{"role": "user", "content": text}]

    if isinstance(context, str):
        text = context.strip()
        parsed = _parse_embedded_qwen_chat(text)
        if parsed is not None:
            return parsed
        return [{"role": "user", "content": text}]

    if not isinstance(context, list):
        text = _stringify_content(context).strip()
        parsed = _parse_embedded_qwen_chat(text)
        if parsed is not None:
            return parsed
        return [{"role": "user", "content": text}]

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(context):
        if isinstance(item, str):
            parsed = _parse_embedded_qwen_chat(item)
            if parsed is not None:
                messages.extend(parsed)
                continue
            role = "user" if idx % 2 == 0 else "assistant"
            content = item
        elif isinstance(item, dict):
            role_raw = item.get("role") or item.get("from") or item.get("speaker") or item.get("author")
            role = ROLE_ALIASES.get(str(role_raw).lower(), "user" if idx % 2 == 0 else "assistant")
            content = item.get("content")
            if content is None:
                content = item.get("value") or item.get("text") or item.get("message")
            content_str = _stringify_content(content).strip()
            parsed = _parse_embedded_qwen_chat(content_str)
            if parsed is not None:
                messages.extend(parsed)
                continue
            content = content_str
        else:
            role = "user" if idx % 2 == 0 else "assistant"
            content = item
        content_str = _stringify_content(content).strip()
        if content_str or role in {"system", "user"}:
            messages.append({"role": role, "content": content_str})

    # Qwen chat templates are happiest when the last message before generation is
    # not an empty assistant response. If the dataset context already includes the
    # assistant answer being compared, data.py removes it before calling us.
    return messages or [{"role": "user", "content": ""}]


def strip_trailing_assistant(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove final assistant turns if present.

    Preference datasets sometimes put the candidate answer in the context. For
    reward modeling and PPO prompting, we want context ending before the compared
    assistant response.  We remove all trailing assistant turns defensively because
    embedded raw chat strings can contain an assistant answer followed by whitespace
    but no explicit closing marker.
    """
    out = list(messages)
    while out and out[-1].get("role") == "assistant":
        out = out[:-1]
    return out


def render_prompt(tokenizer: Any, context: Any, *, add_generation_prompt: bool = True) -> str:
    messages = normalize_messages(context)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback format for tokenizers without a chat template.
    chunks: list[str] = []
    for msg in messages:
        role = msg["role"].capitalize()
        chunks.append(f"{role}: {msg['content']}")
    if add_generation_prompt:
        chunks.append("Assistant:")
    return "\n".join(chunks).strip() + "\n"


def render_prompt_with_response(tokenizer: Any, context: Any, response: str) -> str:
    prompt = render_prompt(tokenizer, context, add_generation_prompt=True)
    eos = tokenizer.eos_token or ""
    return prompt + response.strip() + eos
