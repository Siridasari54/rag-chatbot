# utils/token_counter.py — NEW FILE

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def count_tokens(text: str) -> int:
    """
    Count tokens accurately using tiktoken.
    Falls back to word count if tiktoken not installed.
    """
    if _TIKTOKEN_AVAILABLE:
        return len(_enc.encode(text))
    # fallback — rough estimate, not accurate
    return len(text.split())


def count_tokens_in_messages(prompt: str, answer: str) -> dict:
    prompt_tokens     = count_tokens(prompt)
    completion_tokens = count_tokens(answer)
    return {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      prompt_tokens + completion_tokens,
        "method":            "tiktoken" if _TIKTOKEN_AVAILABLE else "word_count_estimate",
    }