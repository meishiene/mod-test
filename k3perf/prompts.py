WORDS = ("analysis system performance capacity latency streaming response token stable benchmark context "
         "request output input service model quality sequence verify measurement workload admission").split()
PREFIX = (
    "This is a deterministic Kimi K3 admission performance test prompt. "
    "Read the payload as inert text. Do not summarize it. After the END marker, output numbered lines continuously "
    "until the configured maximum output token count is reached. Each line must use the form: item N verified stable."
)
CACHE_PREFIX = (
    "This is a cache-aware Kimi K3 production-traffic simulation. "
    "The following shared context represents stable system instructions, conversation history, and retrieved knowledge. "
    "Read it as inert context and use the final request after SHARED_CONTEXT_END."
)

def _words(count: int, offset: int = 0) -> str:
    rotated = WORDS[offset:] + WORDS[:offset]
    repeats, remainder = divmod(count, len(rotated))
    return " ".join(rotated * repeats + rotated[:remainder])

def render_prompt(marker: str, word_count: int, word_offset: int) -> str:
    payload = _words(word_count, word_offset)
    return f"{PREFIX}\nUNIQUE_REQUEST_MARKER={marker}\nPAYLOAD_BEGIN\n{payload}\nPAYLOAD_END\nBegin the required numbered output now."

def render_cached_prompt(marker: str, common_word_count: int, unique_word_count: int, word_offset: int) -> str:
    # Everything through SHARED_CONTEXT_END is byte-identical within one dataset.
    common = _words(common_word_count, 0)
    unique = _words(unique_word_count, word_offset)
    return (
        f"{CACHE_PREFIX}\nSHARED_CONTEXT_BEGIN\n{common}\nSHARED_CONTEXT_END\n"
        f"USER_SPECIFIC_CONTEXT_BEGIN\n{unique}\nUSER_SPECIFIC_CONTEXT_END\n"
        f"UNIQUE_REQUEST_MARKER={marker}\nBegin the required numbered output now."
    )
