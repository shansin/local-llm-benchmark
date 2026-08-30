# coding / parse-logs

**Prompt:** Write a Python function called `summarize_logs` that takes a list of log line strings and returns a summary dictionary.

Each well-formed log line looks like:
  `2026-04-11T08:15:30Z LEVEL service_name message text here`

where LEVEL is one of DEBUG, INFO, WARN, ERROR.

The function must return a dict with exactly these keys:
- `"total"` — number of lines successfully parsed
- `"malformed"` — number of lines that could not be parsed
- `"by_level"` — dict mapping each level that appeared to its count
- `"services"` — sorted list of distinct service names seen
- `"first_error"` — the message text of the first ERROR line, or None if there were none

Requirements:
- Malformed lines must be counted, not crash the function
- Include type hints
- Do not use regular expressions with catastrophic backtracking; keep it simple

## Repeat 1 (seed 0)

- Score: 9.75/10 — Function matches spec, handles malformed lines, returns exact keys with correct types, type hints present, and all 9 tests pass; clarity slightly reduced by extra timestamp validation beyond minimal requirement.
- Tokens/s: 93.2
- Prefill tok/s: 1196.7
- TTFT: 25.63s
- Gen Time: 29.73s
- Output Tokens: 2743
- answer read from `answer_tag`

<details><summary>Reasoning (8424 chars, not scored)</summary>

Let me think through this carefully.

I need to write a Python function `summarize_logs` that:
1. Takes a list of log line strings
2. Returns a summary dictionary with specific keys

Log line format: `2026-04-11T08:15:30Z LEVEL service_name message text here`

Let me parse this:
- Timestamp: `2026-04-11T08:15:30Z` - this is an ISO 8601 timestamp
- LEVEL: one of DEBUG, INFO, WARN, ERROR
- service_name: a single token (no spaces)
- message: the rest of the line (can contain spaces)

So the format is: `TIMESTAMP LEVEL SERVICE MESSAGE...`

Splitting by spaces:
- parts[0] = timestamp
- parts[1] = level
- parts[2] = service_name
- parts[3:] = message (joined back with spaces)

Validation:
- Must have at least 4 parts (timestamp, level, service, at least one word of message)
- Level must be one of DEBUG, INFO, WARN, ERROR
- Timestamp should look valid (I'll check basic format - maybe just check it has the right structure)

Wait, let me reconsider. The message is "message text here" - could it be empty? The format says "message text here" which implies there's at least some message. But let me be safe - I'll require at least 4 parts (timestamp, level, service, and at least one message word). Actually, re-reading: "message text here" - this is the message part. I think requiring at least 4 parts is reasonable.

Actually, let me reconsider. What if the message is empty? The format shows "message text here" as the message. I'll require at least 4 parts to be safe (timestamp, level, service, and at least one message token).

For the timestamp validation - I should check it looks like an ISO timestamp. I can use `datetime.fromisoformat` but that might be overkill. Let me just check basic structure: it should match the pattern of `YYYY-MM-DDTHH:MM:SSZ`. I can do a simple check without regex - split by 'T' and check parts, or just try to parse it with datetime.

Actually, the requirement says "Do not use regular expressions with catastrophic backtracking; keep it simple". This means I can use simple regex or no regex at all. Let me just use string splitting and basic validation.

For timestamp validation, I'll try `datetime.fromisoformat` (Python 3.7+). But wait, the 'Z' suffix might not be handled by all versions. In Python 3.11+, `fromisoformat` handles 'Z'. For earlier versions, I might need to replace 'Z' with '+00:00'. Let me just do a basic structural check:
- Length should be 20 (YYYY-MM-DDTHH:MM:SSZ)
- Or I can try to parse it

Actually, let me keep it simple. I'll validate:
1. At least 4 space-separated parts
2. Level is in {DEBUG, INFO, WARN, ERROR}
3. Timestamp looks reasonable (I'll try datetime parsing or basic structure check)

Let me go with trying to parse the timestamp using datetime. If it fails, it's malformed.

Wait, but the requirement says "keep it simple" regarding regex. Using datetime is fine.

Let me think about what makes a line malformed:
- Not enough parts (< 4)
- Level not in the valid set
- Timestamp doesn't parse (optional - maybe I should be lenient here?)

Actually, re-reading the problem: "Each well-formed log line looks like: `2026-04-11T08:15:30Z LEVEL service_name message text here`". The key validation points are:
- It has the right structure (4+ parts)
- Level is valid
- The timestamp is in the right format

I'll validate:
1. Split by space, need at least 4 parts
2. Level must be in {DEBUG, INFO, WARN, ERROR}
3. Timestamp: I'll try to parse it with datetime (replacing Z with +00:00 for compatibility)

Actually, let me be a bit more lenient on the timestamp. The main structural requirements are the 4 parts and valid level. Let me validate the timestamp format loosely - check it starts with a 4-digit year, has the right separators, etc. Or just try datetime parsing.

Let me go with: try to parse timestamp with datetime, if it fails, mark as malformed.

For the return dict:
- `"total"`: count of successfully parsed lines
- `"malformed"`: count of lines that couldn't be parsed
- `"by_level"`: dict mapping level -> count (only levels that appeared)
- `"services"`: sorted list of distinct service names
- `"first_error"`: message text of first ERROR line, or None

Let me write the code:

```python
from datetime import datetime
from typing import Optional

def summarize_logs(log_lines: list[str]) -> dict:
    valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"}
    
    total = 0
    malformed = 0
    by_level: dict[str, int] = {}
    services: set[str] = set()
    first_error: Optional[str] = None
    
    for line in log_lines:
        parts = line.split(" ")
        # Need at least 4 parts: timestamp, level, service, message
        if len(parts) < 4:
            malformed += 1
            continue
        
        timestamp_str = parts[0]
        level = parts[1]
        service = parts[2]
        message = " ".join(parts[3:])
        
        # Validate level
        if level not in valid_levels:
            malformed += 1
            continue
        
        # Validate timestamp
        try:
            # Handle Z suffix for Python < 3.11
            ts = timestamp_str
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            malformed += 1
            continue
        
        # Successfully parsed
        total += 1
        by_level[level] = by_level.get(level, 0) + 1
        services.add(service)
        
        if level == "ERROR" and first_error is None:
            first_error = message
    
    return {
        "total": total,
        "malformed": malformed,
        "by_level": by_level,
        "services": sorted(services),
        "first_error": first_error,
    }
```

Wait, I should also consider: what if a line is not a string? The type hint says `list[str]` so I'll assume they're strings. But to be safe against crashes, maybe I should handle non-string inputs? The requirement says "Malformed lines must be counted, not crash the function". I think this refers to lines that don't match the format, not non-string types. But let me add a check just in case.

Actually, let me also consider: what if `line` has leading/trailing spaces? `split(" ")` would create empty strings. Let me use `split()` instead which splits on any whitespace and ignores leading/trailing. But wait, the message part might have multiple spaces that we want to preserve... Hmm, actually for the message, we just need the text. Let me use `split(" ", 3)` to split into at most 4 parts, preserving the message as-is.

Wait, `split(" ", 3)` would give:
- parts[0] = timestamp
- parts[1] = level
- parts[2] = service
- parts[3] = rest of the line (the message)

This is better! It preserves the message exactly. But what about leading spaces? If the line starts with a space, parts[0] would be empty. Let me strip the line first.

Actually, let me use `line.strip().split(" ", 3)`. This handles leading/trailing whitespace and splits into at most 4 parts.

But wait - what if there are multiple spaces between fields? Like `2026-04-11T08:15:30Z  INFO  service  message`. With `split(" ", 3)`, this would give `['2026-04-11T08:15:30Z', '', 'INFO', 'service  message']` which would fail validation (empty level). That's probably fine - it's malformed.

Alternatively, I could use `split(None, 3)` which splits on any whitespace and ignores multiple spaces. Let me use that for robustness.

`"  2026-04-11T08:15:30Z  INFO  service  hello  world  ".split(None, 3)` → `['2026-04-11T08:15:30Z', 'INFO', 'service', 'hello  world']`

Yes, that's better. Let me use `split(None, 3)`.

Let me also think about edge cases:
- Empty string: `"".split(None, 3)` → `[]` → len < 4 → malformed ✓
- Only 3 parts: malformed ✓
- Invalid level: malformed ✓
- Invalid timestamp: malformed ✓

I think this is solid. Let me finalize.

One more thought: the type hint. In Python 3.9+, I can use `list[str]` and `dict[str, int]`. For broader compatibility, I could use `List[str]` and `Dict[str, int]` from typing. Let me use the modern syntax since the problem doesn't specify a Python version, and modern Python (3.9+) is reasonable.

Actually, let me use `from __future__ import annotations` or just use the typing module to be safe. I'll use `List` and `Dict` from typing for maximum compatibility, or just use the built-in generics. Let me go with the built-in generics (3.9+) since it's cleaner.

Let me finalize my solution.

</details>

**Answer (as scored):**

```python
from datetime import datetime
from typing import Optional


def summarize_logs(log_lines: list[str]) -> dict:
    """Parse a list of log lines and return a summary dictionary.

    Expected line format:
        2026-04-11T08:15:30Z LEVEL service_name message text here
    where LEVEL is one of DEBUG, INFO, WARN, ERROR.
    """
    valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"}

    total = 0
    malformed = 0
    by_level: dict[str, int] = {}
    services: set[str] = set()
    first_error: Optional[str] = None

    for line in log_lines:
        # Guard against non-string entries
        if not isinstance(line, str):
            malformed += 1
            continue

        # Split into at most 4 parts on any whitespace:
        # [timestamp, level, service, message]
        parts = line.split(None, 3)

        if len(parts) < 4:
            malformed += 1
            continue

        timestamp_str, level, service, message = parts

        # Validate level
        if level not in valid_levels:
            malformed += 1
            continue

        # Validate timestamp (ISO 8601 with optional Z suffix)
        try:
            ts = timestamp_str
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            malformed += 1
            continue

        # Successfully parsed
        total += 1
        by_level[level] = by_level.get(level, 0) + 1
        services.add(service)

        if level == "ERROR" and first_error is None:
            first_error = message

    return {
        "total": total,
        "malformed": malformed,
        "by_level": by_level,
        "services": sorted(services),
        "first_error": first_error,
    }
```

