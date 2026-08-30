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

- Score: 9.75/10 — Function meets all spec, passes all tests, handles malformed lines, includes type hints, returns exact keys with correct semantics; clarity slightly reduced by extra timestamp validation beyond minimal requirement but still clear.
- Tokens/s: 205.3
- Prefill tok/s: 2553.8
- TTFT: 0.13s
- Gen Time: 25.86s
- Output Tokens: 5281
- answer read from `answer_tag`

<details><summary>Reasoning (16828 chars, not scored)</summary>

Let me analyze this problem carefully.

I need to write a Python function `summarize_logs` that:
1. Takes a list of log line strings
2. Returns a summary dictionary with specific keys

The log line format is:
`2026-04-11T08:15:30Z LEVEL service_name message text here`

The returned dict must have exactly these keys:
- `"total"` — number of lines successfully parsed
- `"malformed"` — number of lines that could not be parsed
- `"by_level"` — dict mapping each level that appeared to its count
- `"services"` — sorted list of distinct service names seen
- `"first_error"` — the message text of the first ERROR line, or None if there were none

Requirements:
- Malformed lines must be counted, not crash the function
- Include type hints
- Do not use regular expressions with catastrophic backtracking; keep it simple

Let me think about the parsing logic.

A well-formed log line:
`2026-04-11T08:15:30Z LEVEL service_name message text here`

So splitting by whitespace:
- Part 0: timestamp (e.g., `2026-04-11T08:15:30Z`)
- Part 1: LEVEL (DEBUG, INFO, WARN, ERROR)
- Part 2: service_name
- Part 3+: message text

For a line to be well-formed:
- It must have at least 4 parts (timestamp, level, service, message)
- The level must be one of DEBUG, INFO, WARN, ERROR
- The timestamp should be valid? The problem says "well-formed log line". Let me think about what makes a line malformed.

Let me consider what makes a line malformed:
- Not enough fields (less than 4)
- Invalid level (not one of the four)
- Maybe invalid timestamp format

Let me be reasonable here. The key requirements are:
- Count malformed lines
- Don't crash

Let me think about the timestamp validation. The format is ISO 8601: `2026-04-11T08:15:30Z`. I could validate this but it might be overkill. Let me include a basic check.

Actually, let me think about what's the minimal reasonable validation. The problem says "well-formed log line". I'll validate:
1. At least 4 whitespace-separated tokens
2. The level is valid
3. The timestamp looks like a valid ISO timestamp

Let me use `datetime.fromisoformat` for timestamp validation. But note that `fromisoformat` in Python 3.11+ handles the 'Z' suffix. In earlier versions, it doesn't. To be safe, I could replace 'Z' with '+00:00'.

Actually, let me keep it simple but robust. Let me think about the approach.

Let me write the function:

```python
from datetime import datetime
from typing import List, Dict


def summarize_logs(logs: List[str]) -> Dict[str, object]:
    total = 0
    malformed = 0
    by_level: Dict[str, int] = {}
    services: set = set()
    first_error = None

    valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"}

    for line in logs:
        parts = line.split()
        # Need at least: timestamp, level, service, message
        if len(parts) < 4:
            malformed += 1
            continue

        timestamp_str, level, service = parts[0], parts[1], parts[2]
        message = " ".join(parts[3:])

        # Validate level
        if level not in valid_levels:
            malformed += 1
            continue

        # Validate timestamp (basic ISO 8601 check)
        try:
            # Handle 'Z' suffix for UTC
            ts = timestamp_str.replace("Z", "+00:00")
            datetime.fromisoformat(ts)
        except ValueError:
            malformed += 1
            continue

        # Line is well-formed
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

Wait, there's an issue with the timestamp validation. `datetime.fromisoformat` in Python 3.7-3.10 doesn't accept the 'Z' suffix. My replace handles that. But there might be other edge cases. Let me think.

Actually, `datetime.fromisoformat` is quite strict. For `2026-04-11T08:15:30+00:00`, it works. For `2026-04-11T08:15:30Z` after replace becomes `2026-04-11T08:15:30+00:00`, which works.

But what about a timestamp like `2026-04-11` (date only)? That would parse fine with fromisoformat. Hmm. But that's probably not a concern for this problem.

Let me reconsider whether I should validate the timestamp at all. The problem statement focuses on the structure. Let me think about what "malformed" means. 

Actually, I think validating the level is the most important thing. The timestamp validation adds robustness. Let me keep a reasonable timestamp check.

Hmm, but there's a subtle issue. What if the timestamp is something weird like `notatimestamp`? My check would catch it via fromisoformat raising ValueError. Good.

But wait, what about a timestamp like `2026-13-45T99:99:99Z`? fromisoformat would raise ValueError because month 13 is invalid. Good, that's caught.

Let me reconsider the "catastrophic backtracking" requirement. This is about regex. Since I'm not using regex at all, I'm fine. The requirement is essentially telling me to keep parsing simple, which I'm doing with `.split()`.

Let me double check the return type. The dict has mixed value types (int, dict, list, object). Using `Dict[str, object]` is a bit loose. Let me use a more precise type. Actually, I could use `Dict[str, Any]` from typing.

Let me refine:

```python
from datetime import datetime
from typing import Any, Dict, List


def summarize_logs(logs: List[str]) -> Dict[str, Any]:
    ...
```

Let me think about edge cases:
- Empty list: returns total=0, malformed=0, by_level={}, services=[], first_error=None. Good.
- All malformed: total=0, malformed=N. Good.
- ERROR with empty message? A line like `2026-04-11T08:15:30Z ERROR service` has only 3 parts, so it's malformed (needs at least 4). So first_error would only be set when there's a message. That seems reasonable.

Wait, should a line with no message but valid level/service count as malformed? The format requires "message text here", so yes, a message is expected. With my `len(parts) < 4` check, a line with no message would be malformed. That's consistent.

Let me reconsider: what if message is required? The format shows `message text here` which implies at least one word. So 4 tokens minimum is correct.

Let me also think about whether I should strip the line first. `split()` without arguments handles leading/trailing whitespace and multiple spaces. So that's fine.

Let me also consider: what if a line is empty string? `"".split()` returns `[]`, len 0 < 4, malformed. Good.

I think the function is solid. Let me also think about whether the timestamp validation might be too strict and reject valid lines. 

Consider `2026-04-11T08:15:30Z`. After replace: `2026-04-11T08:15:30+00:00`. `datetime.fromisoformat("2026-04-11T08:15:30+00:00")` works in Python 3.7+. Good.

What about fractional seconds? `2026-04-11T08:15:30.123Z` → `2026-04-11T08:15:30.123+00:00`. fromisoformat handles this in 3.7+. Good.

I think this is robust enough.

Actually, let me reconsider the timestamp validation once more. There's a potential issue: in Python 3.11+, `fromisoformat` is more lenient and accepts 'Z' directly, but my replace of 'Z' with '+00:00' still works fine (it would replace the Z and then parse). Actually wait, if there's no Z, nothing happens. If there's a Z, it becomes +00:00. Good.

But what if the timestamp has lowercase 'z'? Unlikely for ISO format. I'll not worry about it.

Let me finalize. Actually, let me reconsider whether to validate the timestamp strictly or just check it's a plausible ISO format. I think using fromisoformat is good and simple.

Hmm, one concern: `datetime.fromisoformat` might accept things that aren't really ISO 8601 in the strict sense, but for this purpose it's fine.

Let me also make sure I handle the case where the message could contain the word that looks like a level. E.g., `2026-04-11T08:15:30Z INFO service ERROR happened`. Here level=INFO, message="ERROR happened". first_error should not be set because the level is INFO, not ERROR. My code checks `level == "ERROR"`, so it correctly only sets first_error when the actual level field is ERROR. Good.

Let me write the final version.

Actually, let me reconsider the type hints more carefully. The return dict:
- "total": int
- "malformed": int
- "by_level": Dict[str, int]
- "services": List[str]
- "first_error": Optional[str]

I'll use `Dict[str, Any]` for the return since it's a heterogeneous dict. That's the cleanest.

Let me also add a docstring.

Final code:

```python
from datetime import datetime
from typing import Any, Dict, List


def summarize_logs(logs: List[str]) -> Dict[str, Any]:
    """Summarize a list of log lines.

    Each well-formed line looks like:
        2026-04-11T08:15:30Z LEVEL service_name message text here

    LEVEL must be one of DEBUG, INFO, WARN, ERROR.

    Returns a dict with keys: total, malformed, by_level, services,
    first_error.
    """
    total = 0
    malformed = 0
    by_level: Dict[str, int] = {}
    services: set[str] = set()
    first_error: str | None = None

    valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"}

    for line in logs:
        parts = line.split()
        if len(parts) < 4:
            malformed += 1
            continue

        timestamp_str, level, service = parts[0], parts[1], parts[2]
        message = " ".join(parts[3:])

        if level not in valid_levels:
            malformed += 1
            continue

        try:
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            malformed += 1
            continue

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

Wait, `set[str]` and `str | None` syntax requires Python 3.9+ and 3.10+ respectively. To be safe and compatible, I could use `typing.Set` and `Optional[str]`. But modern Python (3.9+) supports `set[str]`. Let me use `typing` imports to be safe and broadly compatible.

Let me use:
```python
from typing import Any, Dict, List, Optional, Set
```

And:
```python
services: Set[str] = set()
first_error: Optional[str] = None
```

This is more compatible. Let me finalize with this.

Actually, `set[str]` works in 3.9+. But to be maximally compatible, I'll use the typing module versions. Let me go with that.

Let me reconsider the timestamp validation edge case once more. What if someone passes a timestamp like `2026-04-11T08:15:30` (no Z, no timezone)? fromisoformat parses it as naive datetime. That's fine, it's still a valid-ish timestamp. Should it be considered well-formed? The format shows a Z suffix, but I think being lenient here is fine. The main point is to reject clearly invalid timestamps.

Hmm, but actually, maybe I'm overcomplicating. Let me reconsider whether timestamp validation is even needed. The problem says "well-formed log line". The key structural elements are timestamp, level, service, message. The most important validation is the level. But a malformed timestamp should probably count as malformed too.

I'll keep the timestamp validation but make it reasonable. Actually, let me reconsider — maybe I should validate that the timestamp matches the expected ISO format more strictly. But fromisoformat is good enough and simple.

Let me reconsider: is there a risk that fromisoformat is too lenient and accepts garbage? For example, `datetime.fromisoformat("2026")` — does that work? Let me think. In Python 3.7+, `fromisoformat("2026")` raises ValueError because it expects at least YYYY-MM-DD. Actually, let me verify: `datetime.fromisoformat("2026")` → ValueError: Invalid isoformat string. Yes, it raises. Good.

What about `datetime.fromisoformat("2026-04-11")`? That parses as a date at midnight. It's a valid ISO date. Hmm, so a date-only timestamp would pass. But that's probably acceptable for this problem.

OK here's another thought. What if the timestamp is `2026-04-11T08:15:30Zextra`? After replace: `2026-04-11T08:15:30+00:00extra`. fromisoformat would raise ValueError. Good, caught as malformed.

I'm confident in the solution. Let me finalize.

Actually, let me reconsider one more time about whether to require the timestamp to end with Z or have a timezone. The format explicitly shows `Z`. But I don't want to be overly strict and reject valid variations. I'll keep it lenient with fromisoformat.

Let me also double-check: the problem says "Do not use regular expressions with catastrophic backtracking; keep it simple". I'm not using regex at all, so this is satisfied. Good.

Let me write the final answer.

One more consideration: should I handle the case where `line` is not a string? The type hint says List[str], so I'll assume strings. If a non-string is passed, `.split()` would fail. But that's a type violation, not something I need to handle per the spec. I'll leave it.

Final answer:

```python
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


def summarize_logs(logs: List[str]) -> Dict[str, Any]:
    """Summarize a list of log lines.

    Each well-formed line looks like:
        2026-04-11T08:15:30Z LEVEL service_name message text here

    where LEVEL is one of DEBUG, INFO, WARN, ERROR.

    Returns a dict with exactly these keys:
        "total"      - number of lines successfully parsed
        "malformed"  - number of lines that could not be parsed
        "by_level"   - dict mapping each level that appeared to its count
        "services"   - sorted list of distinct service names seen
        "first_error"- message text of the first ERROR line, or None
    """
    total = 0
    malformed = 0
    by_level: Dict[str, int] = {}
    services: Set[str] = set()
    first_error: Optional[str] = None

    valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"}

    for line in logs:
        parts = line.split()
        # Need at least: timestamp, level, service, and one message word.
        if len(parts) < 4:
            malformed += 1
            continue

        timestamp_str, level, service = parts[0], parts[1], parts[2]
        message = " ".join(parts[3:])

        if level not in valid_levels:
            malformed += 1
            continue

        # Basic ISO-8601 timestamp validation (accept trailing 'Z' as UTC).
        try:
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            malformed += 1
            continue

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

This looks complete and correct. Let me trace through an example to verify.

Example logs:
```
[
  "2026-04-11T08:15:30Z INFO auth User logged in",
  "2026-04-11T08:15:31Z ERROR db Connection failed",
  "2026-04-11T08:15:32Z WARN cache Low memory",
  "2026-04-11T08:15:33Z INFO auth Another login",
  "this is malformed",
  "2026-04-11T08:15:34Z BADLEVEL svc msg",
  "notatimestamp INFO svc msg here",
]
```

Processing:
1. Line 1: parts = [timestamp, INFO, auth, User, logged, in], len 6 >= 4. level=INFO valid. timestamp valid. total=1, by_level={INFO:1}, services={auth}, first_error=None.
2. Line 2: parts = [..., ERROR, db, Connection, failed], len 6. level=ERROR valid. timestamp valid. total=2, by_level={INFO:1, ERROR:1}, services={auth, db}, first_error="Connection failed".
3. Line 3: parts = [..., WARN, cache, Low, memory], len 5. level=WARN valid. total=3, by_level={INFO:1, ERROR:1, WARN:1}, services={auth, db, cache}.
4. Line 4: parts = [..., INFO, auth, Another, login], len 5. level=INFO valid. total=4, by_level={INFO:2, ERROR:1, WARN:1}, services={auth, db, cache}.
5. Line 5: "this is malformed" → parts = [this, is, malformed], len 3 < 4. malformed=1.
6. Line 6: parts = [..., BADLEVEL, svc, msg], len 4. level=BADLEVEL not valid. malformed=2.
7. Line 7: parts = [notatimestamp, INFO, svc, msg, here], len 5. level=INFO valid. timestamp "notatimestamp".replace("Z","+00:00") = "notatimestamp". fromisoformat raises ValueError. malformed=3.

Result:
- total: 4
- malformed: 3
- by_level: {INFO: 2, ERROR: 1, WARN: 1}
- services: ['auth', 'cache', 'db'] (sorted)
- first_error: "Connection failed"

This looks correct!

The solution is complete. Let me present it.
</think>

</details>

**Answer (as scored):**

```python
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


def summarize_logs(logs: List[str]) -> Dict[str, Any]:
    """Summarize a list of log lines.

    Each well-formed line looks like:
        2026-04-11T08:15:30Z LEVEL service_name message text here

    where LEVEL is one of DEBUG, INFO, WARN, ERROR.

    Returns a dict with exactly these keys:
        "total"      - number of lines successfully parsed
        "malformed"  - number of lines that could not be parsed
        "by_level"   - dict mapping each level that appeared to its count
        "services"   - sorted list of distinct service names seen
        "first_error"- message text of the first ERROR line, or None
    """
    total = 0
    malformed = 0
    by_level: Dict[str, int] = {}
    services: Set[str] = set()
    first_error: Optional[str] = None

    valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"}

    for line in logs:
        parts = line.split()
        # Need at least: timestamp, level, service, and one message word.
        if len(parts) < 4:
            malformed += 1
            continue

        timestamp_str, level, service = parts[0], parts[1], parts[2]
        message = " ".join(parts[3:])

        if level not in valid_levels:
            malformed += 1
            continue

        # Basic ISO-8601 timestamp validation (accept trailing 'Z' as UTC).
        try:
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            malformed += 1
            continue

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

