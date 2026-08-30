# coding / balanced-brackets

**Prompt:** Write a Python function called `is_balanced` that takes a string and returns True if all brackets in it are correctly balanced and nested, and False otherwise.

Consider three bracket types: `()`, `[]`, and `{}`. Any other characters in the string should be ignored.

Requirements:
- Include type hints
- Handle the empty string
- Brackets must be closed in the correct order: `([)]` is not balanced

Examples:
  is_balanced("(a[b]{c})") => True
  is_balanced("([)]")      => False
  is_balanced("(")         => False

## Repeat 1 (seed 0)

- Score: 10.0/10 — Function is correctly named with type hints, uses stack to enforce nesting order, ignores non-brackets, handles empty string and mismatches, and all 9 tests passed.
- Tokens/s: 56.5
- Prefill tok/s: 1182.6
- TTFT: 0.53s
- Gen Time: 24.80s
- Output Tokens: 1371
- answer read from `answer_tag`

<details><summary>Reasoning (3810 chars, not scored)</summary>

We need answer user's request: write Python function is_balanced with type hints, handle empty string, ignore other chars, correct nesting. Need think first, then final answer between  tags. Need likely provide code only? They say complete final answer between tags. We can include code block. Need ensure no commentary outside? The final answer should be inside tags. Could include explanation? Probably just code. Need think through implementation.

Function:
def is_balanced(s: str) -> bool:
    stack: list[str] = []
    matching = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != matching[ch]:
                return False
            stack.pop()
    return not stack

Type hints: list[str] requires Python 3.9+. Could use from typing import List for compatibility? Requirements include type hints. Use list[str] okay. But if older? Could use typing.List. Simpler:
from typing import List
def is_balanced(s: str) -> bool:
    stack: List[str] = []
...
Need maybe no imports? If include typing import. Could avoid by using list[str]. But type hints for local variable optional. Use built-in generics.

Edge cases: empty string -> stack empty -> True. Any other chars ignored. Correct order.

Need final:


Need ensure final only tags? They said put complete final answer between  tags. Everything inside tags must be answer itself, no commentary. So final should be exactly tags with code. Could include no extra text. Good.

Double-check: If string has unmatched closer when stack empty returns False. If leftover openers returns False. Ignores other chars. Type hints. Empty string returns True.

Potential issue: list[str] type hint in local variable may be okay. If environment Python <3.9, list not subscriptable? But modern. Could use from typing import List to be safe. But then need import. The user asked write function, not necessarily import. Use typing? Let's decide. Many coding tasks expect Python 3.10+. list[str] fine. But if they run in Python 3.8, list[str] fails at runtime? Actually PEP 585 built-in generics available in 3.9. If 3.8, TypeError: 'type' object is not subscriptable? For variable annotation, annotations are evaluated at runtime unless from __future__ annotations? In Python 3.8, list[str] raises TypeError? Let's recall: In Python 3.8, list is not subscriptable? Yes built-in collection generics introduced 3.9. So to maximize compatibility, use typing.List. But type hints requirement. Could do:
from typing import List

def is_balanced(s: str) -> bool:
    stack: List[str] = []
...
This works in 3.8+. But if they only want function, import okay. Or avoid local annotation: stack = [] and no type hint for stack. But requirement include type hints likely function signature enough. Could do:
def is_balanced(s: str) -> bool:
    stack = []
...
This has type hints. But maybe they expect variable type hints? Not necessary. To be safe, include List import. But final answer with import may be considered complete. Use:
from typing import List

def is_balanced(s: str) -> bool:
    stack: List[str] = []
    ...

Need maybe use dict[str, str] for matching? Could use typing.Dict. But not necessary. If using List, matching dict no annotation. Or use dict[str, str] if 3.9. Keep simple.

Could also use collections.deque? No.

Need ensure no commentary. Final:


Check examples:
is_balanced("(a[b]{c})") -> stack: (, [, ] pop, {, } pop, ) pop -> True.
"([)]" -> (, [, ) closer expects [ but top [ != ( -> False.
"(" -> leftover -> False.
Empty -> True.

Could there be issue with Unicode brackets? Only three types. Good.

Need maybe if input not string? Type hints but no validation. Fine.

Now final. Ensure no analysis outside. final only.
</think>

</details>

**Answer (as scored):**

from typing import List

def is_balanced(s: str) -> bool:
    """Return True if all brackets in s are correctly balanced and nested."""
    stack: List[str] = []
    openers = "([{"
    closers = ")]}"
    matching = {")": "(", "]": "[", "}": "{"}

    for ch in s:
        if ch in openers:
            stack.append(ch)
        elif ch in closers:
            if not stack or stack[-1] != matching[ch]:
                return False
            stack.pop()

    return not stack

