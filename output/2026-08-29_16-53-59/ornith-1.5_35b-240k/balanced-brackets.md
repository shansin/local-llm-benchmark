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

- Score: 9.75/10 — Function is correctly named with type hints, uses stack for proper nesting, ignores non-brackets, handles empty string and mismatches, and code is clear and runnable with all tests passing.
- Tokens/s: 207.6
- Prefill tok/s: 2139.2
- TTFT: 0.19s
- Gen Time: 3.75s
- Output Tokens: 739
- answer read from `answer_tag`

<details><summary>Reasoning (2012 chars, not scored)</summary>

The user wants a Python function `is_balanced` that checks if brackets are balanced and nested. Let me write it.

The standard approach uses a stack. For each character:
- If it's an opening bracket, push it onto the stack.
- If it's a closing bracket, check if the stack is non-empty and the top matches the corresponding opening bracket. If not, return False. Pop the top.
- Ignore other characters.

At the end, return True if the stack is empty.

Let me write it with type hints.

```python
def is_balanced(s: str) -> bool:
    stack: list[str] = []
    matching = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
    return not stack
```

This handles the empty string (returns True since stack is empty).

Let me verify with examples:
- `"(a[b]{c})"`: 
  - '(' push -> ['(']
  - 'a' ignore
  - '[' push -> ['(', '[']
  - 'b' ignore
  - ']' matches '[' -> pop -> ['(']
  - '{' push -> ['(', '{']
  - 'c' ignore
  - '}' matches '{' -> pop -> ['(']
  - ')' matches '(' -> pop -> []
  - return True ✓
- `"([)]"`:
  - '(' push -> ['(']
  - '[' push -> ['(', '[']
  - ')' -> top is '[', doesn't match '(' -> return False ✓
- `"("`:
  - '(' push -> ['(']
  - return not ['('] = False ✓

Good. Let me finalize.
</think>

```python
def is_balanced(s: str) -> bool:
    """Return True if all brackets in `s` are balanced and correctly nested.

    Only the bracket characters '(', ')', '[', ']', '{', '}' are considered;
    every other character is ignored. The empty string is considered balanced.
    """
    stack: list[str] = []
    matching = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()

    return not stack
```

</details>

**Answer (as scored):**

def is_balanced(s: str) -> bool:
    """Return True if all brackets in `s` are balanced and correctly nested.

    Only the bracket characters '(', ')', '[', ']', '{', '}' are considered;
    every other character is ignored. The empty string is considered balanced.
    """
    stack: list[str] = []
    matching = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()

    return not stack

