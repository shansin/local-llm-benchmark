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

- Score: 10.0/10 — Implementation uses a stack, ignores non-brackets, handles empty string, includes type hints, and passes all 9 tests.
- Tokens/s: 99.4
- Prefill tok/s: 922.9
- TTFT: 5.47s
- Gen Time: 7.39s
- Output Tokens: 674
- answer read from `answer_tag`

<details><summary>Reasoning (1672 chars, not scored)</summary>

The user wants a Python function called `is_balanced` that checks if brackets in a string are properly balanced and nested.

Let me think through the approach:

1. Use a stack-based approach
2. Iterate through each character in the string
3. If it's an opening bracket (`(`, `[`, `{`), push it onto the stack
4. If it's a closing bracket (`)`, `]`, `}`), check if the stack is non-empty and the top of the stack matches the corresponding opening bracket
5. If it's any other character, ignore it
6. At the end, the stack should be empty for the string to be balanced

Edge cases:
- Empty string: stack is empty at the end, so return True
- Closing bracket when stack is empty: return False
- Mismatched brackets: return False
- Unclosed brackets at end: stack not empty, return False

Let me write this with type hints:

```python
def is_balanced(s: str) -> bool:
    stack: list[str] = []
    matching = {')': '(', ']': '[', '}': '{'}
    opening = {'(', '[', '{'}
    
    for char in s:
        if char in opening:
            stack.append(char)
        elif char in matching:
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
        # else: ignore other characters
    
    return len(stack) == 0
```

Let me verify with the examples:
- `"(a[b]{c})"`: push `(`, push `[`, pop `[` (matches `]`), push `{`, pop `{` (matches `}`), pop `(` (matches `)`) → stack empty → True ✓
- `"([)]"`: push `(`, push `[`, then `)` comes, top is `[` which doesn't match `(` → False ✓
- `"("`: push `(`, end of string, stack not empty → False ✓
- `""`: no iterations, stack empty → True ✓

This looks correct. Let me finalize.

</details>

**Answer (as scored):**

```python
def is_balanced(s: str) -> bool:
    """
    Check if all brackets in the string are correctly balanced and nested.
    
    Supports three bracket types: (), [], {}
    All other characters are ignored.
    
    Args:
        s: The input string to check.
        
    Returns:
        True if all brackets are balanced and properly nested, False otherwise.
    """
    stack: list[str] = []
    matching_close: dict[str, str] = {')': '(', ']': '[', '}': '{'}
    opening: set[str] = {'(', '[', '{'}

    for char in s:
        if char in opening:
            stack.append(char)
        elif char in matching_close:
            if not stack or stack[-1] != matching_close[char]:
                return False
            stack.pop()
        # All other characters are ignored

    return len(stack) == 0
```

