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

- Score: 10.0/10 — Function correctly implements stack-based bracket matching with type hints, ignores non-brackets, handles empty string, and passes all automated tests.
- Tokens/s: 304.7
- Prefill tok/s: 1964.8
- TTFT: 1.04s
- Gen Time: 1.33s
- Output Tokens: 356
- answer read from `answer_tag`

<details><summary>Reasoning (1061 chars, not scored)</summary>

We need to write a Python function `is_balanced` that takes a string and returns True if all brackets in it are correctly balanced and nested, False otherwise. Consider three bracket types: (), [], {}. Any other characters should be ignored. Include type hints. Handle empty string. Brackets must be closed in correct order: `([)]` is not balanced.

Examples: is_balanced("(a[b]{c})") => True, is_balanced("([)]") => False, is_balanced("(") => False.

We need to output the final answer between <answer> and </answer> tags, with no commentary.

We need to write the function. Implementation: Use a stack. Iterate over characters. If char is one of opening brackets '(', '[', '{', push onto stack. If char is a closing bracket ')', ']', '}', check if stack is empty or top doesn't match; if so return False. Else pop. After iteration, return True if stack empty else False.

Add type hints: def is_balanced(s: str) -> bool:.

Edge cases: empty string returns True (since no brackets). The function should ignore non-bracket characters.

Let's produce the answer.

</details>

**Answer (as scored):**

def is_balanced(s: str) -> bool:
    stack = []
    opening = {'(', '[', '{'}
    closing = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                return False
            stack.pop()
    return not stack

