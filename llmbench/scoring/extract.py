"""Separating a model's reasoning from the answer it is being scored on.

Reasoning models emit two different things and the benchmark cares about only
one of them. Where that separation lands depends on the model:

* Ollama reports a chain of thought in a distinct `thinking` field, leaving
  `response` clean. Nothing to do — except that the runner must actually
  capture `thinking`, or a model that puts everything there looks like it
  answered with silence.
* Some models delimit their own reasoning inline with `<think>` tags.
* Some models emit reasoning as ordinary prose with no delimiter at all, then
  the answer underneath it.

The last case is the dangerous one, because nothing about it *looks* like a
failure. The checks measure the planning notes instead of the answer: a
250-word scene is counted as 4311 words, and a "never use the word wolf"
constraint fails on the sentence where the model reminded itself not to use the
word wolf. The judge, reading further, scores the actual answer 9/10. The run
then reports a model 1.5 points below its peers with no visible cause.

Extraction is deliberately conservative: it splits only on evidence that is
actually present in the text. When there is no such evidence the whole response
is treated as the answer, which is the honest reading — a model that buries its
answer in undelimited prose has, from the caller's point of view, emitted
undelimited prose. `looks_like_leaked_reasoning` measures how often that
happens so it appears in the report as a named finding rather than as an
unexplained score gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tags that models use to fence their own chain of thought.
REASONING_TAGS = ("think", "thinking", "reasoning", "thought", "scratchpad", "analysis")

# The tag the benchmark can ask for explicitly (ANSWER_TAGS=1).
ANSWER_TAG = "answer"

ANSWER_FORMAT_INSTRUCTION = (
    "Think first if you need to, then put your complete final answer between "
    "<answer> and </answer> tags. Everything inside those tags must be the answer "
    "itself, with no commentary about how you produced it."
)

_TAG_ALTERNATION = "|".join(REASONING_TAGS)
_REASONING_BLOCK = re.compile(
    rf"<\s*({_TAG_ALTERNATION})\s*>(.*?)<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL
)
_REASONING_OPEN = re.compile(rf"<\s*({_TAG_ALTERNATION})\s*>", re.IGNORECASE)
_ANSWER_BLOCK = re.compile(
    rf"<\s*{ANSWER_TAG}\s*>(.*?)<\s*/\s*{ANSWER_TAG}\s*>", re.IGNORECASE | re.DOTALL
)
_ANSWER_OPEN = re.compile(rf"<\s*{ANSWER_TAG}\s*>", re.IGNORECASE)

# Self-addressed planning language, at the very start of a response. Models that
# leak their reasoning open this way with striking consistency; models that
# answer directly essentially never do.
#
# The distinction that matters is who is being addressed. "Let me think about
# what the user wants" is the model talking to itself and is leakage. "Let's
# analyse this step by step" is the model talking to the reader, and on a task
# that asked for reasoning shown step by step it is the answer — an earlier,
# looser version of this pattern flagged every correct knights-and-knaves
# solution in the run.
_PLANNING_OPENER = re.compile(
    r"^\s*(?:okay|ok|alright|right|hmm|so)?[,.]?\s*"
    r"(?:let me\b"
    r"|i(?:'ll| will| need to| should| am going to| have to| want to)\b"
    r"|first,? i\b"
    r"|the (?:user|prompt) (?:wants|is asking|asked|has asked|requires)\b"
    r"|this (?:task|prompt|request) (?:asks|wants|requires)\b"
    r"|thinking (?:about|through)\b"
    r"|i'm going to\b"
    r")",
    re.IGNORECASE,
)

# How far into the response the planning opener has to appear to count.
_OPENER_WINDOW = 240


@dataclass(frozen=True)
class Extraction:
    """What a response actually offered as its answer.

    `source` records which piece of evidence the split rested on, so a
    surprising score can be traced back to how the text was read.
    `complete` is False when the text ends inside a block that never closed —
    the sign of a generation cut off by the context window rather than by the
    model deciding it was finished.
    """

    answer: str
    reasoning: str
    source: str
    complete: bool = True

    @property
    def empty(self) -> bool:
        return not self.answer.strip()


def extract_answer(response: str, thinking: str = "") -> Extraction:
    """Split a response into (reasoning, answer).

    Evidence is considered in order of how explicit it is: an answer tag we
    asked for, then a reasoning tag the model chose, then Ollama's separate
    thinking channel, then nothing.
    """
    text = response or ""

    closed = _ANSWER_BLOCK.findall(text)
    if closed:
        # Last block wins: a model that restates its answer means the final one.
        answer = closed[-1]
        reasoning = _ANSWER_BLOCK.sub("", text)
        return Extraction(answer.strip(), _join(thinking, reasoning), "answer_tag")

    opened = _ANSWER_OPEN.search(text)
    if opened:
        # An opening tag with no close: the answer began and was cut off.
        return Extraction(
            text[opened.end() :].strip(),
            _join(thinking, text[: opened.start()]),
            "answer_tag_unclosed",
            complete=False,
        )

    blocks = _REASONING_BLOCK.findall(text)
    if blocks:
        answer = _REASONING_BLOCK.sub("", text)
        reasoning = "\n".join(body for _, body in blocks)
        return Extraction(answer.strip(), _join(thinking, reasoning), "reasoning_tag")

    unclosed = _REASONING_OPEN.search(text)
    if unclosed:
        # Reasoning started and never ended: there is no answer, and saying so
        # is more accurate than scoring the chain of thought as one.
        return Extraction(
            text[: unclosed.start()].strip(),
            _join(thinking, text[unclosed.end() :]),
            "reasoning_tag_unclosed",
            complete=False,
        )

    if thinking.strip():
        return Extraction(text.strip(), thinking.strip(), "thinking_field")

    return Extraction(text.strip(), "", "verbatim")


def _join(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p and p.strip())


def looks_like_leaked_reasoning(extraction: Extraction) -> bool:
    """Does this answer appear to still have undelimited reasoning in front of it?

    Only meaningful when no tag was found — once a model delimits its thinking,
    the split is already exact. The signal is first-person planning language in
    the opening lines, which is what the answer to a prompt asking for a product
    description or a dialogue scene never begins with.
    """
    if extraction.source not in ("verbatim", "thinking_field"):
        return False
    return bool(_PLANNING_OPENER.match(extraction.answer[:_OPENER_WINDOW]))


def with_answer_format(prompt: str) -> str:
    """Append the explicit answer-delimiter instruction to a task prompt."""
    return f"{prompt.rstrip()}\n\n{ANSWER_FORMAT_INSTRUCTION}"


def extraction_for(result: dict[str, object]) -> Extraction:
    """Read the answer out of a stored generation record.

    Records written before the runner captured the thinking channel simply have
    no `thinking` key, so old checkpoints and old runs re-score without special
    handling.
    """
    return extract_answer(str(result.get("response") or ""), str(result.get("thinking") or ""))


def no_answer_reason(result: dict[str, object], extraction: Extraction) -> str:
    """Why there is nothing here to score — the distinction that matters.

    A generation cut off at the context limit and a model that genuinely
    answered with silence both leave an empty string, but only one of them is a
    statement about the model.
    """
    if result.get("truncated") or not extraction.complete:
        thought = len(extraction.reasoning)
        return (
            "[NO ANSWER] generation hit the context limit while still reasoning "
            f"({thought} characters of it); raise NUM_CTX to score this task"
        )
    if result.get("error"):
        return f"[NO ANSWER] generation failed: {result['error']}"
    return "[NO ANSWER] the model returned nothing outside its reasoning"
