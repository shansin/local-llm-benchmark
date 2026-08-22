from llmbench.scoring.aggregate import completeness
from llmbench.scoring.extract import (
    extract_answer,
    extraction_for,
    looks_like_leaked_reasoning,
    no_answer_reason,
    with_answer_format,
)
from llmbench.scoring.objective import objective_score, run_checks
from llmbench.tasks import Task

# ---------- reasoning tags ----------


def test_closed_reasoning_tag_is_removed_from_the_answer():
    e = extract_answer("<think>I should count the words</think>The final text.")
    assert e.answer == "The final text."
    assert e.reasoning == "I should count the words"
    assert e.source == "reasoning_tag"
    assert e.complete


def test_alternative_reasoning_tags_are_recognised():
    for tag in ("thinking", "reasoning", "thought", "scratchpad", "analysis"):
        e = extract_answer(f"<{tag}>plan</{tag}>answer")
        assert e.answer == "answer", tag


def test_unclosed_reasoning_tag_means_there_is_no_answer():
    """A generation cut off mid-thought has not answered, however long it is."""
    e = extract_answer("<think>" + "still working " * 500)
    assert e.answer == ""
    assert e.empty
    assert not e.complete
    assert e.source == "reasoning_tag_unclosed"


# ---------- answer tags ----------


def test_answer_tag_wins_over_surrounding_prose():
    e = extract_answer("Let me plan this out.\n<answer>The product description.</answer>")
    assert e.answer == "The product description."
    assert "Let me plan" in e.reasoning
    assert e.source == "answer_tag"


def test_last_answer_block_is_the_answer():
    e = extract_answer("<answer>first try</answer> hmm <answer>better</answer>")
    assert e.answer == "better"


def test_unclosed_answer_tag_is_incomplete_but_still_scored():
    e = extract_answer("thinking\n<answer>half an ans")
    assert e.answer == "half an ans"
    assert not e.complete


def test_with_answer_format_appends_the_instruction():
    assert "<answer>" in with_answer_format("Write a poem.")
    assert with_answer_format("Write a poem.").startswith("Write a poem.")


# ---------- the thinking channel ----------


def test_thinking_field_is_kept_apart_from_the_answer():
    e = extract_answer("The answer.", thinking="a long deliberation")
    assert e.answer == "The answer."
    assert e.reasoning == "a long deliberation"
    assert e.source == "thinking_field"


def test_extraction_for_reads_records_written_before_thinking_was_captured():
    """Old checkpoints have no `thinking` key at all and must still re-score."""
    e = extraction_for({"response": "hello"})
    assert e.answer == "hello"
    assert e.source == "verbatim"


# ---------- leaked reasoning ----------


def test_planning_prose_is_recognised_as_leaked_reasoning():
    for opener in (
        "Let me craft a scene that meets all these constraints:",
        "Okay, let me think about what the user wants here.",
        "I need to write exactly six sentences without the letter e.",
        "The user wants a product description.",
        "First, I'll work out the word count.",
    ):
        e = extract_answer(f"{opener}\n\nHere is the thing.")
        assert looks_like_leaked_reasoning(e), opener


def test_a_direct_answer_is_not_flagged_as_leaked_reasoning():
    for answer in (
        "The sky looks blue because sunlight is made of many colours.",
        "Ava is a knight, Ben is a knave, and Cleo is a knight.",
        '{"total": 3, "malformed": 0}',
        "Let it be known that this mug keeps drinks hot.",  # 'let' but not 'let me'
        # Addressed to the reader on a task that asked for reasoning shown.
        # An earlier, looser pattern flagged every correct solution to the
        # knights-and-knaves puzzle as leaked reasoning.
        "Let's analyze this step-by-step: knights always tell the truth.",
        "Let's translate each statement into a logical condition.",
        "We need three weighings to identify the counterfeit ball.",
    ):
        assert not looks_like_leaked_reasoning(extract_answer(answer)), answer


def test_delimited_reasoning_is_never_counted_as_leaked():
    """Once a model fences its thinking the split is exact; nothing leaked."""
    e = extract_answer("<think>Let me plan this</think>Let me tell you a story.")
    assert not looks_like_leaked_reasoning(e)


# ---------- empty answers do not pass negated checks ----------


def _task(*checks):
    return Task(id="t", category="writing", prompt="p", checks=list(checks))


def test_an_empty_answer_fails_every_check_including_negated_ones():
    """Silence satisfies "must not contain X" only if nobody is paying attention."""
    task = _task(
        {"type": "regex", "pattern": "(?i)wolf", "negate": True},
        {"type": "word_count", "min": 220},
    )
    results = run_checks(task, "")
    assert [r["passed"] for r in results] == [0.0, 0.0]
    assert objective_score(results) == 0.0
    assert all("no answer" in r["detail"] for r in results)


def test_a_real_answer_still_passes_a_negated_check():
    task = _task({"type": "regex", "pattern": "(?i)wolf", "negate": True})
    assert run_checks(task, "A cunning beast met a young traveller.")[0]["passed"] == 1.0


# ---------- explaining an absent answer ----------


def test_truncation_is_named_as_the_cause_when_it_is_the_cause():
    reason = no_answer_reason({"truncated": True}, extract_answer("<think>" + "x" * 100))
    assert "context limit" in reason
    assert "NUM_CTX" in reason


def test_a_genuinely_silent_model_is_not_blamed_on_the_context_window():
    reason = no_answer_reason({"truncated": False}, extract_answer(""))
    assert "context limit" not in reason


# ---------- completeness accounting ----------


def test_completeness_separates_truncation_from_leakage_from_errors():
    results = {
        "a": [
            {"response": "a clean answer"},
            {"response": "<think>never finished", "truncated": True},
        ],
        "b": [
            {"response": "Let me work through this.\n\nThe answer."},
            {"response": "[ERROR: X]", "error": "X"},
        ],
    }
    stats = completeness(results)
    assert stats["total"] == 4
    assert stats["truncated"] == 1
    assert stats["empty"] == 1
    assert stats["leaked_reasoning"] == 1
    assert stats["errors"] == 1
