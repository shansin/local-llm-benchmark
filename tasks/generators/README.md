# Task generators

Some tasks carry an answer key that cannot be checked by eye: a 180-entry
ledger whose totals have to be summed, a state machine that has to be run.
Deriving those by hand is how a benchmark ends up grading a correct answer as
wrong, so the task file and its `[[checks]]` are emitted together from the same
code that computes the ground truth.

Regenerate after editing a generator:

    uv run python tasks/generators/build.py

The generated `.toml` files are committed — the benchmark must be runnable
without executing the generators, and a diff in a task file should be visible
in review.
