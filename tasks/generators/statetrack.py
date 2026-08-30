"""State-tracking tasks: carry a value correctly through many small steps.

Nothing here is clever. Every step is trivial in isolation and the whole thing
is mechanical — which is the point. There is no memorised answer to recall and
no insight to have, so the score measures only whether a model can hold a
changing value across twenty-odd updates without drifting. In the 25-35B band
that is where the spread is, and difficulty scales smoothly by adding steps
rather than by getting cleverer.

Every answer key is produced by running the same simulation the prompt
describes, so the key cannot disagree with the task.
"""

from __future__ import annotations

import random

from common import json_checks, write_task


def ledger_balance() -> None:
    """An account with one rule that fires partway through and changes later steps."""
    rng = random.Random(31337)
    ops = []
    balance = 0
    fee_count = 0
    for _ in range(26):
        kind = rng.choice(["deposit", "withdraw", "withdraw"])
        amount = rng.randrange(20, 400, 5)
        ops.append((kind, amount))
        balance += amount if kind == "deposit" else -amount
        if balance < 0:
            balance -= 25
            fee_count += 1

    listing = "\n".join(f"{index + 1}. {kind} {amount}" for index, (kind, amount) in enumerate(ops))
    prompt = f"""
An account starts with a balance of 0.

Apply the following {len(ops)} operations in order. After each operation, if the balance
is below zero, an overdraft fee of 25 is charged immediately — that fee itself can leave
the balance further below zero, but a fee is never charged on a fee: check the balance
once per numbered operation, after applying it.

{listing}

Give the final balance. End your response with a line of exactly this form and nothing
after it:

Answer: <number>
"""

    criteria = f"""
Evaluation criteria:
- The only thing that matters is the final balance. The task is mechanical: 26 additions
  and subtractions with one conditional. There is no insight to have and no partial credit
  for a good method that lands on the wrong number.
- Common failure is drift — a model that loses track around step 15 produces a number in
  the right range and is simply wrong. Do not reward plausibility.
- The overdraft fee is charged once per numbered operation, not recursively.
- The response must end with the `Answer: <number>` line.

The correct final balance is {balance}, with {fee_count} overdraft fees charged.
"""

    write_task(
        "statetrack",
        "ledger-balance",
        difficulty="medium",
        prompt=prompt,
        criteria=criteria,
        checks=[{"type": "answer_equals", "expected": balance, "numeric": True}],
    )


def stack_machine() -> None:
    """A five-instruction VM, run for thirty instructions."""
    rng = random.Random(2718)
    stack: list[int] = []
    program: list[str] = []
    while len(program) < 30:
        choices = ["PUSH"]
        if len(stack) >= 1:
            choices += ["DUP", "POP"]
        if len(stack) >= 2:
            choices += ["ADD", "SWAP", "SUB"]
        op = rng.choice(choices)
        if op == "PUSH":
            value = rng.randint(1, 20)
            stack.append(value)
            program.append(f"PUSH {value}")
        elif op == "POP":
            stack.pop()
            program.append("POP")
        elif op == "DUP":
            stack.append(stack[-1])
            program.append("DUP")
        elif op == "SWAP":
            stack[-1], stack[-2] = stack[-2], stack[-1]
            program.append("SWAP")
        elif op == "ADD":
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
            program.append("ADD")
        else:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
            program.append("SUB")

    listing = "\n".join(f"{i + 1:2d}  {line}" for i, line in enumerate(program))
    prompt = f"""
A stack machine starts with an empty stack and supports five instructions:

- `PUSH n` — push the integer n on top of the stack
- `POP` — remove the top value and discard it
- `DUP` — push a copy of the top value
- `SWAP` — exchange the top two values
- `ADD` — pop two values and push their sum
- `SUB` — pop two values, and push (the value that was second from top) minus (the value
  that was on top)

Run this program:

{listing}

Report the final state of the stack. Return only a JSON object, no code fence, no
commentary:

{{"depth": <number>, "top": <number>, "sum": <number>}}

where `depth` is how many values remain on the stack, `top` is the value on top, and `sum`
is the sum of every value remaining on the stack.
"""

    criteria = f"""
Evaluation criteria:
- Three exact integers, obtained by executing 30 instructions in order. Each instruction is
  trivial; the difficulty is entirely in not losing the stack partway through.
- `SUB` order is the usual one and is spelled out in the prompt — second-from-top minus top.
  Getting it backwards is a real error, not an ambiguity.
- The three fields are checked separately, so a model that tracks depth correctly but drops
  a value scores partly rather than not at all.
- Bare JSON, no fence, no working shown.

The correct answers are: depth {len(stack)}, top {stack[-1]}, sum {sum(stack)}.
"""

    write_task(
        "statetrack",
        "stack-machine",
        difficulty="hard",
        prompt=prompt,
        criteria=criteria,
        checks=json_checks({"depth": len(stack), "top": stack[-1], "sum": sum(stack)}),
    )


def robot_grid() -> None:
    """Position and heading after a long move sequence on a wrapping grid."""
    rng = random.Random(1123)
    width = height = 8
    x, y, heading = 0, 0, 0  # heading: 0=north, 1=east, 2=south, 3=west
    deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    moves = []
    visited_origin = 0
    for _ in range(40):
        move = rng.choice(["F", "F", "L", "R"])
        moves.append(move)
        if move == "L":
            heading = (heading - 1) % 4
        elif move == "R":
            heading = (heading + 1) % 4
        else:
            dx, dy = deltas[heading]
            x, y = (x + dx) % width, (y + dy) % height
            if (x, y) == (0, 0):
                visited_origin += 1

    sequence = " ".join(moves)
    headings = ["north", "east", "south", "west"]
    prompt = f"""
A robot stands on an {width}x{height} grid at cell (0, 0), facing north. Cells are
addressed (x, y): x increases to the east, y increases to the north. The grid wraps —
moving north from y={height - 1} arrives at y=0, and moving west from x=0 arrives at
x={width - 1}.

The robot obeys three commands:

- `F` — move one cell forward in the direction it is facing
- `L` — turn 90 degrees left, without moving
- `R` — turn 90 degrees right, without moving

Execute this sequence:

{sequence}

Return only a JSON object, no code fence, no commentary:

{{"x": <number>, "y": <number>, "facing": "<north|east|south|west>", "returned_to_origin": <number>}}

where `returned_to_origin` counts how many times the robot arrived back at cell (0, 0)
during the sequence — not counting the start.
"""

    criteria = f"""
Evaluation criteria:
- Four exact values from executing 40 commands. Turns do not move the robot and moves do
  not turn it; wrapping applies in both axes.
- `returned_to_origin` is the discriminating field: it requires tracking the whole path
  rather than only the final position, and a model that reconstructs the endpoint by
  counting net displacement will get x and y right and this one wrong.
- Each field is checked separately so partial tracking scores partly.
- Bare JSON, no fence, no working shown.

The correct answers are: x {x}, y {y}, facing "{headings[heading]}",
returned_to_origin {visited_origin}.
"""

    write_task(
        "statetrack",
        "robot-grid",
        difficulty="medium",
        prompt=prompt,
        criteria=criteria,
        checks=json_checks(
            {
                "x": x,
                "y": y,
                "facing": headings[heading],
                "returned_to_origin": visited_origin,
            }
        ),
    )


def build() -> None:
    ledger_balance()
    stack_machine()
    robot_grid()
