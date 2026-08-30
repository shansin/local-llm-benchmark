# statetrack / robot-grid

**Prompt:** A robot stands on an 8x8 grid at cell (0, 0), facing north. Cells are
addressed (x, y): x increases to the east, y increases to the north. The grid wraps —
moving north from y=7 arrives at y=0, and moving west from x=0 arrives at
x=7.

The robot obeys three commands:

- `F` — move one cell forward in the direction it is facing
- `L` — turn 90 degrees left, without moving
- `R` — turn 90 degrees right, without moving

Execute this sequence:

L L F F R L F F F R F F F L R R L R F R F R F L R R L L R R L F F F L F L F F R

Return only a JSON object, no code fence, no commentary:

{"x": <number>, "y": <number>, "facing": "<north|east|south|west>", "returned_to_origin": <number>}

where `returned_to_origin` counts how many times the robot arrived back at cell (0, 0)
during the sequence — not counting the start.

## Repeat 1 (seed 0)

- Score: 1.0/10 — Response is [TIMEOUT] with no JSON output, so no correct fields are provided and instructions are not followed.
- Tokens/s: 0.0
- Prefill tok/s: 0.0
- TTFT: 0.00s
- Gen Time: 0.00s
- Output Tokens: 0

**Answer (as scored):**

[TIMEOUT]

