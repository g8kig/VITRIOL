### Optimal config search algorithm

Sweep over ctx ∈ [max_ctx, max_ctx/2, max_ctx/4], ubatch ∈ [128, 256].
For each (ctx, ubatch), find max feasible pin (step 2, 0..min(block_count, 24)).
Score by (pin, ctx, -ubatch).

Heuristics (clearly labeled):
- draft_n_max = min(5, max(1, block_count / 8)) — architecture scaling rule
- No pin cap — recommend max feasible pin (sweep finds true optimum)
- ubatch defaults to 128 (experimentally proven faster than 256)

Output struct:

