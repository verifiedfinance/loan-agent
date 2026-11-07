# Headroom experiment: evaluation plan (fixed 2026-11-06, before any system is run)

A separate experiment from the main paper story. Question: on the questions a credit analyst asks each quarter
(is the borrower in compliance, how far can earnings fall, how much more debt can it take), does a small open model
that formalizes the covenant and lets Z3 compute the answers beat the same model reading the clause directly, and how
does it compare with larger models reading directly?

## Clauses (borrower-disjoint)
`experiments/clean_split/test_unseen.jsonl`: 533 financial-covenant clauses whose borrowers never appear in any
training data (built by `build_unseen_test.py`; near-copies of training clauses removed).

## Reference reading of each clause (`build_specs.py` -> `specs.jsonl`)
Each clause is read into a structured spec: the line items its covenants use (name, description, role: earnings /
debt / other), and per limit: numerator items with signs, denominator items with signs (none for a level limit such as
net worth), comparison, threshold or date schedule of thresholds, unit, and an optional condition flag.
- Reader A: Claude (claude-sonnet-5-5) proposes the items and limits.
- Reader B: Qwen3.8-27B (a different model family) independently reads the clause given only A's item names and
  descriptions, and writes its own limits.
- A spec is kept only if A and B agree exactly on every limit (items, signs, comparison, thresholds, dates, condition)
  and every threshold and date appears in the clause text, and the spec has an earnings item or an outstanding-debt
  item (a clause that only names a ratio without its parts allows only the status question and is excluded). Clauses with thresholds that depend on other quantities
  (e.g. "$60M plus 75% of equity proceeds"), unparseable structure, or reader disagreement are excluded with a reason.
- Not human-verified; stated as a limitation. An expert-labelled subset can be added later.

## Cases (`make_cases.py` -> `cases.jsonl`)
A case = clause + a constructed compliance certificate: the test date and the amount of every line item in the spec
(the ratio itself is never given), plus status facts. All figures are constructed and labelled as such. Per clause:
the binding limit placed at a cushion of -5%, -1%, +1%, +5% and +20% of the earnings item (or of the limit's own
item if no earnings item); for date schedules, the same at a date in each schedule step; one case with one line
item not reported.

## Questions and reference answers (`headroom.py`)
1. Status: comply / breach / undetermined (undetermined when a needed item is not reported).
2. Earnings cushion: the largest percentage decrease of the earnings item (all other items unchanged) that keeps every
   limit satisfied; negative if already in breach; "none" if no limit depends on it; "undetermined" if not computable.
3. Debt capacity: the largest increase of the debt item in US dollars that keeps every limit satisfied; negative if
   already in breach; "none" if no limit depends on it (unlimited); "undetermined" if not computable (a needed item
   is not reported, or no level of debt restores compliance, e.g. a breach of a limit that does not involve debt).
A ratio whose denominator is not positive fails its limit (e.g. negative EBITDA breaches a leverage covenant). The
search for cushion and capacity stays within the range where every ratio's denominator in the reference reading is
positive; both arms use the same range. Reference answers are computed from the spec by exact evaluation and bisection. Validation before any system is run:
the spec rendered as an SMT formula and passed through the SAME formula-arm code must reproduce every reference
answer (100% required).

## Systems
- Direct arm: the model reads clause + certificate and answers the three questions (JSON). Claude Sonnet 5.5,
  Qwen3.8-27B, Qwen2.5-Coder-32B, Qwen2.5-Coder-7B.
- Formula arm: the model writes `covenant_holds` over exactly the certificate's parameter names (given in the prompt,
  with descriptions; plus `test_date` as an integer YYYYMMDD and any condition flags as Bool). Facts come directly
  from the certificate (no fact-writing model). Z3 decides the status; cushion and capacity by bisection with Z3.
  Systems: our SFT-789 and SFT-789 + execution RL (seed 42), and Qwen2.5-Coder-7B prompted (zero-shot, same prompt).
  Greedy decoding throughout.

## Metrics and tests
- Status accuracy; cushion correct if within 0.5 percentage points of the reference (exact match for none /
  undetermined); capacity correct if within 0.5% of the current debt (mirrors the 0.5-point cushion tolerance; exact match
  for none / undetermined). Per-case score = mean of the three; clause score = mean over its cases.
- Primary: SFT-789 + execution RL formula arm vs Qwen2.5-Coder-7B direct arm, mean per-clause score, paired
  sign-flip test over clauses (10,000 permutations, seed 0), 95% clause-bootstrap interval.
- Secondary (Holm within the table): every formula-arm system vs every direct-arm system; per question; per case
  type. All results reported, whatever their direction.

## Post-hoc sensitivity analysis (added 2026-11-06 10:2x, AFTER seeing the formula-arm results; not part of the primary)
The RL model often omits required parameters it does not use (e.g. test_date): 48 of 163 clauses fail the "exactly
these parameters" rule (SFT-789: 8). The primary analysis keeps the pre-registered strict rule. As a labelled
sensitivity analysis, formulas whose parameters are a correctly typed subset of the required list (any order) are
also evaluated (`run_formula.py ... --lenient`, answers in answers_lenient/); unused parameters cannot change the
answers. Reported separately, never in place of the primary.

## Sensitivity: answer-length limit (added 2026-11-06 ~12:20, after seeing the 7B and 32B direct results)
The direct arm used a 1,024-token answer limit (not fixed in this plan). 54/1,064 (7B) and 151/1,064 (32B) answers
were cut off and count as wrong. As a labelled sensitivity analysis, every cut-off direct answer is regenerated with a
4,096-token limit (`rerun_truncated.py`, answers in answers_long/); all other answers are kept. Reported next to the
primary numbers, never in place of them.

## RL v2 (added 2026-11-06 ~17:00, after the results above; a new run, reported as such)
The first RL model (exec_rl warm_fresh) was trained on the old prompt format and often drops required parameters.
RL v2 = same start (SFT-789), same 1,000 unseen training clauses (pool_fresh.jsonl, train borrowers only), same GRPO
settings (1,000 steps, lr 5e-6, 4 generations, T=0.8, beta 0, seed 42), with two changes:
- prompts in the formula-arm format of this experiment (exact parameter list; items named by reader A, Claude, for
  each training clause: `rl_v2_pool.py`); only clauses without date schedules or condition flags are kept, because the
  execution reward cannot score those fairly with an open test date / flag;
- reward `exec_sig` (train_exec_grpo.py): 0 unless the formula parses and keeps exactly the required parameters,
  otherwise 0.3 + 0.7 x the execution score (same scenarios as before, test_date fixed to 2099-12-31).
Evaluated on the same 1,064 cases with the strict rule. Comparisons: RL v2 vs SFT-789 formula (does RL add anything?),
RL v2 vs Qwen2.5-Coder-7B direct (primary question of this experiment, same statistics). One seed; reported as such.

## Exploratory: formal verification and cross-model agreement (added 2026-11-06 ~19:30, after the results above)
`verify_equivalence.py`: Z3 checks each generated formula against the reference reading for ALL inputs (items > 0,
reference denominators > 0, test_date 1990-2030, flags free). Sanity: every reference is equivalent to itself (20/20)
and a +10% threshold mutation is always detected (20/20). Also a label-free trust signal: two independently generated
formulas for the same clause are checked for equivalence with each other (no reference used). Exploratory; reported
as such.
RL v2 replication (added 2026-11-06 ~22:30, after seed 42): seeds 43 and 44, identical configuration (rl_v2/run_s43,
rl_v2/run_s44; eval names rl789_v2_s43, rl789_v2_s44). A gain over SFT-789 is claimed only if all three seeds show it.
