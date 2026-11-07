# Data and benchmark construction

This section describes, step by step, where every clause, training example and test case comes from, and how the
headroom benchmark and its reference answers were built and checked. All counts are read from the files in `data/`; the
code for each step is named in brackets.

```
EDGAR credit agreements (14,526)
  └─ clause extraction ─────────────── 340,798 clause records, 17 keyword categories
       └─ financial-ratio clauses ───── 12,517 records
            └─ grouped split ────────── train 10,016 | validation 1,251 | test 1,250
                 ├─ train ──── SFT source clauses (1,000) ── teacher pairs, filtered ── 789 SFT pairs
                 ├─ train ──── unseen RL clauses (1,000) ─── RL v2 prompts (586, no schedules or conditions)
                 └─ validation + test ── borrower-disjoint, deduplicated ── 533 clauses (361 borrowers)
                        └─ two independent readings that agree ── 163 reference specs
                               └─ constructed certificates ─── 1,064 cases over 143 clauses (the benchmark)
```

## 1. Source corpus

The clauses come from the loan-agreement text data of Ganglmair and Wardlaw (Zenodo record 20014336; `data/raw/
loan_data/`). It contains the full plain text of credit agreements that public companies filed with the SEC as exhibits
on EDGAR, re-flowed from the filings' HTML. The control file lists 14,526 agreement documents dated 1990–2017 (most from
1996–2008) with the borrower's name, the agreement date, Compustat `gvkey` (present for 96.9% of documents), the EDGAR
filing path (which gives the borrower's CIK) and the line ranges of the covenant and event-of-default sections. The
archive's checksums are verified before use. Borrowers are SEC registrants, so the corpus covers public companies.
Lender identities are not part of the data (only Dealscan linkage keys are), and the corpus holds no filled-in
compliance certificates (Section 6).

## 2. Clause extraction [`data_prep/extract_covenant_sections.py`]

Each covenant and event-of-default section is cut into self-contained clauses using the corpus's clause breakpoints
(142,704 records), with fallbacks on numbered headings when breakpoints are missing or misaligned (198,094 records).
The heading of the enclosing article is kept with each clause as context. This gives 340,798 clause records from
13,586 agreements.

Each record gets one category from keyword rules applied first to the clause's opening 250 characters, then to its
full text. A clause is a **financial-ratio** clause if it mentions *ratio, leverage, interest coverage, fixed charge
coverage* or *debt service coverage*; other categories (financial thresholds, reporting, restricted payments, liens,
and so on) follow in a fixed order. These are heuristic labels, not human annotations; a single-annotator spot
check of 50 financial-ratio records found 2 mislabels. All later steps use the 12,517 financial-ratio records.

## 3. Splits

### 3.1 Grouped train / validation / test split [`data_prep/build_rl_datasets.py`]

Records are grouped so that all clauses of one agreement, and all agreements that share an identical (whitespace- and
case-normalised) clause text, stay together (7,744 groups; largest 30 records). Groups are assigned in a seeded hash
order (seed 42) to train / validation / test with target shares 80 / 10 / 10%:

| Split | Clauses | Agreements |
|---|---|---|
| train | 10,016 | 6,522 |
| validation | 1,251 | 809 |
| test | 1,250 | 835 |

### 3.2 Borrower-disjoint test set [`data_prep/build_unseen_test.py`]

A data audit (`docs/data_audit.md`) found that this split separates agreements but not borrowers: 64% of test clauses
come from a borrower that also has agreements in train, and companies reuse their own covenant wording across
agreements (114 of 600 sampled test clauses had a same-borrower training clause with token-set Jaccard ≥ 0.9, against
1 for a random training sample). All models in this work were trained only on the train split, so we built a new test
set that no training data can leak into:

| Step | Clauses left |
|---|---|
| Validation + test clauses, minus the 250 validation clauses used as the RL development set | 2,251 |
| Borrower (gvkey; CIK if gvkey is missing) never appears in train | 846 |
| Text is not a copy of any training clause after lower-casing and masking numbers | 845 |
| No training clause from any borrower has token-set Jaccard ≥ 0.7 with it | **533** |

The result, `data/splits/test_unseen.jsonl`, has 533 clauses from 385 agreements and 361 borrowers, dated 1990–2016
(14 / 229 / 109 / 85 / 73 / 23 clauses in 1990–94 / 95–99 / 2000–04 / 05–09 / 10–14 / 15–16).

## 4. Training data

### 4.1 Supervised pairs [`training/make_sft_pairs.py`]

The SFT source is 1,000 financial-ratio clauses from the train split (`data/rl/pool_ratio.jsonl`). For each, a teacher
model (Claude Sonnet 5.5) writes an SMT-LIB formula `covenant_holds(...)` with a comment explaining every parameter.
A pair is kept only if (i) Z3 parses the formula and it stays in the supported fragment, (ii) a semantic review by the
same model accepts it as faithful to the clause, (iii) every parameter has exactly one grounded comment, and (iv) the
whole chat fits 4,096 tokens. 789 pairs survive (`data/sft/sft_pairs.jsonl`):

| Outcome of the 1,000 attempts | Count |
|---|---|
| Accepted (789 also fit the context) | 799 |
| Rejected by the semantic review | 123 |
| Invalid SMT or outside the supported fragment | 48 |
| Missing or duplicate parameter comments | 12 |
| Teacher asked for more context than the clause gives | 7 |
| Symbols not grounded in the clause | 3 |
| Errors | 8 |

The teacher and the reviewer are the same model; the benchmark therefore does not use that model's judgement for its
answers (Section 5.2 uses it only as one of two independent readers).

### 4.2 Unlabeled RL clauses [`data_prep/build_fresh_pool.py`, `headroom/rl_v2_pool.py`]

RL uses 1,000 further financial-ratio clauses from train agreements that appear neither among the SFT source clauses
nor in the SFT pairs, 200–2,543 characters long, sampled with a fixed seed from 7,470 candidates
(`data/rl/pool_fresh.jsonl`). No labels are attached. For RL v2, the line items of each clause are named by one reading
(Claude, the same format as reader A below), so that the prompt can list the formula's exact parameters as in the
evaluation; only clauses without date schedules or condition flags are kept, because the execution reward cannot score
those fairly. That leaves 586 prompts (`data/rl/rl_v2/pool.jsonl`; 242 excluded for schedules or flags, 136 as
unsupported, 36 for thresholds or dates not found in the text). Training and test borrowers are disjoint by
construction.

## 5. The headroom benchmark

### 5.1 Task

Each case gives a covenant clause and a compliance certificate for one test date: the amount of every line item the
clause's covenants use (for example consolidated EBITDA, consolidated total debt, interest expense), never the ratio
itself, plus status facts. The analyst's three questions are:

1. **Status**: comply, breach, or undetermined (undetermined when a needed figure is not reported).
2. **Earnings cushion**: the largest percentage decrease of the earnings item, all other figures unchanged, that keeps
   every applicable covenant satisfied (negative if the borrower is already in breach).
3. **Debt capacity**: the largest increase of the outstanding-debt item, in US dollars, that keeps every applicable
   covenant satisfied (negative if already in breach).

"None" is the answer when no covenant depends on the item, and "undetermined" when it cannot be computed. Questions 2
and 3 are asked only when the clause has an earnings item or a debt item.

### 5.2 Reference readings [`headroom/build_specs.py`]

Every one of the 533 test clauses is read into a structured **spec**:

- **items**: the defined amounts the covenants use, each with a name, the clause's own description, and a role
  (*earnings*: the single earnings measure; *debt*: the single outstanding-debt balance; otherwise *other*);
- **flags**: conditions that switch a limit on or off (for example "during an acquisition period");
- **limits**: for each covenant, the numerator items with signs, the denominator items with signs (empty for a level
  limit such as minimum net worth), the comparison that means compliance (≤, <, ≥, >), the unit (ratio, percentage,
  US dollars), the threshold or a schedule of thresholds by test date, and an optional condition flag.

Two readers from different model families produce the spec independently:

- **Reader A** (Claude Sonnet 5.5) reads the clause and writes the full spec.
- **Reader B** (Qwen3.8-27B, run locally) receives only reader A's item names and descriptions, then writes its own
  limits: signs, comparisons, units, thresholds, schedules and conditions.

A spec is kept only if (i) the two readers' limits are identical, compared as sets after normalisation; (ii) every
threshold and schedule date appears in the clause text; (iii) every item used is declared and the earnings and debt
items each appear in some limit; and (iv) the spec has an earnings or a debt item. A clause that only names a ratio
without saying what it is a ratio of admits only the status question and is excluded. Clauses are also excluded when
a threshold depends on other quantities (for example "$60,000,000 plus 75% of equity proceeds"), is redacted, or sits
in a schedule whose dates cannot be pinned to calendar dates.

| Reference readings | Clauses |
|---|---|
| Borrower-disjoint test clauses | 533 |
| Excluded as unsupported by reader A or reader B | 189 |
| Excluded: ratio named without its components (status question only) | 113 |
| Excluded: the two readers disagree (of 225 clauses both read in full) | 62 |
| Excluded: a threshold or date not found in the text | 6 |
| **Kept** | **163** (from 113 borrowers) |

The two readers agree on 72% of the clauses both could read; disagreements are dropped, not adjudicated, which keeps
the references conservative at the cost of somewhat simpler surviving clauses.

### 5.3 Cases [`headroom/make_cases.py`]

For every kept spec, compliance certificates are **constructed** so that the answers fall where they are informative:

1. **Test dates.** One quarter-end date the year after the agreement if the thresholds are constant; otherwise one
   quarter-end date inside each step of the threshold schedule.
2. **Figures.** All line items are drawn at random (log-uniform between about $3 million and $1 billion, seed 2026)
   until every limit that does not involve the varied item holds with at least 10% slack and every numerator and
   denominator is positive (up to 5,000 draws).
3. **Placing the cushion.** The varied item is the earnings item, or the debt item when the clause has no earnings
   item. Its value is then set so that the earnings cushion is exactly −5%, −1%, +1%, +5% or +20% (for a debt-only
   clause, debt at 105, 101, 99, 95 or 80% of its maximum). The −5% and −1% cases are breaches; the others comply,
   two of them by a thin margin.
4. **Conditions.** For each condition flag, one further case with the flag switched on.
5. **Missing figure.** One case per clause in which one needed line item (other than the varied one) is not reported,
   so the status is undetermined.

The certificate text gives the borrower's name, the test date, a fixed status paragraph (agreement in force, loans
outstanding, quarter-end test date, four-quarter period amounts, no waiver, figures constructed for this test), and one
line per item with the clause's own description and the amount in US dollars. Twenty kept specs produce no case
because no set of figures meets step 2 or the target in step 3 cannot be placed; 143 clauses remain.

### 5.4 Reference answers [`headroom/headroom.py`]

Answers are computed from the spec with exact rational arithmetic. A limit applies on a date if a schedule step covers
it and its condition flag (if any) has the required value. A ratio with a non-positive denominator fails its limit (for
example negative EBITDA breaches a leverage covenant). Status is *breach* if any applicable limit fails, *undetermined*
if a limit that needs a missing item is the only open one, otherwise *comply*. The cushion and the capacity are found
by bisection (80 steps) over the varied item, within the range where every reference ratio denominator stays positive;
the same range is used when grading every system, so a system is judged only where the ratios are defined.

### 5.5 Validation

Before any system was run, the reference spec of every clause was rendered as an SMT formula and passed through **the
same Z3 code that evaluates the systems' formulas**. It reproduces all 1,064 reference answers (status, cushion and
capacity) exactly. The validation also confirms that each case's figures, dates and flags are wired to the formula
correctly.

### 5.6 Composition

| Property | Value |
|---|---|
| Clauses / agreements / borrowers | 143 / 108 / 103 |
| Agreement years | 1993–2015 |
| Covenant limits per clause: 1 / 2 / 3 / 4 | 111 / 15 / 13 / 4 |
| Limits by unit: ratio / US dollars / percentage | 176 / 11 / 9 |
| Clauses with date schedules or conditions | 27 |
| Clauses with an earnings item only / a debt item only / both | 46 / 49 / 48 |
| Cases | 1,064 |
| Status: comply / breach / undetermined | 556 / 370 / 138 |
| Cases asking the earnings cushion / the debt capacity | 745 / 718 |

### 5.7 Scoring

Status must match exactly. A cushion is correct within 0.5 percentage points and a capacity within 0.5% of the
current debt (the same tolerance in relative terms; "none" and "undetermined" must match exactly). A case's score is the
share of its asked questions answered correctly; a system's score is the mean over clauses of its cases' mean score,
with 95% intervals from 10,000 clause-bootstrap resamples and paired comparisons by a sign-flip test over clauses.

Both evaluation arms see the same case. **Reading directly**, a model gets the clause and the certificate and answers
the three questions in JSON. **Formalizing**, a model gets the clause and the exact list of parameters (the
certificate's items, `test_date` as an integer, and the condition flags) and writes `covenant_holds`; the certificate
figures become solver facts without any model in between, and Z3 computes all three answers. A formula that does not
keep exactly the required parameters, or that Z3 cannot parse, scores zero on every case of its clause.

## 6. What the data is not

- **The certificate figures are constructed.** Borrowers deliver quarterly compliance certificates privately to their
  lenders, so none are filed on EDGAR: across all 18,508 archive files we found blank certificate forms (in about 20% of
  agreements) and six filled at closing between 1995 and 2003, but no delivered quarterly certificate. The benchmark
  therefore tests the computation a lender performs, on realistic clause text with constructed figures, not real-world
  compliance.
- **References are two agreeing model readings, validated through Z3,** not expert annotations.
- **Certificates report each defined term as one amount.** Applying the agreement's own definitions (add-backs, caps,
  exclusions) to raw financial statements is a harder, more realistic task left for future work.
- **The borrowers are public companies,** mostly with agreements from 1996–2008.
