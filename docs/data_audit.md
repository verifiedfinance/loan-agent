# Data quality report: `version3/data/financial_project`

**Scope.** An independent verifier recomputed every number below unless it is marked **[unverified]**. Where the verifier's number differed, the corrected value is used. Audit scripts are in `/tmp/claude-1000/audit/` and verifier scripts are in `/tmp/claude-1000/audit/verify/`. No project file was modified.

**Short answer to the questions asked**

| Question | Answer |
|---|---|
| Do we know the borrower? | **Yes**, for every agreement. The usable keys are Compustat gvkey (present for 96.9% of documents) and EDGAR CIK. |
| Do we know the lender? | **Not as a field.** The lender appears only in the agreement text. A regex finds the agent on the cover page in about 70% of all agreements and in 83% of the downstream ones. |
| Do we have compliance certificates? | **No real ones.** About 20% of agreements include a blank certificate form. Six certificates were filled in at closing. No quarterly certificate a borrower actually delivered exists in the data. |
| Do we have borrower financials? | **Only for 48 companies, from SEC XBRL.** This covers 51 of the 600 test clauses, almost all from 2009 or later. |

---

## 1. What we have

| Item | Size | Coverage | Source |
|---|---|---|---|
| `loan_data/main_control_file.json` | 14,526 agreement documents | Agreements dated 1990–2017. 5,887 gvkeys, 6,035 CIKs, 6,818 company names. 13,407 packageids. | Dealscan–EDGAR link file (keys only) |
| `loan_data/publicly_filed_contracts.zip` | 18,508 files (25,032 entries) | Full text. 3,982 files are not listed in the control file. | EDGAR exhibits |
| `processed/rl_rules.jsonl` | 340,798 clauses, 17 categories | Categories come from keyword rules, not human labels | `extract_covenant_sections.py` |
| `processed/financial_ratio_records.jsonl` | 12,517 | Byte-identical to `financial_ratio_rules.jsonl` (same md5) | same |
| `datasets/` train / validation / test | 10,016 / 1,251 / 1,250 clauses | 6,522 / 809 / 835 contracts; all `financial_ratio` | `build_rl_datasets.py` |
| `test_600`, `ood_300` | 600 / 300 | Both are subsets of `test.jsonl` | — |
| `sft_pairs` (789), `pool_fresh` (1,000) | — | Both are strict subsets of train, with identical text | — |
| Downstream `reference_agreement.csv` | 128 included clauses | 123 contracts, all from test_600 | — |
| `cases.jsonl` / `hard/cases.jsonl` | 1,494 / 1,007 cases | Mostly synthetic figures | — |
| `borrower_financials.csv` | 147 rows | 51 clauses, 48 CIKs | SEC XBRL |
| `sec_cache/companyfacts/` | 48 companies | 48 of the 99 candidate CIKs | SEC XBRL |
| **Not on disk** | — | Compustat financials, Dealscan deal or lender data, SIC/industry codes, filed compliance certificates | — |

## 2. Parties

**Borrowers**

- `cnum` is not a company number. It is different on every row (14,526 distinct values). For example, DEERE & CO has 25 documents and 25 different cnums. Use gvkey, with CIK when gvkey is missing (446 documents have no gvkey).
- Documents per gvkey: 2,700 borrowers have 1, the median is 2, the mean is 2.39 and the maximum is 25 (DEERE). Borrowers with 5 or more documents supply 33.5% of all documents.
- 2,908 borrowers have agreements in 2 or more different years, 492 in 5 or more, and 25 in 10 or more.
- **The splits keep contracts apart but not borrowers.** No contract appears in both train and an evaluation set, but many borrowers do:

| Split | Clauses whose borrower (gvkey) is also in train |
|---|---|
| test | 797/1,250 (63.8%); 814 by CIK |
| test_600 | 388/600 (64.7%) |
| ood_300 | 191/300 (63.7%). It is a subset of test, so it is not out-of-distribution by borrower. |
| validation | 754/1,251 (60.3%) |
| downstream 128 | **95/128 (74%)**, corrected from 97 |

- **Train often contains near-copies of test text from the same borrower.** 114 of the 600 test_600 clauses have a train clause from the same borrower with token Jaccard ≥ 0.9, and 182 at ≥ 0.7. A random size-matched train sample gives 1. Example: DEERE's 2005 "SECTION 7. NEGATIVE COVENANTS OF THE CAPITAL CORPORATION…" in test against its 2003 version in train (J = 0.947).
- The README says agreements cover 1996–2017. The real range is 1990–2017: 648 documents are dated before 1996, and the peak is 1997 with 1,078.
- No industry (SIC) or public/private field was found **[unverified negative]**.

**Lenders**

- No file has a lender, agent or bank column. README line 35 says lender identities are excluded on purpose. The fact sheets in both case files have no Lender or Agent line.
- The agreement text usually names the agent on the cover page or in the preamble:
  - Across all 14,526 documents, the audit regex found an agent in 71.6% and an "Administrative Agent" in 54.5%. An independent regex found 70.4% and 53.9%.
  - Downstream, an agent is named in 102 of 123 contracts (82.9%).
  - The 88.9% "any lender name" rate (which adds single-lender detection) is **[unverified]**.
- Syndicated vs bilateral is **[unverified]**. By drafting form, 74.3% look syndicated, 17.0% bilateral and 8.6% unknown. Only 40.6% list 2 or more lenders on their signature pages or commitment schedules. The bilateral share falls from 28% (1995–99) to 3% (2015+).

## 3. Compliance certificates

| Certificate status (all 18,508 zip files) | Files |
|---|---|
| Mention "compliance certificate" | 11,165 (60.3%) |
| Contain "form of compliance certificate" | 5,488 |
| Include the certificate body | 3,653–3,654 (19.7%) |
| Reference the form, but the exhibit is missing | 2,910–2,911 |
| Mention a certificate, with no form | 4,601 |
| Never mention one | 7,343 |

- **Control-file documents only.** 9,038 mention a certificate and about 2,898 include the body, covering 1,793 gvkeys. The share with a body grows over time: 10.2% (1990–94), 11.7%, 16.4%, 19.1%, 36.4%, and 38.6% (2015–17).
- **Almost all included certificates are blank templates.** At least 3,077 bodies have placeholders such as `_____` or `[insert`. This is a lower bound; the authors' broader regex gives 3,225.
- **Filled certificates.** Six were found, all signed at closing between 1995 and 2003 (Shoney's, Stokely, IPEC, General Bearing, Papa John's, FBL). Their figures were confirmed against the text. For example, Shoney's states "The Funded Debt Ratio was 4.06:1.00", and the verifier's arithmetic gives 501,174 / 123,431 = 4.06. The total of six was **not re-derived** independently. All six predate XBRL.
- No standalone certificate document was found **[unverified; 36 candidate files were read by the author]**. Borrowers deliver quarterly certificates privately to lenders, so they are not filed on EDGAR.
- **The `compliance` category in `rl_rules` has nothing to do with certificates.** It holds 16,794 clauses about compliance with laws, environmental rules and ERISA. Only 69 of them mention a compliance certificate. Clauses about delivering certificates are mostly in `reporting` (5,343 of the 7,476 that mention one).
- **Downstream cases.**
  - In `cases.jsonl`, 1,397 of 1,494 cases (93.5%) use synthetic figures. The other 97 use 10-K XBRL figures, covering 34 clauses with expected verdicts of 85 comply, 7 breach and 5 undetermined.
  - All 1,007 cases in `hard/cases.jsonl` are synthetic. 346 of them present their figures as "Other figures reported in the compliance certificate", so these synthetic figures are labelled as if they came from a certificate.
  - The README's "51" refers to clauses that have XBRL data. Only 34 of those produce real cases.

## 4. Borrower financials

- There are 48 SEC companyfacts files and no other financial source. The candidate list held 117 test_600 clauses from 99 CIKs, but only 48 CIKs were downloaded.
- **Coverage:** 51 of the 128 downstream clauses and 51 of test_600 (8.5%). Of these 51 agreements, 50 are dated 2009–2016 and one 2002. Of the test_600 agreements, 28 are dated before 1996, 445 in 1996–2008 and 127 in 2009 or later. Only 7 of the 473 pre-2009 clauses have a cached CIK.
- **Quality of the 147 rows:**
  - 70 rows have warnings and 33 have missing items (interest expense 23, total debt 14, EBITDA 9, equity 3).
  - **Sign errors kept from the original filings.** L-3 FY2010 has a fully negated income statement (revenue −$15.68B, EBITDA −$1.519B). UIL FY2011 has D&A of −$167M, so debt/EBITDA comes out at 29.2 instead of about 4; its FY2012 value is 3.91.
  - **Restatements.** 448 of the 2,827 facts used (15.8%) have more than one value across 10-K filings, 18 flip sign, and 104–108 differ by more than 10%.
  - **Annual figures only.** The cache also holds 541,018 quarterly (10-Q) facts, which are unused. The fact sheets nevertheless describe each test date as a quarter-end.
- EBITDA is computed from 10-K line items, not from each agreement's definition, so it has no add-backs. FCCR = (EBITDA − capex)/interest. Both points are **[unverified]**.
- Compustat is not available. Only the gvkey key exists, for 14,080 of 14,526 documents.

## 5. Covenant variation

**Verified**

| Measure | train | test_600 | pool | downstream 128 |
|---|---|---|---|---|
| Median length (characters) | 832 | ~788 | — | ~620–630 |
| Maximum length (characters) | 55,628 | 9,015 | 2,539 (capped) | 3,713 |
| Records over 2,543 characters | 1,627 (16.2%) | 59 | 0 | 1 |
| Clauses containing dates (schedule signal) | — | 230/600 | — | **0/128** (removed by the downstream sampler) |
| Body identical to a train body | — | 47 (7.8%) | — | 17 (13.3%) |

- After normalizing the text, 64–70 test_600 clauses and 22–25 downstream clauses match a train body. With numbers masked, the counts are 94–102 and 36–40.
- In train, 657 records have an exact full-text duplicate in another contract.
- Clauses per contract in train: 4,113 contracts have 1, 1,619 have 2, 567 have 3, and so on up to 8.
- All 1,000 RL prompts have an empty `source_context`, so no definitions are given.
- 577–582 of the 600 test clauses use a defined financial term, and 540–548 contain no inline definition.

**[unverified] Mix of covenant types and thresholds (train)**

| Type | Share | Median threshold |
|---|---|---|
| Total leverage | 31.5% | 3.5x |
| Fixed charge / debt service coverage | 17.8% | 1.25x |
| Interest coverage | 16.1% | 2.5x |
| Debt to capitalization | 11.4% | 0.70 or 60% |
| Senior / secured leverage | 5.8% | 2.5x |
| Current ratio | 5.5% | 1.25 |
| Net worth | 2.5% | $34M floor |

Also **[unverified]**: about a third of clauses contain several covenant types, and step-down or step-up schedules appear in 29.5% of train clauses but none of the downstream 128. About 22–25% of train records are near-duplicates. Equity-cure language is under 0.5%, and EBITDA add-back language appears in 0.8–2.4% of clauses.

## 6. Quality problems, ranked by impact on our results

| # | Problem | Size | Fix |
|---|---|---|---|
| 1 | Train leaks into test, both by borrower and by identical clause bodies | About 64% of test clauses share a borrower with train. 114/600 have a near-copy from the same borrower. 47 test_600 and 17 downstream bodies are identical to train. | Re-split by gvkey (CIK fallback) and deduplicate on the clause body. At minimum, report results separately for the 212 unseen-borrower clauses and with duplicates removed. Rename `ood_300`. |
| 2 | Real-figure coverage is tiny and limited to 2009 or later | 51/600 test clauses; 34 clauses / 97 real cases | Report as a separate small subset. Fetch the 51 remaining candidate CIKs (up to 117 clauses). |
| 3 | No real certificates; the "certificate" figures are synthetic | 6 filled certificates in the whole archive, none quarterly | State this plainly in the paper. Label `hard/` figures as synthetic. |
| 4 | Clauses come without the agreement's definitions | About 90% of test clauses; all 1,000 RL prompts | Attach definitions taken from the full text. Definitions can be found for nearly all term families (confirmed directionally). |
| 5 | XBRL sign errors and mixed filing vintages | 18 sign flips; 448 restated facts | Take absolute values for flows that should be positive, take each period's facts from a single filing, and hand-check the affected cases. |
| 6 | Downstream drops dated clauses that training contains | 0/128 downstream vs 230/600 test_600 | Add a stratum of clauses with schedules, or state the exclusion as a scope limit. |
| 7 | Keyword category labels are noisy | `financial_threshold` has 841/6,735 Use of Proceeds headings; `indebtedness` has 2,197/24,735. Hand-sample error rates (single annotator): indebtedness 18/40, threshold 10/40, ratio 2/50. | Relabel by section heading. Do not report category statistics as topics. |
| 8 | Thresholds redacted | 4 in test_600 (fin_109981, fin_05123, fin_109984 marked `***`; fin_217628 marked `[confidential]`), 1 in pool (fin_233989) | Drop these, or allow abstention only. |
| 9 | Documentation errors | README says years 1996–2017 (actually 1990–2017) and calls cnum a company number; the `compliance` category name misleads; two files are duplicates; README count "51" | Correct the README and the data card. |
| 10 | No lender field | All rows | Use the regex agent extraction, which finds an agent in 83% of downstream contracts. |

## 7. What this means

**For a realistic compliance benchmark**

- The borrower's certificate has to be **constructed**; none can be collected. Use the agreement's own blank certificate form, which exists for about 20% of agreements and is more common after 2010, as the format. Fill it with figures, and label each case honestly as synthetic or as a 10-K approximation.
- The real-financials tier is at most 51 clauses from 48 large public filers, almost all from 2009 or later. Fix the sign errors and mixed filing vintages first, and report this tier separately.
- Add-back traps (an adjusted EBITDA the agreement does not allow) can only be built if each case includes the agreement's definitions. The clause records alone cannot support them.
- The lender or agent name can go into each case header from the text extraction (83% of downstream contracts). It cannot come from a data field.
- Hold out test agreements by borrower and remove clauses whose body duplicates train. Include dated, step-down clauses.

**For training data**

- Before any new training run, rebuild the splits grouped by gvkey and deduplicate on the normalized clause body. The current test, test_600 and ood_300 numbers overstate generalization.
- Put the relevant definitions into each RL prompt's `source_context`.
- Drop the redacted clauses, and split section-length records longer than 10,000 characters.
- Do not use the `rl_rules` categories as topics without relabeling them.