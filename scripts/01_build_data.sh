#!/bin/bash
# Stage 1 (CPU): agreements -> clause records -> splits -> borrower-disjoint test set -> unseen RL clause pool.
set -eu; source "$(dirname "$0")/env.sh"
python src/data_prep/extract_covenant_sections.py      # data/raw/loan_data -> data/processed
python src/data_prep/build_rl_datasets.py              # -> data/splits/{train_unlabeled,validation,test}.jsonl
python src/data_prep/build_unseen_test.py              # -> data/splits/test_unseen.jsonl (borrowers never in training)
python src/data_prep/build_fresh_pool.py               # -> data/rl/pool_fresh.jsonl (1,000 unseen training clauses)
