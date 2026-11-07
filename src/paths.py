"""Every data, model and result location of the repository (one place to change).

Code lives in src/<group>/; each entry module puts src/ and its sub-folders on sys.path, so modules import each other by
plain name (e.g. `import headroom`, `from decide import decide`).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / 'data'
RAW = DATA / 'raw/loan_data'                   # Ganglmair & Wardlaw EDGAR loan agreements (zip + control files)
PROCESSED = DATA / 'processed'                 # clause records extracted from the agreements
SPLITS = DATA / 'splits'                       # train / validation / test splits and the borrower-disjoint test set
SFT_DATA = DATA / 'sft'                        # 789 teacher-written covenant -> SMT pairs
RL_DATA = DATA / 'rl'                          # unlabeled RL clause pools
HEADROOM_DATA = DATA / 'headroom'              # reference readings, cases, validation of the headroom benchmark

MODELS = ROOT / 'models'                       # LoRA adapters on Qwen2.5-Coder-7B-Instruct
RESULTS = ROOT / 'results'
HEADROOM_RESULTS = RESULTS / 'headroom'        # formulas, answers, verification, scores
PAPER = RESULTS / 'paper'                      # figures and LaTeX tables
LOGS = ROOT / 'logs'

TRAIN = SPLITS / 'train_unlabeled.jsonl'
VALIDATION = SPLITS / 'validation.jsonl'
TEST = SPLITS / 'test.jsonl'
VALIDATION_250 = SPLITS / 'validation_250.jsonl'
TEST_600 = SPLITS / 'test_600.jsonl'
TEST_UNSEEN = SPLITS / 'test_unseen.jsonl'
SFT_PAIRS = SFT_DATA / 'sft_pairs.jsonl'
POOL_RATIO = RL_DATA / 'pool_ratio.jsonl'      # the 1,000 clauses the SFT pairs were written for
POOL_FRESH = RL_DATA / 'pool_fresh.jsonl'      # 1,000 unseen training clauses for RL
RL_V2_POOL = RL_DATA / 'rl_v2/pool.jsonl'
