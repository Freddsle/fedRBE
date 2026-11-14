"""
Paths, names and small utilities shared by the random forest evaluations
(train_test_split.py and loco.py) and their steps (steps/).
"""
import contextlib
import io
from pathlib import Path
import sys

HELPER_FOLDER = Path(__file__).parent
COMPONENT_FOLDER = HELPER_FOLDER.parent
REPO_ROOT = COMPONENT_FOLDER.parent
# batchcorrection, evaluation_utils and generate_fedrbe_corrected_datasets are
# imported from the repository root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_FOLDER = COMPONENT_FOLDER / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)
EVALUATION_DATA = REPO_ROOT / "evaluation_data"

LEARNING_TYPE_FEDERATED = "federated"
LEARNING_TYPE_CENTRALIZED = "centralized"
LEARNING_TYPES = [LEARNING_TYPE_FEDERATED, LEARNING_TYPE_CENTRALIZED]
# Names of the correction arms in the results
PREPROCESSING_UNCORRECTED = "uncorrected"
ARM_LABEL_AWARE = "label_aware"
ARM_LABEL_FREE = "label_free"
ARMS = (ARM_LABEL_FREE, ARM_LABEL_AWARE)
# predicted_client_name of the scores pooled over all test cohorts
ALL_COHORTS = "All"
METRICS = ("MCC", "F1_score")
DEFAULT_SEEDS = list(range(42, 52))  # ten repetitions


def run_quietly(function, *args, **kwargs):
    """Runs function with its output captured, the output's end is shown on failure."""
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            return function(*args, **kwargs)
    except BaseException:
        print("\n".join(log.getvalue().splitlines()[-30:]))
        raise
