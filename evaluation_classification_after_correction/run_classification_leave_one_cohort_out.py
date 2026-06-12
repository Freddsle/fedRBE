"""
Runs a federated random forest classification on batch effect and non batch effect corrected
data and compares the results. Uses a native simulation approach.
"""
from pathlib import Path
from helper.helper_run_classification_report_metrics import (
    ClassificationExperimentLeaveOneCohortOut,
    DataInfo,
    ResultFile,
)

SCRIPT_FOLDER = Path(__file__).parent
EVALUATION_DATA_FOLDER = SCRIPT_FOLDER.parent / "evaluation_data"
NUM_RUNS = 10

RESULTS_FILE = ResultFile(SCRIPT_FOLDER / "results" / "classification_metric_report.csv")

for num_run in range(NUM_RUNS):
    seed = 42 + num_run

    # SIMULATED
    # Balanced Simulated Data
    folder_balanced = EVALUATION_DATA_FOLDER / "simulated" / "balanced"
    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_balanced / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_balanced / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Ovarian cancer Data
    folder_ovarian_cancer = EVALUATION_DATA_FOLDER / "ovarian_cancer"
    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ovarian_cancer / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ovarian_cancer / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Ovarian cancer Data
    folder_ovarian_cancer = EVALUATION_DATA_FOLDER / "ovarian_cancer"
    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ovarian_cancer / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ovarian_cancer / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # E. coli Data
    folder_ecoli = EVALUATION_DATA_FOLDER / "ecoli"
    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ecoli / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ecoli / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # ccRCC Data
    folder_ccRCC = EVALUATION_DATA_FOLDER / "ccRCC_studies"
    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ccRCC / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ccRCC / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Multiomics Data
    folder_multiomics = EVALUATION_DATA_FOLDER / "multiomics" / "merged_omics"
    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_multiomics / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_multiomics / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)
