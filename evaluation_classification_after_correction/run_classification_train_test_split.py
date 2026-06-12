"""
Runs a federated random forest classification on batch effect and non batch effect corrected
data and compares the results. Uses a native simulation approach.
"""
from pathlib import Path
from helper.helper_run_classification_report_metrics import (
    ClassificationExperimentTrainTestSplit,
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
    ClassificationExperimentTrainTestSplit(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_balanced / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentTrainTestSplit(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_balanced / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Ovarian cancer Data
    folder_ovarian_cancer = EVALUATION_DATA_FOLDER / "ovarian_cancer"
    ClassificationExperimentTrainTestSplit(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ovarian_cancer / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentTrainTestSplit(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ovarian_cancer / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Ovarian cancer Data
    folder_ovarian_cancer = EVALUATION_DATA_FOLDER / "ovarian_cancer"
    ClassificationExperimentTrainTestSplit(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ovarian_cancer / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentTrainTestSplit(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ovarian_cancer / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # E. coli Data
    folder_ecoli = EVALUATION_DATA_FOLDER / "ecoli"
    ClassificationExperimentTrainTestSplit(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ecoli / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentTrainTestSplit(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ecoli / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # ccRCC Studies Data
    folder_ccRCC = EVALUATION_DATA_FOLDER / "ccRCC_studies"
    ClassificationExperimentTrainTestSplit(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_ccRCC / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentTrainTestSplit(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_ccRCC / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Multiomics Data
    folder_multiomics = EVALUATION_DATA_FOLDER / "multiomics" / "merged_omics"
    ClassificationExperimentTrainTestSplit(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_multiomics / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentTrainTestSplit(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_multiomics / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)
