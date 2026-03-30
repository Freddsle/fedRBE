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
        data_name="Balanced Simulated Data",
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_balanced / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        data_name="Balanced Simulated Data",
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_balanced / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Mildly Imbalanced Simulated Data
    folder_mildly_imbalanced = EVALUATION_DATA_FOLDER / "simulated" / "mild_imbalanced"
    ClassificationExperimentLeaveOneCohortOut(
        data_name="Mildly Imbalanced Simulated Data",
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_mildly_imbalanced / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        data_name="Mildly Imbalanced Simulated Data",
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_mildly_imbalanced / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Strong Imbalanced Simulated Data
    folder_strongly_imbalanced = EVALUATION_DATA_FOLDER / "simulated" / "strong_imbalanced"
    ClassificationExperimentLeaveOneCohortOut(
        data_name="Strongly Imbalanced Simulated Data",
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_strongly_imbalanced / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        data_name="Strongly Imbalanced Simulated Data",
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_strongly_imbalanced / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Proteomics Data
    folder_proteomics = EVALUATION_DATA_FOLDER / "proteomics"
    ClassificationExperimentLeaveOneCohortOut(
        data_name="Proteomics Data",
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_proteomics / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        data_name="Proteomics Data",
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_proteomics / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # Microarray Data
    folder_microarray = EVALUATION_DATA_FOLDER / "microarray"
    ClassificationExperimentLeaveOneCohortOut(
        data_name="Microarray Data",
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_microarray / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentLeaveOneCohortOut(
        data_name="Microarray Data",
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_microarray / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)
