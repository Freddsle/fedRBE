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

    # # SIMULATED
    # # Balanced Simulated Data
    # folder_balanced = EVALUATION_DATA_FOLDER / "simulated" / "balanced"
    # ClassificationExperimentTrainTestSplit(
    #     data_name="Balanced Simulated Data",
    #     preprocessing_name="uncorrected",
    #     datainfo=DataInfo(folder_balanced / "before" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # ClassificationExperimentTrainTestSplit(
    #     data_name="Balanced Simulated Data",
    #     preprocessing_name="corrected",
    #     datainfo=DataInfo(folder_balanced / "after" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # # Mildly Imbalanced Simulated Data
    # folder_mildly_imbalanced = EVALUATION_DATA_FOLDER / "simulated" / "mild_imbalanced"
    # ClassificationExperimentTrainTestSplit(
    #     data_name="Mildly Imbalanced Simulated Data",
    #     preprocessing_name="uncorrected",
    #     datainfo=DataInfo(folder_mildly_imbalanced / "before" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # ClassificationExperimentTrainTestSplit(
    #     data_name="Mildly Imbalanced Simulated Data",
    #     preprocessing_name="corrected",
    #     datainfo=DataInfo(folder_mildly_imbalanced / "after" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # # Strong Imbalanced Simulated Data
    # folder_strongly_imbalanced = EVALUATION_DATA_FOLDER / "simulated" / "strong_imbalanced"
    # ClassificationExperimentTrainTestSplit(
    #     data_name="Strongly Imbalanced Simulated Data",
    #     preprocessing_name="uncorrected",
    #     datainfo=DataInfo(folder_strongly_imbalanced / "before" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # ClassificationExperimentTrainTestSplit(
    #     data_name="Strongly Imbalanced Simulated Data",
    #     preprocessing_name="corrected",
    #     datainfo=DataInfo(folder_strongly_imbalanced / "after" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # # Ovarian cancer Data
    # folder_ovarian_cancer = EVALUATION_DATA_FOLDER / "ovarian_cancer"
    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="uncorrected",
    #     datainfo=DataInfo(folder_ovarian_cancer / "before" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="corrected",
    #     datainfo=DataInfo(folder_ovarian_cancer / "after" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # # Ovarian cancer Data
    # folder_ovarian_cancer = EVALUATION_DATA_FOLDER / "ovarian_cancer"
    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="uncorrected",
    #     datainfo=DataInfo(folder_ovarian_cancer / "before" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="corrected",
    #     datainfo=DataInfo(folder_ovarian_cancer / "after" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # # E. coli Data
    # folder_ecoli = EVALUATION_DATA_FOLDER / "ecoli"
    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="uncorrected",
    #     datainfo=DataInfo(folder_ecoli / "before" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="corrected",
    #     datainfo=DataInfo(folder_ecoli / "after" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # # Quartet Data
    #TODO fix this to use the correct merged data!
    folder_quartet = EVALUATION_DATA_FOLDER / "quartet"
    ClassificationExperimentTrainTestSplit(
        preprocessing_name="uncorrected",
        datainfo=DataInfo(folder_quartet / "before" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    ClassificationExperimentTrainTestSplit(
        preprocessing_name="corrected",
        datainfo=DataInfo(folder_quartet / "after" / "datainfo.json"),
        resultfile=RESULTS_FILE,
    ).run_experiment(seed=seed)

    # # ccRCC Studies Data
    # folder_ccRCC = EVALUATION_DATA_FOLDER / "ccRCC_studies"
    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="uncorrected",
    #     datainfo=DataInfo(folder_ccRCC / "before" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)

    # ClassificationExperimentTrainTestSplit(
    #     preprocessing_name="corrected",
    #     datainfo=DataInfo(folder_ccRCC / "after" / "datainfo.json"),
    #     resultfile=RESULTS_FILE,
    # ).run_experiment(seed=seed)
