"""Run configured ClassificationExperimentTrainTestSplit experiments with the existing RF implementation."""

from helper.helper_run_classification_report_metrics import (
    ClassificationExperimentTrainTestSplit, run_classification_experiments,
)


if __name__ == "__main__":
    run_classification_experiments(ClassificationExperimentTrainTestSplit)
