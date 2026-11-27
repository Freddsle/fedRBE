"""Run the leave-one-cohort-out classification, correcting with fedRBE fitted on the training cohorts."""

from helper.loco import run_loco_experiments


if __name__ == "__main__":
    run_loco_experiments()
