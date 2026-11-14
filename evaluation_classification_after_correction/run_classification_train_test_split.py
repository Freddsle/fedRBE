"""Run the repeated 70/30 train/test classification, correcting with fedRBE fitted on the training data."""

from helper.train_test_split import run_train_test_split_experiments


if __name__ == "__main__":
    run_train_test_split_experiments()
