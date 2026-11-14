"""
The steps of the random forest evaluations, in pipeline order:

- datasets: the dataset registry and the clients' uncorrected fedRBE inputs
- prepare: train/test parts (70/30 split or held-out cohort) and their input folders
- correct: fedRBE on the training data, and correcting test samples or cohorts
- train_forest: forest input tables and federated or centralized training
- evaluate: predictions, scores and the result file
"""
