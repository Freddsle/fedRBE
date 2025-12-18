"""
Runs a federated random forest classification on batch effect and non batch effect corrected
data and compares the results. Uses a native simulation approach.
"""
from pathlib import Path
from helper_run_classification_report_metrics import ClassificationExperimentTrainTestSplit

SCRIPT_FOLDER = Path(__file__).parent  # TODO: Update if needed
EVALUATION_DATA_FOLDER = SCRIPT_FOLDER.parent / "evaluation_data"  # TODO: Update if needed

# Balanced Simulated Data
folder_balanced = EVALUATION_DATA_FOLDER / "simulated" / "balanced"
experiment_balanced_uncorrected = ClassificationExperimentTrainTestSplit(
    data_name = "Balanced Simulated Data",
    preprocessing_name="uncorrected",
    input_folders=[
        str(folder_balanced / "before" / "lab1"),
        str(folder_balanced / "before" / "lab2"),
        str(folder_balanced / "before" / "lab3"),
    ],
    output_base_folder= str(folder_balanced / "after" / "individual_results"),
)

experiment_balanced_corrected = ClassificationExperimentTrainTestSplit(
    data_name = "Balanced Simulated Data",
    preprocessing_name="corrected",
    input_folders=[
        str(folder_balanced / "after" / "individual_results" / "lab1"),
        str(folder_balanced / "after" / "individual_results" / "lab2"),
        str(folder_balanced / "after" / "individual_results" / "lab3"),
    ],
    output_base_folder= str(folder_balanced / "after" / "individual_results"),
)
experiment_balanced_uncorrected.run_experiment()
experiment_balanced_corrected.run_experiment()


# Mildly Imbalanced Simulated Data
folder_mildly_imbalanced = EVALUATION_DATA_FOLDER / "simulated" / "mild_imbalanced"
experiment_mildly_imbalanced_uncorrected = ClassificationExperimentTrainTestSplit(
    data_name = "Mildly Imbalanced Simulated Data",
    preprocessing_name="uncorrected",
    input_folders=[
        str(folder_mildly_imbalanced / "before" / "lab1"),
        str(folder_mildly_imbalanced / "before" / "lab2"),
        str(folder_mildly_imbalanced / "before" / "lab3"),
    ],
    output_base_folder= str(folder_mildly_imbalanced / "after" / "individual_results"),
)
experiment_mildly_imbalanced_corrected = ClassificationExperimentTrainTestSplit(
    data_name = "Mildly Imbalanced Simulated Data",
    preprocessing_name="corrected",
    input_folders=[
        str(folder_mildly_imbalanced / "after" / "individual_results" / "lab1"),
        str(folder_mildly_imbalanced / "after" / "individual_results" / "lab2"),
        str(folder_mildly_imbalanced / "after" / "individual_results" / "lab3"),
    ],
    output_base_folder= str(folder_mildly_imbalanced / "after" / "individual_results"),
)
experiment_mildly_imbalanced_uncorrected.run_experiment()
experiment_mildly_imbalanced_corrected.run_experiment()

# Strong Imbalanced Simulated Data
folder_strongly_imbalanced = EVALUATION_DATA_FOLDER / "simulated" / "strong_imbalanced"
experiment_strongly_imbalanced_uncorrected = ClassificationExperimentTrainTestSplit(
    data_name = "Strongly Imbalanced Simulated Data",
    preprocessing_name="uncorrected",
    input_folders=[
        str(folder_strongly_imbalanced / "before" / "lab1"),
        str(folder_strongly_imbalanced / "before" / "lab2"),
        str(folder_strongly_imbalanced / "before" / "lab3"),
    ],
    output_base_folder= str(folder_strongly_imbalanced / "after" / "individual_results"),
)
experiment_strongly_imbalanced_corrected = ClassificationExperimentTrainTestSplit(
    data_name = "Strongly Imbalanced Simulated Data",
    preprocessing_name="corrected",
    input_folders=[
        str(folder_strongly_imbalanced / "after" / "individual_results" / "lab1"),
        str(folder_strongly_imbalanced / "after" / "individual_results" / "lab2"),
        str(folder_strongly_imbalanced / "after" / "individual_results" / "lab3"),
    ],
    output_base_folder= str(folder_strongly_imbalanced / "after" / "individual_results"),
)
experiment_strongly_imbalanced_uncorrected.run_experiment()
experiment_strongly_imbalanced_corrected.run_experiment()

# Proteomics Data
folder_proteomics = EVALUATION_DATA_FOLDER / "proteomics"
experiment_proteomics_uncorrected = ClassificationExperimentTrainTestSplit(
    data_name = "Proteomics Data",
    preprocessing_name="uncorrected",
    input_folders=[
        str(folder_proteomics / "before" / "lab_A"),
        str(folder_proteomics / "before" / "lab_B"),
        str(folder_proteomics / "before" / "lab_C"),
        str(folder_proteomics / "before" / "lab_D"),
        str(folder_proteomics / "before" / "lab_E"),
    ],
    output_base_folder= str(folder_proteomics / "after" / "individual_results"),
)
experiment_proteomics_corrected = ClassificationExperimentTrainTestSplit(
    data_name = "Proteomics Data",
    preprocessing_name="corrected",
    input_folders=[
        str(folder_proteomics / "after" / "individual_results" / "lab_A"),
        str(folder_proteomics / "after" / "individual_results" / "lab_B"),
        str(folder_proteomics / "after" / "individual_results" / "lab_C"),
        str(folder_proteomics / "after" / "individual_results" / "lab_D"),
        str(folder_proteomics / "after" / "individual_results" / "lab_E"),
    ],
    output_base_folder= str(folder_proteomics / "after" / "individual_results"),
)
experiment_proteomics_uncorrected.run_experiment()
experiment_proteomics_corrected.run_experiment()

# Microarray Data
folder_microarray = EVALUATION_DATA_FOLDER / "microarray"
experiment_microarray_uncorrected = ClassificationExperimentTrainTestSplit(
    data_name = "Microarray Data",
    preprocessing_name="uncorrected",
    input_folders=[
        str(folder_microarray / "before" / "GSE14407"),
        str(folder_microarray / "before" / "GSE26712"),
        str(folder_microarray / "before" / "GSE38666"),
        str(folder_microarray / "before" / "GSE40595"),
        str(folder_microarray / "before" / "GSE6008"),
        str(folder_microarray / "before" / "GSE69428"),
    ],
    output_base_folder= str(folder_microarray / "after" / "individual_results"),
)
experiment_microarray_corrected = ClassificationExperimentTrainTestSplit(
    data_name = "Microarray Data",
    preprocessing_name="corrected",
    input_folders=[
        str(folder_microarray / "after" / "individual_results" / "GSE14407"),
        str(folder_microarray / "after" / "individual_results" / "GSE26712"),
        str(folder_microarray / "after" / "individual_results" / "GSE38666"),
        str(folder_microarray / "after" / "individual_results" / "GSE40595"),
        str(folder_microarray / "after" / "individual_results" / "GSE6008"),
        str(folder_microarray / "after" / "individual_results" / "GSE69428"),
    ],
    output_base_folder= str(folder_microarray / "after" / "individual_results"),
)
experiment_microarray_uncorrected.run_experiment()
experiment_microarray_corrected.run_experiment()

# Microbiome Data
folder_microbiome = EVALUATION_DATA_FOLDER / "microbiome"
experiment_microbiome_uncorrected = ClassificationExperimentTrainTestSplit(
    data_name = "Microbiome Data",
    preprocessing_name="uncorrected",
    input_folders=[
        str(folder_microbiome / "before" / "China1"),
        str(folder_microbiome / "before" / "China3"),
        str(folder_microbiome / "before" / "China5"),
        str(folder_microbiome / "before" / "France1"),
        str(folder_microbiome / "before" / "Germany1"),
        str(folder_microbiome / "before" / "Germany2"),
    ],
    output_base_folder= str(folder_microbiome / "after" / "individual_results"),
)
experiment_microbiome_corrected = ClassificationExperimentTrainTestSplit(
    data_name = "Microbiome Data",
    preprocessing_name="corrected",
    input_folders=[
        str(folder_microbiome / "after" / "individual_results" / "China1"),
        str(folder_microbiome / "after" / "individual_results" / "China3"),
        str(folder_microbiome / "after" / "individual_results" / "China5"),
        str(folder_microbiome / "after" / "individual_results" / "France1"),
        str(folder_microbiome / "after" / "individual_results" / "Germany1"),
        str(folder_microbiome / "after" / "individual_results" / "Germany2"),
    ],
    output_base_folder= str(folder_microbiome / "after" / "individual_results"),
)
experiment_microbiome_uncorrected.run_experiment()
experiment_microbiome_corrected.run_experiment()
