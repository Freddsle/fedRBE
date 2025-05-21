
<table>
  <tr>
    <td><a href="https://freddsle.github.io/fedRBE/docs/how_to_guide.html"><img src="https://img.shields.io/badge/HowTo_Guide-Click_Here!-007EC6?style=for-the-badge" alt="HowTo Guide"></a></td>
    <td><a href="https://freddsle.github.io/fedRBE/"><img src="https://img.shields.io/badge/Documentation-Click_Here!-007EC6?style=for-the-badge" alt="Documentation"></a></td>
    <td><a href="https://github.com/Freddsle/fedRBE/"><img src="https://img.shields.io/badge/GitHub-Click_Here!-007EC6?style=for-the-badge" alt="GitHub"></a></td>
    <td><a href="https://featurecloud.ai/app/fedrbe"><img src="https://img.shields.io/badge/FeatureCloud_App-Click_Here!-007EC6?style=for-the-badge" alt="FeatureCloud App"></a></td>
  </tr>
</table>

# fedRBE - FeatureCloud <!-- omit in toc -->

**A federated implementation of the limma `removeBatchEffect` method.** 
Supports log scaling and multiple batches per client and secure computation.

- **Open Source & Free**: [GitHub Repository](https://github.com/Freddsle/fedRBE/tree/main/batchcorrection)  
- **Federated Privacy-preserving tool**: Based on [FeatureCloud](https://featurecloud.ai/app/fedrbe) platform  
- **Preprint**: [ArXiv](https://doi.org/10.48550/arXiv.2412.05894)
- **Full Documentation**: [GitHub Pages](https://freddsle.github.io/fedRBE/)

---

## Table of Contents <!-- omit in toc -->
- [Overview](#overview)
- [Prerequisites and setup](#prerequisites-and-setup)
- [Usage](#usage)
  - [Simulating a federated Workflow Locally](#simulating-a-federated-workflow-locally)
  - [Running a true federated workflow](#running-a-true-federated-workflow)
- [Input requirements](#input-requirements)
- [Outputs](#outputs)
- [Configuration (config.yml)](#configuration-configyml)
- [FeatureCloud App states](#featurecloud-app-states)
- [Additional resources](#additional-resources)
- [License](#license)
- [How to cite](#how-to-cite)

---

## Overview
`fedRBE` is a privacy-preserving tool for removing batch effects from omics data distributed across multiple research centers. It leverages the limma's `removeBatchEffect()` method and utilizes **federated learning (FL)** and **secure multi-party computation (SMPC)** to ensure data privacy. Sensitive data remains on each participant's site, and only summary-level information is shared.

fedRBE supports two usage modes:

* [Federated mode](#running-a-true-federated-workflow-login-required): Removes batch effects from decentralized, sensitive data. (Registration required)

* [Simulation mode](#simulating-a-federated-workflow-locally-no-login-required): Runs fedRBE locally to simulate federated batch effect correction. (No registration or login required)

 For advanced parameters, see the [Configuration](#configuration-configyml) section.

---

## Prerequisites and setup

Before using `fedRBE`, ensure:
1. **Docker** is installed ([FeatureCloud prerequisites](https://featurecloud.ai/developers)).
2. **FeatureCloud CLI**:
   ```bash
   pip install featurecloud
   featurecloud controller start
   ```
3. **App Image**:  
   - For linux/amd64:
     ```bash
     # pull the pre-built image
     featurecloud app download featurecloud.ai/bcorrect
     ```
     or directly via Docker
     ```bash
     docker pull featurecloud.ai/bcorrect:latest
     ```
   - Alternatively, If you are using a ARM architecture (e.g., Mac M-series), you may need to build the image locally as shown below._
     ```bash
     docker build . -t featurecloud.ai/bcorrect:latest
     ```

     or build the image from GitHub locally:
     ```bash
      cd batchcorrection
      docker build . -t featurecloud.ai/bcorrect:latest
      ```

The app image which is provided in the docker registry of featurecloud built on the linux/amd64 platform. Especially if you're using a Macbook with any of the M-series chips or any other device not compatible with linux/amd64, please build the image locally.

---

## Usage

### Simulating a federated Workflow Locally (No login required)
To test how `fedRBE` behaves with multiple datasets on one machine:

1. **Ensure the full repository including sample data is cloned and the current working directory**:
   ```bash
   git clone https://github.com/Freddsle/fedRBE.git
   cd fedRBE
   ```

2. **Start the FeatureCloud Controller with the correct input folder**:
   ```bash
   featurecloud controller start --data-dir=./evaluation_data/simulated/mild_imbalanced/before/
   ```

3. **Run a Sample Experiment**:
   ```bash
   # if you have the controller running in a different folder, stop it first
   # featurecloud controller stop 
   featurecloud test start --app-image=featurecloud.ai/bcorrect:latest --client-dirs=lab1,lab2,lab3
   ```
   Alternatively, you can start the experiment from the [frontend](https://featurecloud.ai/development/test/new)

   Select 3 clients, add lab1, lab2, lab3 respecitvely for the 3 clients to their path. 
   
   Use `featurecloud.ai/bcorrect:latest` as the app image.
  
This runs an experiment bundled with the app, illustrating how `fedRBE` works.
The given repository contains the app but furthermore includes all the experiments done with the app.

### Running a true federated workflow (Login required)
For an actual multi-party setting:
1. **Create a Project** in [FeatureCloud](https://featurecloud.ai/projects) and invite at least 3 clients.
2. **Clients Join with Tokens** provided by the coordinator.
3. **Each Client** uploads their data and `config.yml` to their local FeatureCloud instance.
4. **Start the Project**: `fedRBE` runs securely, never sharing raw data.

See [HOW TO GUIDE](https://freddsle.github.io/fedRBE/docs/how_to_guide.html) for guidance on creating and joining projects.
Please note that an account and login is required for this to protect the federated workflow from malicious participants.

---

## Input requirements
- **Data File**: CSV or TSV with either:
  - Samples x Features, or
  - Features x Samples
- **`config.yml`**: Configuration file controlling formats, normalization, and additional parameters.
- **Optional Design Matrix**: CSV/TSV with covariates (samples x covariates).

For details, see the [Configuration](#configuration-configyml) section.

---

## Outputs
Each client after completion receives:
- **`only_batch_corrected_data.csv`**: Batch-corrected features.
- **`report.txt`**: Includes:
  - Excluded features (and why)
  - Calculated beta values
  - Internally used design matrix

**Note**: Output files use the same `separator` defined in `config.yml`.

---

## Configuration (config.yml)
Upload a `config.yml` alongside your data. Adjust parameters as needed:

```yaml
flimmaBatchCorrection:
  data_filename: "lab_A_protein_groups_matrix.tsv"
    # Main data file: either features x samples or samples x features.

  design_filename: "lab_A_design.tsv"
    # Optional design matrix: samples x covariates.
    # Must have first column as sample indices.
    # it is read in the following way:
    # pd.read_csv(design_file_path, sep=seperator, index_col=0)
    # should therefore be in the format samples x covariates
    # with the first column being the sample indices

  expression_file_flag: True
    # True: data_file = features (rows) x samples (columns)
    # False: data_file = samples (rows) x features (columns)
    # format: boolean

  index_col: "sample"
    # If expression_file_flag True: index_col is the feature column name.
    # If expression_file_flag False: index_col is the sample column name.
    # If not given, defaults apply - the index is taken from the 0th column for
    # expression files and generated automatically for samples x features datafiles
    # format: str or int, int is interpreted as the column index (starting from 0)

  covariates: ["Pyr"]
    # Covariates included in the linear model.
    # If no design file, covariates must be present as features in the data file.

  separator: "\t"
    # Separator for main data file.

  design_separator: "\t"
    # Separator for design file.

  batch_col: "batch"
    # Column name in the design file that contains batch information 
    # (if multiple batches present in one client).
    # If not given, all client data is considered as one batch.
    # format: str

  normalizationMethod: "log2(x+1)"
    # Normalization: "log2(x+1)" or None.
    # If None, no normalization is applied.
    # More options will be available in future versions.

  smpc: True
    # Enable secure multiparty computation for privacy-preserving aggregation.
    # For more information see https://featurecloud.ai/assets/developer_documentation/privacy_preserving_techniques.html#smpc-secure-multiparty-computation

  min_samples: 5      # format: int
    # Minimum samples per feature required. Adjusted for privacy if needed.
    # If for a feature less than min_samples samples are present,
    # the client will not send any information about that feature
    # Please note that the actual used min_samples might be different
    # as for privacy reasons min_samples = max(min_samples, len(design.columns)+1)
    # This is to ensure that a sent Xty matrix always has more samples
    # than features so that neither X not y can be reconstructed from the Xty matrix.

  position: 1      # format: int
    # Defines client order. The last client in order is the reference batch.
    # Example:
    #  C1(position=0), C2(position=2), C3(position=1) -> Order: C1, C3, C2 (C2 is reference).
    # If empty/None, the order is random, making the batch correction run non deterministic

  reference_batch: ""
    # Explicitly set a reference batch (string) or leave empty.
    # Conflicts in ordering/reference will halt execution.
```

---

## FeatureCloud App states

The app has the following states:

<p align="center">
   <img src="https://github.com/Freddsle/fedRBE/blob/main/figures/states.png?raw=true" alt="fedRBE app states" width="60%">
</p>

---
## Additional resources
- **FeatureCloud Docs**: [featurecloud.ai](https://featurecloud.ai/)
- **SMPC & Privacy Docs**: [Privacy-preserving techniques](https://featurecloud.ai/assets/developer_documentation/privacy_preserving_techniques.html#smpc-secure-multiparty-computation)
- **GitHub Repo**: [fedRBE](https://github.com/Freddsle/fedRBE)

## License

This project is licensed under the [Apache License 2.0](LICENSE).



## How to cite

If you use `fedRBE` in your research, please cite our [ArXiv preprint](https://arxiv.org/abs/2412.05894):

 > Burankova, Y., Klemm, J., Lohmann, J.J., Taheri, A., Probul, N., Baumbach, J. and Zolotareva, O., 2024. FedRBE--a decentralized privacy-preserving federated batch effect correction tool for omics data based on limma. arXiv preprint arXiv:2412.05894.

   ```bibtex
   @misc{burankova2024fedrbedecentralizedprivacypreserving,
         title={FedRBE -- a decentralized privacy-preserving federated batch effect correction tool for omics data based on limma}, 
         author={Yuliya Burankova and Julian Klemm and Jens J. G. Lohmann and Ahmad Taheri and Niklas Probul and Jan Baumbach and Olga Zolotareva},
         year={2024},
         eprint={2412.05894},
         archivePrefix={arXiv},
         primaryClass={q-bio.QM},
         url={https://arxiv.org/abs/2412.05894}, 
   }
   ```
