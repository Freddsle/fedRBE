
# Federated Limma Remove Batch Effect (fedRBE)

[![License](https://img.shields.io/github/license/Freddsle/removeBatch)](LICENSE)
[![ArXiv](https://img.shields.io/badge/ArXiv-2412.05894-B31B1B)](https://arxiv.org/abs/2412.05894)

---

## Table of Contents
- [Federated Limma Remove Batch Effect (fedRBE)](#federated-limma-remove-batch-effect-fedrbe)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Architecture Overview](#architecture-overview)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Get the `fedRBE` App](#get-the-fedrbe-app)
  - [Usage](#usage)
    - [Quick Start](#quick-start)
  - [Reproducing the Paper](#reproducing-the-paper)
    - [Steps to Reproduce](#steps-to-reproduce)
  - [Configuration](#configuration)
    - [Example `config.yml`](#example-configyml)
  - [Examples](#examples)
    - [Single-Machine Simulation](#single-machine-simulation)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)
  - [Contact Information](#contact-information)


---

The **Federated Limma Remove Batch Effect (fedRBE)** offers a federated implementation of the limma `removeBatchEffect` algorithm. Implemented within the [FeatureCloud](https://featurecloud.ai/) platform, `fedRBE` enables batch effect correction in a privacy-preserving manner, ensuring that raw data remains decentralized.

This repository serves two main purposes:
1. **fedRBE Implementation**: Located in the `batchcorrection` subfolder, providing the federated batch effect removal tool.
2. **Reproducibility**: Contains code and scripts to reproduce the analyses presented in our [ArXiv preprint](https://arxiv.org/abs/2412.05894).

For usage instructions and how-to guides, refer to the [How To Guide](./how_to.md).
For more detailed information on the `fedRBE` implementation and configuration, see the [README](./batchcorrection/README.md).


---

## Features

- **Federated Learning**: Collaborate across multiple clients without sharing raw data, ensuring data privacy.
- **Batch Effect Removal**: Effectively removes non-biological variations using limma’s `removeBatchEffect` in a federated setting.
- **Flexible Input Formats**: Supports various data formats.
- **Secure Computation**: Utilizes Secure Multiparty Computation (SMPC) for privacy-preserving data aggregation.
- **Easy Integration**: Integrates with the FeatureCloud platform for streamlined workflow management.

<p align="center">
   <img src="./figures/readme1.png" alt="fedRBE app states" width="30%">
</p>

---

## Architecture Overview

`fedRBE` operates within the FeatureCloud ecosystem. The workflow involves a coordinator managing the project and multiple clients performing batch effect correction locally. Data remains with each client, and only summary statistics are shared, ensuring data privacy throughout the process.

<p align="center">
   <img src="./figures/readme2.png" alt="fedRBE app states" width="30%">
</p>

_For a detailed workflow, see the [How To Guide](./how_to.md#understanding-the-workflow)._

---

## Installation

### Prerequisites

Before installing `fedRBE`, ensure you have the following installed:
1. **Docker**: [Installation Instructions](https://www.docker.com/get-started)
2. **FeatureCloud CLI**:
   ```bash
   pip install featurecloud
   featurecloud controller start
   ```

### Clone the Repository

```bash
git clone https://github.com/Freddsle/removeBatch.git
cd removeBatch
```

This will clone the repository to your local machine with example files and simulation scripts.

### Get the `fedRBE` App

pull the pre-built image:

```bash
featurecloud app download featurecloud.ai/bcorrect
# Or directly via Docker
docker pull featurecloud.ai/bcorrect:latest
```

_**Note**: 
Alternatively, If you are using a non-linux/amd64 architecture (e.g., Mac M-series), you may need to build the image locally as shown below._

Navigate to the `batchcorrection` directory and build the Docker image:

```bash
cd batchcorrection
docker build . -t featurecloud.ai/bcorrect:latest
```

---

## Usage

### Quick Start

Run simulations locally to understand `fedRBE`'s behavior:

1. **Start the FeatureCloud Controller**:
   ```bash
   featurecloud controller start
   ```

2. **Build or Pull the `fedRBE` App** as per the [Installation](#installation) instructions.

3. **Run a Sample Experiment**:
   ```bash
   python3 ./evaluation_utils/run_sample_experiment.py
   ```

_For a step-by-step detailed instructions on how to start collaboration using multiple machines, refer to the [How To Guide](./how_to.md)._

---

## Reproducing the Paper

This repository includes all necessary code and data to reproduce the analyses presented in our [ArXiv preprint](https://arxiv.org/abs/2412.05894).

### Steps to Reproduce

1. **Ensure Prerequisites are Met**:
   - Docker installed
   - FeatureCloud CLI installed and running (`featurecloud controller start`)
   - `fedRBE` app built or pulled as per the [Installation](#installation) section

2. **Run the Federated Batch Effect Removal**:
   ```bash
   python3 ./evaluation_utils/get_federated_corrected_data.py
   ```

3. **Compare with Centralized Correction**:
   ```bash
   python3 ./evaluation_utils/analyse_fedvscentral.py
   ```

---

## Configuration

`fedRBE` is highly configurable via the `config.yml` file. This file controls data formats, normalization methods, and other essential parameters.

### Example `config.yml`

```yaml
flimmaBatchCorrection:
  data_filename: "expression_data_client1.csv"
  expression_file_flag: False
  index_col: "GeneIDs"
  covariates: ["Pyr"]
  separator: ","
  design_separator: ","
  normalizationMethod: "log2(x+1)"
  smpc: True
  min_samples: 2
  position: 1
  reference_batch: ""
```

_For a comprehensive list of configuration options, refer to the [Configuration Section](./batchcorrection/README.md#config) in the batchcorrection README._

---

## Examples

### Single-Machine Simulation

To simulate a federated workflow on a single machine using provided sample data:

1. **Run the Sample Experiment**:
   ```bash
   python3 ./evaluation_utils/run_sample_experiment.py
   ```

2. **Review Results**:
   - Batch-corrected data: `only_batch_corrected_data.csv`
   - Report: `report.txt`

---

## Troubleshooting

Encountering issues? Here are some common problems and their solutions:

- **Missing Files**: Ensure `config.yml` and data files are in the correct directory.
- **Incorrect Format**: Verify `expression_file_flag` and `index_col` settings in `config.yml`.
- **No Output Produced**: Check `report.txt` and logs for error messages.

_For detailed troubleshooting tips, refer to the [How To Guide](./how_to.md#troubleshooting-tips)._


## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Contact Information

For questions, issues, or support, please open an issue on the [GitHub repository](https://github.com/Freddsle/removeBatch).

