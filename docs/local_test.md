
<table>
  <tr>
    <td><a href="https://freddsle.github.io/fedRBE/docs/how_to_guide.html"><img src="https://img.shields.io/badge/HowTo_Guide-Click_Here!-007EC6?style=for-the-badge" alt="HowTo Guide"></a></td>
    <td><a href="https://freddsle.github.io/fedRBE/"><img src="https://img.shields.io/badge/Documentation-Click_Here!-007EC6?style=for-the-badge" alt="Documentation"></a></td>
    <td><a href="https://github.com/Freddsle/fedRBE/"><img src="https://img.shields.io/badge/GitHub-Click_Here!-007EC6?style=for-the-badge" alt="GitHub"></a></td>
    <td><a href="https://featurecloud.ai/app/fedrbe"><img src="https://img.shields.io/badge/FeatureCloud_App-Click_Here!-007EC6?style=for-the-badge" alt="FeatureCloud App"></a></td>
  </tr>
</table>

# Local test simulation

Test `fedRBE` locally with the FeatureCloud CLI. See the
[Documentation](https://freddsle.github.io/fedRBE/) for other usage modes.

## Prerequisites and setup

Install:
1. **Docker**: [Installation Instructions](https://www.docker.com/get-started)
1. **Python 3.8+**: [Installation Instructions](https://www.python.org/)
1. **FeatureCloud CLI**:
   ```bash
   pip install featurecloud
   ```
1. **App Image** (either pull or build locally — see [Installation](https://freddsle.github.io/fedRBE/#installation)).

For Windows users, git must also be installed and added to PATH. We recommend using [WSL](https://docs.docker.com/desktop/features/wsl/).


## Usage

1. **Clone the repository and enter it**:
   ```bash
   git clone https://github.com/Freddsle/fedRBE.git
   cd fedRBE
   ```
   Some sample data uses [Git LFS](https://git-lfs.com/); install it and run
   `git lfs pull` after cloning to fetch the full evaluation datasets.

2. **Start the FeatureCloud Controller with the correct input folder**:
   ```bash
   featurecloud controller start --data-dir=./evaluation_data/simulated/mild_imbalanced/before/
   ```

3. **Run a Sample Experiment**:
   ```bash
   featurecloud test start --app-image=featurecloud.ai/bcorrect:latest --client-dirs=lab1,lab2,lab3
   ```
   Alternatively, use the [frontend](https://featurecloud.ai/development/test/new)
   with three clients (`lab1`, `lab2`, `lab3`) and image
   `featurecloud.ai/bcorrect:latest`. Monitor logs and collect client results in
   the [browser](https://featurecloud.ai/development/test).

For multiple machines, follow the [How To Guide](https://freddsle.github.io/fedRBE/docs/how_to_guide.html).
