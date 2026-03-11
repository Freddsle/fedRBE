
<table>
  <tr>
    <td><a href="https://freddsle.github.io/fedRBE/docs/how_to_guide.html"><img src="https://img.shields.io/badge/HowTo_Guide-Click_Here!-007EC6?style=for-the-badge" alt="HowTo Guide"></a></td>
    <td><a href="https://freddsle.github.io/fedRBE/"><img src="https://img.shields.io/badge/Documentation-Click_Here!-007EC6?style=for-the-badge" alt="Documentation"></a></td>
    <td><a href="https://github.com/Freddsle/fedRBE/"><img src="https://img.shields.io/badge/GitHub-Click_Here!-007EC6?style=for-the-badge" alt="GitHub"></a></td>
    <td><a href="https://featurecloud.ai/app/fedrbe"><img src="https://img.shields.io/badge/FeatureCloud_App-Click_Here!-007EC6?style=for-the-badge" alt="FeatureCloud App"></a></td>
  </tr>
</table>

# Local test simulation

This guide is designed for people who want to test the `fedRBE` tool locally on their machine via command line.

For more detailed information and advanced usage, please refer to the [Documentation](https://freddsle.github.io/fedRBE/).

## Prerequisites and setup

Ensure you have the following installed before starting:
1. **Docker**: [Installation Instructions](https://www.docker.com/get-started)
1. **Python 3.8+**: [Installation Instructions](https://www.python.org/)
1. **FeatureCloud CLI**:
   ```bash
   pip install featurecloud
   ```
1. **App Image** (either pull or build locally — see [Installation](https://freddsle.github.io/fedRBE/#installation)).

For Windows users, git must also be installed and added to PATH. We recommend using [WSL](https://docs.docker.com/desktop/features/wsl/).


## Usage

Run simulations locally to understand `fedRBE`'s behavior:

1. **Ensure the full repository including sample data is cloned and the current working directory**:
   ```bash
   git clone https://github.com/Freddsle/fedRBE.git
   cd fedRBE
   ```
   > **Note**: Some of the sample data used in this guide is tracked by Git LFS. If you need the full evaluation datasets, install [Git LFS](https://git-lfs.com/) and run `git lfs pull` after cloning.

2. **Start the FeatureCloud Controller with the correct input folder**:
   ```bash
   featurecloud controller start --data-dir=./evaluation_data/simulated/mild_imbalanced/before/
   ```

3. **Run a Sample Experiment**:
   ```bash
   featurecloud test start --app-image=featurecloud.ai/bcorrect:latest --client-dirs=lab1,lab2,lab3
   ```
   Alternatively, you can start the experiment from the **[frontend](https://featurecloud.ai/development/test/new)**.  
   Select 3 clients, add lab1, lab2, lab3 respecitvely for the 3 clients to their path. 
   
   Use `featurecloud.ai/bcorrect:latest` as the app image.

   You can monitor the logs of the batch effect correction run and receive the final results of the
   different clients on the [frontend in your browser](https://featurecloud.ai/development/test).
   
_For a step-by-step detailed instructions on how to start collaboration using multiple machines, refer to the [How To Guide](https://freddsle.github.io/fedRBE/docs/how_to_guide.html)._
