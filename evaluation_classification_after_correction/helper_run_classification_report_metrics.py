from typing import List
from pathlib import Path

from fc_fed_forest_simple_app.fedForestSimple import main
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import run_simulation_native

import pandas as pd

class ClassificationExperiment:
    name: str # The name to report that's associated with this experiment
    input_folders: List[str] # The client folders containing the config.yml and data file
    output_folders: List[str] # where to write the final metrics (and predictions)

    def __init__(self, name: str, input_folders: List[str], output_folders: List[str]):
        self.name = name
        self.input_folders = input_folders
        self.output_folders = output_folders

    def run_experiment(self):
        run_simulation_native(clientpaths=self.input_folders,
                              outputfolders=self.output_folders,
                              generic_dir=None,
                              fed_learning_main_function=main)

        # Collect and print metrics from coordinator only (first client)
        print(f"\n{'='*60}")
        print(f"Metrics for {self.name}")
        print(f"{'='*60}")

        # Only check coordinator's output folder (first one)
        output_path = Path(self.output_folders[0])
        metrics_file = output_path / "metrics.csv"

        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            print(f"\nCoordinator ({output_path.name}):")
            for _, row in metrics_df.iterrows():
                if pd.notna(row['score']):
                    print(f"  {row['class']} - {row['metric']}: {row['score']:.4f}")
        else:
            print(f"\nWarning: No metrics file found at {metrics_file}")
