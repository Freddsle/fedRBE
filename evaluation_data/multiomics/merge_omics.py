"""
Helper script to merge multiple omics folders (Metabolomics, Proteomics, Transcriptomics)
into one merged folder that retains the same cohort/client subfolder structure.

Expected layout (relative to this file):
    <before|after>/
        Metabolomics/   datainfo.json + <cohort>/ folders
        Proteomics/     datainfo.json + <cohort>/ folders
        Transcriptomics/datainfo.json + <cohort>/ folders

Produces:
    merged_omics/
        <before|after>/
            <cohort>/
                merged_data.tsv      - features x samples (all omics + meta columns)
            datainfo.json            - ready to be consumed by DataInfo
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level file helpers (mirror DataInfo._load_file logic)
# ---------------------------------------------------------------------------

def _load_file(filepath: Path, separator: str, rotation: str,
               samplename_column: Optional[str],
               featurename_column: Optional[str]) -> pd.DataFrame:
    """
    Load a TSV/CSV file and return it in **samples × features** orientation.
    Mirrors DataInfo._load_file so we stay consistent with the rest of the repo.
    """
    read_kwargs: dict = {"sep": separator}
    if rotation == "features x samples":
        read_kwargs["index_col"] = featurename_column if featurename_column else 0
    elif featurename_column:
        read_kwargs["index_col"] = featurename_column

    # Guard against accidental Git-LFS pointers
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        first_line = fh.readline()
    if first_line.startswith("version https://git-lfs"):
        raise ValueError(
            f"File '{filepath.name}' is a Git LFS pointer – did you forget `git lfs pull`?"
        )

    df = pd.read_csv(filepath, **read_kwargs)

    if rotation == "features x samples":
        df = df.T                          # → samples × features
    else:
        if samplename_column:
            df = df.set_index(samplename_column)

    return df


def _load_cohort_data_from_spec(
    omics_data_dir: Path,
    cohort: dict,
    datafile_spec: dict,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load data + optional design file for a single cohort.

    Returns
    -------
    data_df   : samples × features  (index = sample names)
    design_df : samples × metadata  (index = sample names) or None
    """
    cohort_folder = omics_data_dir / cohort["folder"]

    # ---- data file ----
    data_df = _load_file(
        filepath=cohort_folder / datafile_spec["filename"],
        separator=datafile_spec["separator"],
        rotation=datafile_spec["rotation"],
        samplename_column=datafile_spec.get("samplename_column"),
        featurename_column=datafile_spec.get("featurename_column"),
    )

    # ---- design file (optional) ----
    design_df: Optional[pd.DataFrame] = None
    if cohort.get("designfile"):
        ds = cohort["designfile"]
        design_df = _load_file(
            filepath=cohort_folder / ds["filename"],
            separator=ds["separator"],
            rotation=ds["rotation"],
            samplename_column=ds.get("samplename_column"),
            featurename_column=ds.get("featurename_column"),
        )

    return data_df, design_df


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def _merge_data_and_design(
    data_df: pd.DataFrame,
    design_df: Optional[pd.DataFrame],
    prediction_targets: list[str],
    covariates: Optional[list[str]],
    cohort_name: str,
    omics_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align data + design for one cohort and return them separately.

    Raises if the sample sets do not match exactly (strict mode).

    Returns
    -------
    features_df : samples × omics-features only  (no meta columns)
    design_out  : samples × (prediction_targets + covariates)
    """
    if design_df is not None:
        only_in_data = set(data_df.index) - set(design_df.index)
        only_in_design = set(design_df.index) - set(data_df.index)

        if only_in_data or only_in_design:
            raise ValueError(
                f"Sample mismatch between data and design file for cohort "
                f"'{cohort_name}' in {omics_name}.\n"
                f"  Only in data file   ({len(only_in_data)}): {sorted(only_in_data)}\n"
                f"  Only in design file ({len(only_in_design)}): {sorted(only_in_design)}"
            )

        design_out = design_df.loc[data_df.index]
        features_df = data_df.drop(
            columns=[c for c in prediction_targets if c in data_df.columns],
            errors="ignore",
        )
    else:
        # prediction targets must live inside the data file
        meta_cols = [c for c in (prediction_targets or []) + (covariates or []) if c in data_df.columns]
        design_out = data_df[meta_cols].copy()
        features_df = data_df.drop(columns=meta_cols, errors="ignore")

    # Drop covariates from the feature matrix (they should not be there, but be safe)
    if covariates:
        features_df = features_df.drop(
            columns=[c for c in covariates if c in features_df.columns],
            errors="ignore",
        )

    return features_df, design_out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def merge_omics_folders(script_folder: Optional[Path] = None) -> None:
    """
    Merge Metabolomics / Proteomics / Transcriptomics into merged_omics/.

    Parameters
    ----------
    script_folder : root directory that contains before/, after/ sub-dirs
                    and the three omics dirs inside each.
                    Defaults to the directory that contains *this* file.
    """
    if script_folder is None:
        script_folder = Path(__file__).parent

    data_folders = [
        script_folder / "before",
        script_folder / "after",
    ]
    omics_names = ["Metabolomics", "Proteomics", "Transcriptomics"]

    for data_folder in data_folders:
        print(f"\n{'='*60}")
        print(f"Processing data folder: {data_folder.name}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------ #
        # 1. Load all omics data keyed by (cohort_name, omics_name)           #
        # ------------------------------------------------------------------ #
        # cohort2omic2features[cohort_name][omics_name] = features DataFrame
        # We also store one design DataFrame per cohort (they must be identical
        # across omics for the same cohort).
        cohort2omic2features: dict[str, dict[str, pd.DataFrame]] = {}
        cohort2design: dict[str, pd.DataFrame] = {}
        reference_cohort_names: Optional[list[str]] = None
        global_prediction_targets: Optional[list[str]] = None
        global_covariates: Optional[list[str]] = None

        for omics_name in omics_names:
            omics_data_dir = data_folder / omics_name
            datainfo_path = omics_data_dir / "datainfo.json"

            if not datainfo_path.exists():
                raise FileNotFoundError(
                    f"Expected datainfo.json at {datainfo_path}"
                )

            with open(datainfo_path, "r", encoding="utf-8") as fh:
                datainfo = json.load(fh)

            prediction_targets: list[str] = datainfo["prediction_targets"]
            covariates: Optional[list[str]] = datainfo.get("covariates") or None
            cohorts: list[dict] = datainfo["cohorts"]
            datafile_spec: dict = datainfo["datafile"]

            # --- Validate / accumulate global meta fields ---
            if global_prediction_targets is None:
                global_prediction_targets = prediction_targets
            elif set(global_prediction_targets) != set(prediction_targets):
                raise ValueError(
                    f"prediction_targets in {omics_name} do not match those in "
                    f"{omics_names[0]}.\n"
                    f"  {omics_names[0]}: {sorted(global_prediction_targets)}\n"
                    f"  {omics_name}:    {sorted(prediction_targets)}"
                )

            # Union covariates across omics (they may legitimately differ)
            if covariates:
                if global_covariates is None:
                    global_covariates = list(covariates)
                else:
                    new_covs = [c for c in covariates if c not in global_covariates]
                    if new_covs:
                        print(f"    Adding covariates from {omics_name}: {new_covs}")
                    global_covariates = global_covariates + new_covs

            cohort_names = [c["name"] for c in cohorts]

            # Validate that cohort names are consistent across omics
            if reference_cohort_names is None:
                reference_cohort_names = cohort_names
            else:
                if set(reference_cohort_names) != set(cohort_names):
                    raise ValueError(
                        f"Cohort names in {omics_name} do not match the reference "
                        f"({omics_names[0]}). "
                        f"Reference: {sorted(reference_cohort_names)}, "
                        f"Got: {sorted(cohort_names)}"
                    )

            print(f"\n  Omics: {omics_name}")
            for cohort in cohorts:
                cohort_name = cohort["name"]
                print(f"    Loading cohort '{cohort_name}' ...", end=" ")

                features_df, design_df = _load_cohort_data_from_spec(
                    omics_data_dir, cohort, datafile_spec
                )
                features_df, design_out = _merge_data_and_design(
                    features_df, design_df, prediction_targets, covariates,
                    cohort_name=cohort_name, omics_name=omics_name,
                )

                # Prefix feature column names to avoid clashes across omics
                features_df = features_df.add_prefix(f"{omics_name}__")

                print(
                    f"{features_df.shape[0]} samples × {features_df.shape[1]} features"
                )

                cohort2omic2features.setdefault(cohort_name, {})[omics_name] = features_df

                # Store design / validate it matches exactly across omics
                if cohort_name not in cohort2design:
                    cohort2design[cohort_name] = design_out
                else:
                    existing = cohort2design[cohort_name]
                    only_in_existing = set(existing.index) - set(design_out.index)
                    only_in_new = set(design_out.index) - set(existing.index)
                    if only_in_existing or only_in_new:
                        raise ValueError(
                            f"Design sample mismatch for cohort '{cohort_name}' between "
                            f"{omics_names[0]} and {omics_name}.\n"
                            f"  Only in previous omics ({len(only_in_existing)}): {sorted(only_in_existing)}\n"
                            f"  Only in {omics_name}   ({len(only_in_new)}): {sorted(only_in_new)}"
                        )

        # ------------------------------------------------------------------ #
        # 2. Merge omics per cohort + report sample statistics                #
        # ------------------------------------------------------------------ #
        print(f"\n  --- Sample statistics for {data_folder.name} ---")

        merged_output_base = script_folder / "merged_omics" / data_folder.name
        merged_output_base.mkdir(parents=True, exist_ok=True)

        merged_cohort_entries: list[dict] = []

        for cohort_name in reference_cohort_names:  # type: ignore[union-attr]
            omic2df = cohort2omic2features[cohort_name]

            # --- Sample statistics & strict intersection check ---
            sample_sets = {om: set(df.index) for om, df in omic2df.items()}
            all_samples = set.union(*sample_sets.values())
            intersection_samples = set.intersection(*sample_sets.values())

            print(f"\n  Cohort: {cohort_name}")
            for om, s in sample_sets.items():
                print(f"    {om}: {len(s)} samples")
            print(f"    Union (any omics):         {len(all_samples)} samples")
            print(f"    Intersection (all omics):  {len(intersection_samples)} samples")

            missing = all_samples - intersection_samples
            if missing:
                raise ValueError(
                    f"Sample set mismatch for cohort '{cohort_name}': "
                    f"{len(missing)} sample(s) are not present in all omics: "
                    f"{sorted(missing)}"
                )

            common_samples = sorted(intersection_samples)

            # --- Concatenate omics features (samples × all_omics_features) ---
            aligned_feature_dfs = [
                omic2df[om].loc[common_samples] for om in omics_names
            ]
            merged_features = pd.concat(aligned_feature_dfs, axis=1)

            # --- Append meta columns (prediction targets + covariates) ---
            design_df = cohort2design[cohort_name].loc[common_samples]
            merged_full = pd.concat([merged_features, design_df], axis=1)

            # ------------------------------------------------------------ #
            # 3. Write single output file                                    #
            # ------------------------------------------------------------ #
            out_cohort_dir = merged_output_base / cohort_name
            out_cohort_dir.mkdir(parents=True, exist_ok=True)

            # features × samples orientation (repo convention), index = feature/meta name
            data_out_path = out_cohort_dir / "merged_data.tsv"
            merged_full.T.to_csv(data_out_path, sep="\t", index=True)
            print(f"    Written: {data_out_path.relative_to(script_folder)}")

            # Collect cohort entry for datainfo.json – no separate designfile needed
            merged_cohort_entries.append({
                "name": cohort_name,
                "folder": cohort_name,
                "designfile": None,
            })

        # ------------------------------------------------------------------ #
        # 4. Write top-level datainfo.json for the merged omics folder        #
        # ------------------------------------------------------------------ #
        # The merged file is features × samples with the index column named
        # "rowname".  Meta columns (prediction targets, covariates) are rows
        # in the file just like any feature – DataInfo will pull them out by
        # name after loading.  No separate designfile is needed.
        merged_datainfo = {
            "data_name": f"Merged multiomics ({', '.join(omics_names)}) – {data_folder.name}",
            "covariates": global_covariates or [],
            "prediction_targets": global_prediction_targets,
            "datafile": {
                "filename": "merged_data.tsv",
                "separator": "\t",
                "rotation": "features x samples",
                "samplename_column": None,
                "featurename_column": "rowname",
            },
            "cohorts": merged_cohort_entries,
        }

        datainfo_out_path = merged_output_base / "datainfo.json"
        with open(datainfo_out_path, "w", encoding="utf-8") as fh:
            json.dump(merged_datainfo, fh, indent=2)
        print(f"\n  Written: {datainfo_out_path.relative_to(script_folder)}")

    print(f"\n{'='*60}")
    print("Done – merged omics written to: merged_omics/")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    merge_omics_folders()
