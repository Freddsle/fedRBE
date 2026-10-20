# Historical ovarian classification export

`prepare_data.py` was moved here unchanged on 16 October 2026
from `evaluation_data/ovarian_cancer/prepare_data.py`. It converted expression
matrices and HGSC targets to per-study `data.csv` files. Its original working
directory was `evaluation_data/ovarian_cancer/`; it was not rerun during archiving.

## Recovering the historical source

The source is preserved at its original path in public commit `a7fc7f9641d79d29882224dfe116d03b83654dd0`.
From the repository root, inspect the source with:

```bash
git show a7fc7f9641d79d29882224dfe116d03b83654dd0:evaluation_data/ovarian_cancer/prepare_data.py
```

To recover a file, redirect that output to an unused local path. The preservation
commit must be available locally; shallow clones may lack it.

## Why the script is no longer required

The corrected-data loop expects numbered files such as
`after/individual_results/only_batch_corrected_data_0.csv`, silently skipping them
when absent. Current results use study-named directories, so that loop is stale.

Current classification uses [before/datainfo.json](../before/datainfo.json) and
[after/label_aware/datainfo.json](../after/label_aware/datainfo.json), with study-level expression/design
inputs and corrected `full_corrected_data.csv` files. The
[DataInfo loader](../../../evaluation_classification_after_correction/helper/helper_run_classification_report_metrics.py)
performs alignment, transposition, target joining, and temporary input creation.
Clustering reads matrices and designs through
[datasets.yaml](../../../evaluation_utils/datasets.yaml).
Follow the [current dataset guide](../README.md); this script is retained only
as historical source, not as a required preparation step.
