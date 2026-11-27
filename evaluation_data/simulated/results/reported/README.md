# Reported simulation figures and tables

| Item | Figure or table | Source or plotted values |
| --- | --- | --- |
| SR1 | [Recovery table](sr1_recovery.tsv), [differential expression](../../../../figures/suppl/sr1_confounding_correction.png) | [Run summary](../../robustness/summary.tsv), [run metrics](../confounding_downstream/reference_metrics.tsv.gz), [plotted means](plotted_values_three_client.tsv) |
| S4 | [FedRBE–limma agreement](s4_agreement.tsv) | [Three-client comparisons](../../robustness/equivalence.tsv), [five-client metrics](../missingness_overlap/informative_metrics.tsv.gz) |
| SR2 | [Slopes by missing fraction](sr2_missing_fraction.tsv), [learning under confounding](../../../../figures/suppl/sr2_confounding_learning.png) | [Per-dataset slopes](../missing_fraction_slopes.tsv), [common-cell slopes](../missingness_overlap/informative_common_cell_slopes.tsv), [plotted values](sr2_plotted_values.tsv) |
| SR3–SR4 tables | [Between-lab error and condition contrasts](sr3_sr4_correction.tsv) | [Five-client scores](../missingness_overlap/README.md#correction-scores-on-shared-measurements) |
| SR3 | [Feature retention](../../../../figures/suppl/sr3_five_client.png) | [Feature counts](sr3_feature_counts.tsv), [retention by lab count](sr3_retention_by_labs.tsv), [plotted means](sr3_plotted_values.tsv) |
| SR4 | [Learning by missing-value route](../../../../figures/suppl/sr4_five_client_learning.png) | [Plotted values](sr4_plotted_values.tsv) |
| SR5–SR6, all-sample d-ComBat | [LOCO](../../../../figures/suppl/sr5_dcombat_learning.png), [within-lab 70:30](../../../../figures/suppl/sr6_dcombat_learning_70_30.png) | [LOCO](sr5_plotted_values.tsv), [70:30](sr6_plotted_values.tsv) |
| Training-only d-ComBat | [LOCO](../../../../figures/candidate/dcombat_learning_training_only.png), [within-lab 70:30](../../../../figures/candidate/dcombat_learning_70_30_training_only.png) | [LOCO](dcombat_training_only_plotted_values.tsv), [70:30](dcombat_70_30_training_only_plotted_values.tsv) |
| SM1–SM2 | [Three-client scenarios](../../../../figures/suppl/figure_1_scenarios.png), [five-client panels and missingness](../../../../figures/suppl/figure_2_five_client.png) | Toy data |

The tables give unrounded values. A blank dataset count means the comparison is
inapplicable. A blank value with zero datasets means all slopes are undefined.

Random forest (RF) plots average MCC over five seeds. SR2 shows leave-one-lab-out
(LOCO) scores by held-out lab. The other RF panels average across labs. SR2 and
SR4 fit RF correction on training samples. SR2 k-means and SR5–SR6 use all samples.

## Rebuild the results

Run from the repository root with Poppler (`pdftoppm`) installed.

```bash
pixi run --frozen python evaluation_data/simulated/reporting/build_report.py
```

PNGs go to `figures/suppl/` and `figures/candidate/`. TSVs go here. To rebuild
only the tables, pass an output directory to the [table builder](../../reporting/table_values.py).
For regenerating experiment inputs, see the [simulation protocol](../../README.md).

## Inputs to the design and interpretation

The five-client design targets a ratio of 10 for the SD of feature means divided
by the median within-feature SD. Calibration uses features with at least ten log
intensities per lab. [E. coli calibration](ecoli_spread.tsv)

To compare noise and correction by lab, join `root` to `run` and match the lab.
[Lab noise scales](noise_scales.tsv) · [LOCO dataset means](../missingness_overlap/informative_dcombat_generated_run_means.tsv)

```bash
pixi run --frozen python evaluation_data/simulated/reporting/ecoli_spread.py /path/to/ecoli_spread.tsv
pixi run --frozen Rscript evaluation_data/simulated/reporting/noise_scales.R \
  /path/to/informative_missingness_campaign/runs /path/to/noise_scales.tsv
```
