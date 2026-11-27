#!/usr/bin/env bash
# Reproduces the analysis of docs/reproduce.md step by step and stops at the
# first failing step. A questionnaire selects the expensive stages; the run
# continues in a detached screen session. Successful steps are recorded, so a
# later start can continue where a run stopped.
#
#   bash reproduce.sh --help      options
#   bash reproduce.sh --explain   what every question and step does, and why
#
# Layout of this file:
#   1. Settings
#   2. Questions: one define_question call per question
#   3. Steps: one define_step call per step, followed by its function step_<id>
#   4. The machinery: output checks, progress, questionnaire, planning, running

set -Eeuo pipefail
shopt -s inherit_errexit extglob

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
cd "$REPO_ROOT"

# ===========================================================================
# 1. Settings
# ===========================================================================
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == 1 ]]; then
    STATE_DIR="${REPRO_STATE_DIR:-${REPO_ROOT}/reproduce_runs/dry_run}"
else
    STATE_DIR="${REPRO_STATE_DIR:-${REPO_ROOT}/reproduce_runs}"
fi
PROGRESS="${STATE_DIR}/progress.tsv"
LOCK="${STATE_DIR}/.lock"
LAST_ANSWERS="${STATE_DIR}/last_answers.env"
PX=(pixi run --frozen)
MIN_DISK_GB=20
MIN_RAM_GB=16

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Collapses line breaks and indentation of multi-line texts into single spaces
one_line() {
    local text="${1//+([[:space:]])/ }"
    text="${text# }"
    echo "${text% }"
}

# define_question VAR --ask TEXT --explain TEXT [--kind yn|int|dir] [--default VALUE]
#                 [--if "VAR..."]
#   --kind     yn (default): y or n; int: positive integer; dir: Distributed-ComBat folder or empty
#   --default  used when no earlier answer exists
#   --if       ask only when all these answers are y (otherwise the earlier or default answer stays)
ANSWER_VARS=()
declare -A Q_ASK Q_EXPLAIN Q_KIND Q_DEFAULT Q_IF
define_question() {
    local var="$1"
    shift
    ANSWER_VARS+=("$var")
    Q_KIND[$var]=yn
    Q_DEFAULT[$var]=""
    Q_IF[$var]=""
    while (( $# )); do
        case "$1" in
            --ask) Q_ASK[$var]="$2" ;;
            --explain) Q_EXPLAIN[$var]="$(one_line "$2")" ;;
            --kind) Q_KIND[$var]="$2" ;;
            --default) Q_DEFAULT[$var]="$2" ;;
            --if) Q_IF[$var]="$2" ;;
            *) die "define_question ${var}: unknown field $1" ;;
        esac
        shift 2
    done
}

# define_step ID --title TEXT --explain TEXT [--if "VAR..."] [--rerun-when-changed "VAR..."]
#   --title               one line for plans and listings
#   --explain             what the step does and why; printed in the log when it starts
#   --if                  run only when all these answers are y
#   --rerun-when-changed  answers that change the step's result: a step recorded as done
#                         with other values of these answers counts as not done
# The step's commands are the function step_<ID>. Steps run in the order defined here.
STEP_IDS=()
declare -A STEP_TITLE STEP_EXPLAIN STEP_IF STEP_RERUN_WHEN
define_step() {
    local id="$1"
    shift
    STEP_IDS+=("$id")
    STEP_IF[$id]=""
    STEP_RERUN_WHEN[$id]=""
    while (( $# )); do
        case "$1" in
            --title) STEP_TITLE[$id]="$2" ;;
            --explain) STEP_EXPLAIN[$id]="$(one_line "$2")" ;;
            --if) STEP_IF[$id]="$2" ;;
            --rerun-when-changed) STEP_RERUN_WHEN[$id]="$2" ;;
            *) die "define_step ${id}: unknown field $1" ;;
        esac
        shift 2
    done
}

# ===========================================================================
# 2. Questions, asked in this order
#    LOW_RAM (less than MIN_RAM_GB) and HAVE_DCOMBAT (DCOMBAT_DIR set) are
#    derived from the machine and the answers, see set_derived
# ===========================================================================
define_question DO_OVARIAN_RAW --default y \
    --ask "Redo the raw ovarian cancer preprocessing (~30 min, downloads from GEO)?" \
    --explain "Downloads the ovarian cancer microarray cohorts from GEO again, normalizes
        the raw Affymetrix data with RMA and harmonizes the metadata (step 1d). Answer
        n to start from the committed harmonized data instead; the correction inputs
        (step 1e) are prepared either way."

define_question DO_SIM30 --default n \
    --ask "Regenerate all 30 simulated runs and their evaluation (several hours)?" \
    --explain "The committed simulated data contain run 1 of each scenario, which is
        enough for the federated-versus-central comparison. The 30-run statistics
        (steps 1a and 4b) need all runs. Answer n to skip both and keep the committed
        30-run results."

define_question DO_NATIVE_CHECK --default y \
    --ask "Check the fedRBE protocol without Docker first (minutes)?" \
    --explain "Runs every federated experiment in one process, without Docker and SMPC,
        and compares it with the central result (step 2a). It catches broken inputs or
        protocol errors in minutes, before the Docker runs that take hours."

define_question DO_FEDERATED --default y \
    --ask "Run the federated correction with Docker/FeatureCloud (hours to a day)?" \
    --explain "Builds the fedRBE app image and runs all federated experiments in the
        FeatureCloud test environment, one Docker container per client, with and
        without SMPC (steps 2b and 2c). Answer n to compare and evaluate the committed
        federated results instead."

define_question ALLOW_LOW_RAM --default n --if "DO_FEDERATED LOW_RAM" \
    --ask "This machine has less than ${MIN_RAM_GB} GB RAM. Run the federated correction anyway?" \
    --explain "The ovarian cancer experiments can need ${MIN_RAM_GB} GB RAM or more; with
        less, their containers may be killed. Answer n to stop in the preflight instead."

define_question DO_DIAGNOSTICS --default y \
    --ask "Draw the dataset diagnostics plots (step 4a)?" \
    --explain "PCA and distribution plots of each dataset after central correction, for
        a visual check that the batch effects are removed. No later step uses them."

define_question DO_CLASSIFICATION --default y \
    --ask "Run the random forest classification experiments (several hours)?" \
    --explain "Trains federated and centralized random forests on uncorrected and
        corrected data, with 70:30 splits and leave-one-cohort-out (steps 5a and 5b),
        for Figures 4, S7 and S8. Answer n to plot the committed results instead."

define_question N_JOBS --kind int --default auto --if DO_CLASSIFICATION \
    --ask "How many classification units should run in parallel?" \
    --explain "A unit is one dataset with one seed or held-out cohort. Each needs up to
        about 4 GB RAM when ovarian cancer runs. The results do not depend on this
        number. The default uses the CPU cores, limited by RAM / 4 GB."

define_question FORCE_RUN --default y --if DO_CLASSIFICATION \
    --ask "Recompute classification results that are already in the results file?" \
    --explain "The classification scripts skip rows that are already in the results file
        (--force-run disables this). Cached rows do not record the code or inputs that
        produced them, so recompute after any change."

define_question DO_KMEANS_RERUN --default n \
    --ask "Rerun the federated k-means clustering with Docker?" \
    --explain "The clustering step (6b) always recomputes the central k-means fits and
        reuses the saved federated assignments. Answer y to build the federated k-means
        app image (step 6a) and compute new federated fits."

define_question DO_CONTROLLED_SIM --default n \
    --ask "Run the controlled simulation study (days)?" \
    --explain "The supplementary simulation study of evaluation_data/simulated/README.md
        (steps 8a-8f): biological recovery under confounding, sample-size scaling,
        missingness and feature overlap, and downstream learning. It has its own saved
        results and is independent of the other steps."

define_question SCALING_FULL --default n --if DO_CONTROLLED_SIM \
    --ask "Run the full sample-size scaling, up to 50,004 samples (~60 GB RAM per run)?" \
    --explain "Answer n for a quick check with 600 samples and one run (step 8b)."

define_question DCOMBAT_DIR --kind dir --if DO_CONTROLLED_SIM \
    --ask "Folder of Distributed-ComBat (commit 45b5a25), empty to skip?" \
    --explain "The missingness learners (step 8f) compare fedRBE with d-ComBat, which is
        not part of this repository: give the folder with distributedCombat.R,
        neuroCombat.R and neuroCombat_helpers.R from github.com/andy1764/Distributed-ComBat.
        Empty skips step 8f."

# ===========================================================================
# 3. Steps
# ===========================================================================

# --- 0. Preflight ----------------------------------------------------------
define_step 0 \
    --title "Preflight: tools, Docker, Git LFS, pixi, fonts, disk/RAM, inputs" \
    --explain "Checks everything later steps need, so that a problem stops the run within
        minutes instead of hours in: the commands, the Docker daemon (when a Docker step
        runs), the Git LFS files, the locked pixi environment with its R notebook
        kernel, the Figure S11 font, disk space and RAM, and the files that steps of
        this run read but no step of this run writes. Runs at every start."

step_0() {
    check_tools
    check_environment
    check_resources
    check_earlier_inputs
}

check_tools() {
    need_cmd git git-lfs pixi flock awk sha256sum find df
    if in_plan 2b || in_plan 2c || in_plan 6a; then
        need_cmd docker
        local docker_info
        if ! docker_info="$(docker info 2>&1)"; then
            echo "The Docker daemon is not reachable:" >&2
            echo "$docker_info" >&2
            return 1
        fi
        echo "Docker daemon reachable"
    fi
}

check_environment() {
    # Large inputs and results are in Git LFS; a file left as a pointer would be
    # read as a broken table
    x git lfs install --local
    x git lfs pull
    local pointers
    pointers="$(git lfs ls-files | awk '$2 == "-"')"
    if [[ -n "$pointers" ]]; then
        echo "Git LFS files are still pointers after git lfs pull:" >&2
        echo "$pointers" >&2
        return 1
    fi
    echo "Git LFS files present"

    x pixi install --locked
    local kernels
    kernels="$("${PX[@]}" jupyter kernelspec list)"
    if ! grep -qE '^\s*ir\s' <<< "$kernels"; then
        echo "The R Jupyter kernel 'ir' is missing from the pixi environment:" >&2
        echo "$kernels" >&2
        return 1
    fi
    echo "Jupyter kernel 'ir' available"

    if in_plan 7c; then
        # Ask matplotlib itself: the conda fc-list does not see system fonts
        if ! "${PX[@]}" python -c "from matplotlib import font_manager as f; f.findfont('Liberation Sans', fallback_to_default=False)"; then
            echo "Figure S11 needs the Liberation Sans font (e.g. the fonts-liberation package)" >&2
            return 1
        fi
        echo "Liberation Sans available to matplotlib"
    fi
}

check_resources() {
    local disk ram
    disk="$(free_disk_gb)"
    ram="$(ram_gb)"
    echo "Free disk: ${disk} GB, RAM: ${ram} GB"
    if (( disk < MIN_DISK_GB )); then
        echo "At least ${MIN_DISK_GB} GB free disk space are needed" >&2
        return 1
    fi
    if in_plan 2c && [[ "$LOW_RAM" == y && "$ALLOW_LOW_RAM" != y ]]; then
        echo "The federated correction needs at least ${MIN_RAM_GB} GB RAM (ALLOW_LOW_RAM=n)" >&2
        return 1
    fi
}

# Inputs that steps of this run read but no step of this run writes: the
# committed files, or those of an earlier run
check_earlier_inputs() {
    if in_plan 3 && ! in_plan 1b; then
        require_listed exist central-outputs --simulated
    fi
    if in_plan 3 && ! in_plan 1h; then
        require_listed exist central-outputs --real
    fi
    if in_plan 3 && ! in_plan 2c; then
        require_listed exist federated-outputs --compared
    fi
    if { in_plan 5c || in_plan 7a; } && ! in_plan 5a && ! in_plan 5b; then
        require_exist evaluation_classification_after_correction/results/classification_metric_report.csv
    fi
    if in_plan 7a && ! in_plan 6b; then
        require_exist evaluation_clusterization_after_correction/real_datasets/{,no_label/}metrics_ari_multirun.tsv
    fi
    if in_plan 8f && ! in_plan 8e; then
        require_exist "${STATE_DIR}/controlled_sim/missingness_latest/runs"
    fi
    echo "Inputs from earlier runs present"

    # Steps 2a and 2c write config.yml into the tracked client folders and restore it
    # afterwards; a crashed run can leave a changed copy that the next run would use
    local configs
    configs="$(git status --porcelain -- 'evaluation_data/**/config.yml')"
    if [[ -n "$configs" ]]; then
        echo "WARNING: client config.yml files differ from Git; a crashed earlier run may have left them:"
        echo "$configs"
    fi
}

# --- 1. Inputs and central correction -------------------------------------
define_step 1a --if DO_SIM30 \
    --title "Simulated: generate 30 runs (00_data_simulation.ipynb, 1-2 h)" \
    --explain "Simulates 30 runs of the three simulated scenarios (balanced, mildly and
        strongly imbalanced; three labs each) with known batch and condition effects.
        Writes before/intermediate/ and all_metadata.tsv per scenario. Only the 30-run
        statistics (step 4b) need them; run 1 is committed."

step_1a() {
    run_nb evaluation_data/simulated/00_data_simulation.ipynb
    require_fresh evaluation_data/simulated/{balanced,mild_imbalanced,strong_imbalanced}/all_metadata.tsv
}

define_step 1b \
    --title "Simulated: inputs + central correction (01_data_prep_and_central_RBE.ipynb)" \
    --explain "Prepares the client inputs of the simulated scenarios (before/lab*/) and
        corrects the pooled data centrally with limma (after/intensities_R_corrected.tsv),
        the reference the federated result is compared with in step 3. With all 30 runs
        present, it also corrects every run (after/runs/)."

step_1b() {
    run_nb evaluation_data/simulated/01_data_prep_and_central_RBE.ipynb
    require_listed fresh central-outputs --simulated
}

define_step 1c \
    --title "E. coli: inputs (01_data_prep_and_central_RBE.ipynb)" \
    --explain "Filters the E. coli proteomics data of the five labs and writes the client
        inputs and the pooled central input (before/central_input.tsv) for the central
        (1h) and federated (2c) correction."

step_1c() {
    run_nb evaluation_data/ecoli/01_data_prep_and_central_RBE.ipynb
    require_fresh evaluation_data/ecoli/before/central_input.tsv
}

define_step 1d --if DO_OVARIAN_RAW \
    --title "Ovarian: setup-affymetrix + raw preprocessing (00_harmonize_meta_load_data.ipynb, ~30 min)" \
    --explain "Downloads the ovarian cancer microarray cohorts from GEO, normalizes the raw
        Affymetrix CEL files with RMA and harmonizes the metadata. setup-affymetrix first
        installs the array annotation packages and rebuilds preprocessCore without
        threading, which affy::rma needs in the pixi environment."

step_1d() {
    x "${PX[@]}" setup-affymetrix
    run_nb evaluation_data/ovarian_cancer/00_harmonize_meta_load_data.ipynb
}

define_step 1e \
    --title "Ovarian: inputs (02_prepare_correction_inputs.ipynb)" \
    --explain "Filters the harmonized ovarian cancer data and writes one client input per
        cohort and the pooled central input (before/central_input.zip) for the central
        (1h) and federated (2c) correction."

step_1e() {
    run_nb evaluation_data/ovarian_cancer/02_prepare_correction_inputs.ipynb
    require_fresh evaluation_data/ovarian_cancer/before/central_input.zip
}

define_step 1f \
    --title "ccRCC: prepare_ccRCC_data.py + 01_central_RBE.ipynb" \
    --explain "Starts from the tracked report and design files of the three ccRCC
        proteomics studies: aligns their sample order, encodes the condition, builds
        the union matrix and writes the client inputs and the pooled central input
        (before/central_input.tsv)."

step_1f() {
    x env -C evaluation_data/ccRCC_studies "${PX[@]}" python prepare_ccRCC_data.py
    run_nb evaluation_data/ccRCC_studies/01_central_RBE.ipynb
    require_fresh evaluation_data/ccRCC_studies/before/central_input.tsv
}

define_step 1g \
    --title "Quartet: notebooks 01, 02, 03" \
    --explain "Preprocesses the Quartet multiomics data (transcriptomics, proteomics,
        metabolomics) with an exploratory analysis (01), writes the client folders per
        modality (02) and the pooled central input per modality (03)."

step_1g() {
    local notebook
    for notebook in 01_preprocess_eda 02_prepare_RBE_inputs 03_central_RBE; do
        run_nb "evaluation_data/quartet_multiomics/${notebook}.ipynb"
    done
    require_fresh evaluation_data/quartet_multiomics/before/{Transcriptomics,Proteomics,Metabolomics}/central_input.tsv
}

define_step 1h \
    --title "Central limma correction, both arms (run_correction_experiments)" \
    --explain "Corrects the pooled data of the four real datasets centrally with limma
        removeBatchEffect, in both arms: label_aware (biological condition as covariate)
        and no_label (batches only). These are the references for the federated results
        (step 3)."

step_1h() {
    x "${PX[@]}" python -m evaluation_utils.run_correction_experiments correct \
        --arms label_aware no_label
    require_listed fresh central-outputs --real
}

# --- 2. Federated correction ----------------------------------------------
define_step 2a --if DO_NATIVE_CHECK \
    --title "Federated: native protocol check without Docker" \
    --explain "Runs the fedRBE protocol code of every federated experiment in one process,
        without Docker and SMPC, and compares each merged result with its central result.
        Fails on any difference, before the long Docker runs."

step_2a() {
    # Exits with 1 when any experiment differs from its central result
    x "${PX[@]}" python evaluation_utils/fedRBE_simulation_scrip_simdata.py --experiments
}

define_step 2b --if DO_FEDERATED \
    --title "Federated: build the fedRBE app image" \
    --explain "Builds the Docker image featurecloud.ai/bcorrect from ./batchcorrection, so
        the federated runs use the app of the current code."

step_2b() {
    x "${PX[@]}" featurecloud app build ./batchcorrection featurecloud.ai/bcorrect latest
}

define_step 2c --if DO_FEDERATED \
    --title "Federated: generate_fedrbe_corrected_datasets.py (hours to a day)" \
    --explain "Runs every federated experiment in the FeatureCloud test environment, one
        Docker container per client, with and without SMPC and in both arms. Merges the
        clients' corrected data (FedApp_corrected_data*.tsv) and saves the batch
        coefficients and the runtime and memory reports used by step 7b."

step_2c() {
    FC_CONTROLLER_USED=y
    # Exits nonzero when any experiment fails
    x "${PX[@]}" python ./generate_fedrbe_corrected_datasets.py
    stop_controller
    require_listed fresh federated-outputs
}

# --- 3. Federated versus central ------------------------------------------
define_step 3 \
    --title "Compare federated and central corrections" \
    --explain "Compares every federated result with its central result value by value
        (evaluation_data/fed_vc_cent_results.tsv): the paper's claim that fedRBE gives
        the same correction as central limma. Fails when a maximum difference exceeds
        1e-10."

step_3() {
    x "${PX[@]}" python ./analyse_fedvscentral.py
    require_fresh evaluation_data/fed_vc_cent_results.tsv
    # analyse_fedvscentral.py only prints failed comparisons; fail on them here
    x "${PX[@]}" python -m evaluation_utils.reproduce_checks check-fedvscentral
}

# --- 4. Diagnostics -------------------------------------------------------
define_step 4a --if DO_DIAGNOSTICS \
    --title "Dataset diagnostics (dataset_diagnostics.R)" \
    --explain "Draws PCA and distribution plots of each dataset after central correction,
        and the PCA plots of the simulated scenarios before and after fedRBE."

step_4a() {
    local dataset
    for dataset in simulated ecoli ovarian_cancer ccRCC_studies quartet_multiomics; do
        x "${PX[@]}" Rscript evaluation/dataset_diagnostics.R "$dataset"
    done
    require_fresh evaluation_data/ecoli/plots/data_plot_Rcorrected.png \
        evaluation_data/ovarian_cancer/before/plots/norm_vs_tumor_plot_Rcorrected_UNION.png \
        evaluation_data/ccRCC_studies/plots/data_plot_Rcorrected.png \
        evaluation_data/quartet_multiomics/plots/{Transcriptomics,Proteomics,Metabolomics}_limma_after_diagnostics.png \
        evaluation/eval_simulation/pca_plots/pca_{balanced,mild_imbalanced,strong_imbalanced}.png
}

define_step 4b --if DO_SIM30 \
    --title "Simulated 30-run differences + variance partitioning (hours)" \
    --explain "Recomputes the fedRBE-versus-limma differences of all 30 simulated runs per
        scenario and plots the variance partitioning before and after correction, to
        show that the agreement holds across repeated simulations."

step_4b() {
    x "${PX[@]}" Rscript evaluation/evaluation_simulated_30runs.R
    require_fresh evaluation/eval_simulation/lmpv_plots
}

# --- 5. Classification ----------------------------------------------------
# run_rf <script>: one classification experiment type with the questionnaire's options
run_rf() {
    local options=(--jobs "$N_JOBS")
    if [[ "$FORCE_RUN" == y ]]; then
        options+=(--force-run)
    fi
    x "${PX[@]}" python -u "evaluation_classification_after_correction/$1" "${options[@]}"
    # Without --force-run, cached rows may leave the report untouched
    if [[ "$FORCE_RUN" == y ]]; then
        require_fresh evaluation_classification_after_correction/results/classification_metric_report.csv
    fi
}

define_step 5a --if DO_CLASSIFICATION --rerun-when-changed FORCE_RUN \
    --title "Classification: 70:30 train/test (hours)" \
    --explain "For ten seeds, splits every cohort 70:30, fits the correction on the
        training samples only (fedRBE for the federated forest, limma for the
        centralized one) and scores both forests on the corrected test samples against
        the uncorrected baseline. Results for Figures 4c and S8."

step_5a() {
    run_rf run_classification_train_test_split.py
}

define_step 5b --if DO_CLASSIFICATION --rerun-when-changed FORCE_RUN \
    --title "Classification: leave-one-cohort-out (hours)" \
    --explain "Holds out each cohort once, corrects the other cohorts, aligns the held-out
        cohort to their corrected level without its labels and scores both forests on
        it, for ten forest seeds. Results for Figures 4b and S7."

step_5b() {
    run_rf run_classification_leave_one_cohort_out.py
}

define_step 5c \
    --title "Classification: metric boxplots" \
    --explain "Plots MCC and F1 per dataset and validation method from the classification
        results, for both correction arms."

step_5c() {
    x "${PX[@]}" python -u evaluation_classification_after_correction/analyse_classification_metric_report.py
    require_fresh evaluation_classification_after_correction/plots/{label_aware,label_free}
}

# --- 6. Clustering --------------------------------------------------------
define_step 6a --if DO_KMEANS_RERUN \
    --title "Clustering: build the federated k-means app image" \
    --explain "Builds the Docker image fc_kmeans_upd, needed only for new federated
        k-means fits."

step_6a() {
    x env -C evaluation_clusterization_after_correction/federated_kmeans_upd ./build.sh
}

define_step 6b --rerun-when-changed DO_KMEANS_RERUN \
    --title "Clustering: multiple_runs.py" \
    --explain "Clusters the uncorrected and corrected data of the four real datasets with
        central, local federated and coordinator k-means for ten seeds and scores them
        with the adjusted Rand index (metrics_ari_multirun.tsv). Results for Figure 4a.
        Reuses the saved federated assignments unless you chose to rerun them."

step_6b() {
    local options=()
    if [[ "$DO_KMEANS_RERUN" == y ]]; then
        options+=(--rerun-federated)
    fi
    x "${PX[@]}" python -u evaluation_clusterization_after_correction/real_datasets/multiple_runs.py \
        ${options[@]+"${options[@]}"}
    require_fresh evaluation_clusterization_after_correction/real_datasets/{,no_label/}metrics_ari_multirun.tsv
}

# --- 7. Figures -----------------------------------------------------------
define_step 7a \
    --title "Figures 3, 4, S5, S7, S8" \
    --explain "Draws Figure 3 (the four real datasets before and after correction, with
        the median variance fractions), Figure 4 with its statistical tests
        (clustering and classification results of steps 5 and 6), and Figures S5, S7
        and S8, for both correction arms. The arm the paper reports goes to
        figures/main/ or figures/suppl/, the other one to figures/candidate/."

step_7a() {
    local script
    # Unset the overrides, so every script draws both arms to the default paths
    for script in aggregate_figure3 aggregate_figure4 quartet_separated_modality_pca_fixed_effect_variance; do
        x env -u CORRECTION_ARM -u FIGURE4_ARI_TABLE -u FIGURE4_RF_REPORT -u FIGURE4_OUTPUT \
            "${PX[@]}" Rscript "evaluation/${script}.R"
    done
    require_fresh figures/main/{figure3_label_aware,figure4_no_label}.png \
        figures/suppl/{figureS5_label_aware,figureS7_no_label,figureS8_no_label}.png \
        figures/candidate/{figure3_no_label,figure4_label_aware,figureS5_no_label,figureS7_label_aware,figureS8_label_aware}.png \
        figures/candidate/figure4_{no_label,label_aware}_statistics.png
}

define_step 7b \
    --title "Performance figures (aggregate_performance.R)" \
    --explain "Plots runtime and memory of the federated runs per dataset, with and without
        SMPC, from the performance reports of the Docker runs (step 2c or committed)."

step_7b() {
    x env -u PERFORMANCE_DATA_DIR -u PERFORMANCE_RUNTIME_OUTPUT -u PERFORMANCE_MEMORY_OUTPUT \
        "${PX[@]}" Rscript evaluation/aggregate_performance.R
    require_fresh figures/suppl/performance_{runtime,memory}.png
}

define_step 7c \
    --title "Figure S11 (plot_feature_overlap.py)" \
    --explain "Counts the features each client observes in the prepared inputs and
        compares them with the features fedRBE corrected (Figure S11 and the overlap
        tables under figures/)."

step_7c() {
    x "${PX[@]}" python evaluation/plot_feature_overlap.py
    require_fresh figures/suppl/Figure_S11.png figures/feature_overlap_{summary,intersections}.tsv
}

# --- 8. Controlled simulation study ---------------------------------------
# The study's scripts refuse existing output folders, so every run writes to
# reproduce_runs/controlled_sim/<run>/
cs_dir() {
    mkdir -p "${STATE_DIR}/controlled_sim/${RUN_ID}"
    echo "${STATE_DIR}/controlled_sim/${RUN_ID}/$1"
}

define_step 8a --if DO_CONTROLLED_SIM \
    --title "Controlled simulation: robustness" \
    --explain "Nine three-client scenarios with ten runs each (class imbalance, lab-condition
        confounding, intensity-dependent batch effects): how well the correction
        recovers the true biological signal, and whether fedRBE equals limma."

step_8a() {
    local out
    out="$(cs_dir robustness)"
    x "${PX[@]}" Rscript evaluation_data/simulated/robustness_simulation.R --out "$out"
    require_fresh "$out"
}

define_step 8b --if DO_CONTROLLED_SIM --rerun-when-changed SCALING_FULL \
    --title "Controlled simulation: sample-size scaling" \
    --explain "Compares fedRBE and limma in both arms as the number of samples grows (600
        up to 50,004), or only at 600 samples for a quick check."

step_8b() {
    local out options=()
    out="$(cs_dir sample_size_scaling)"
    if [[ "$SCALING_FULL" != y ]]; then
        options+=(--samples 600 --runs 1)
    fi
    x "${PX[@]}" Rscript evaluation_data/simulated/sample_size_scaling.R \
        ${options[@]+"${options[@]}"} --out "$out"
    require_fresh "$out"
}

define_step 8c --if DO_CONTROLLED_SIM \
    --title "Controlled simulation: contrast slopes by missing fraction" \
    --explain "Compares the contrasts of masked and corrected data with the complete truth
        by missing fraction."

step_8c() {
    local out
    out="$(cs_dir missing_fraction_slopes.tsv)"
    x "${PX[@]}" Rscript evaluation_data/simulated/missing_fraction_slopes.R "$out"
    require_fresh "$out"
}

define_step 8d --if DO_CONTROLLED_SIM \
    --title "Controlled simulation: downstream utility under confounding (~1 day)" \
    --explain "Random forests (leave-one-lab-out and within-lab 70:30) and k-means on
        confounded simulated data: does correction help or hurt downstream learning
        when labs and conditions are confounded."

step_8d() {
    local rf kmeans common
    rf="$(cs_dir confounding_rf)"
    kmeans="$(cs_dir confounding_kmeans)"
    # shellcheck disable=SC2054  # the commas belong to the option values
    common=(--runs 1,2,3,4,5,6,7,8,9,10 --learner-seeds 101,211,321,431,541
            --reference evaluation_data/simulated/results/confounding_downstream/reference_metrics.tsv.gz)
    x "${PX[@]}" python evaluation_data/simulated/downstream_simulation.py \
        --rf-protocols loco within_lab_70_30 "${common[@]}" --out "$rf"
    x "${PX[@]}" python evaluation_data/simulated/downstream_simulation.py \
        --correction joint --tasks kmeans "${common[@]}" --out "$kmeans"
    require_fresh "$rf" "$kmeans"
}

define_step 8e --if DO_CONTROLLED_SIM \
    --title "Controlled simulation: five-client missingness inputs" \
    --explain "Simulates five clients for ten roots, two feature panels and four levels of
        abundance-dependent missingness, and scores the correction on them. Step 8f
        trains learners on these inputs."

step_8e() {
    local out
    out="$(cs_dir missingness)"
    x "${PX[@]}" Rscript evaluation_data/simulated/missingness_overlap.R \
        --roots 1,2,3,4,5,6,7,8,9,10 --out "$out"
    require_fresh "$out/runs"
    # Step 8f reads the latest inputs, also in a later run
    x ln -sfn "$out" "${STATE_DIR}/controlled_sim/missingness_latest"
}

define_step 8f --if "DO_CONTROLLED_SIM HAVE_DCOMBAT" --rerun-when-changed DCOMBAT_DIR \
    --title "Controlled simulation: missingness learners with d-ComBat" \
    --explain "Trains random forests on the missingness inputs of step 8e with label-free
        correction (keep, drop and impute routes) and compares them with d-ComBat."

step_8f() {
    local out
    out="$(cs_dir missingness_rf)"
    x "${PX[@]}" python evaluation_data/simulated/downstream_simulation.py \
        --analysis missingness_overlap --rf-protocols loco within_lab_70_30 \
        --saved-inputs "${STATE_DIR}/controlled_sim/missingness_latest/runs" \
        --dcombat "$DCOMBAT_DIR" \
        --runs 1,2,3,4,5,6,7,8,9,10 --learner-seeds 101,211,321,431,541 \
        --reference evaluation_data/simulated/results/missingness_overlap/informative_missingness.tsv \
        --out "$out"
    require_fresh "$out"
}

# ===========================================================================
# 4. Machinery
# ===========================================================================

# --- General helpers ------------------------------------------------------
now() { date '+%F %T'; }
head_commit() { git rev-parse --short HEAD; }
ram_gb() { awk '/^MemTotal:/ {printf "%d", $2 / 1024 / 1024 + 0.5}' /proc/meminfo; }
free_disk_gb() { df -Pk "$REPO_ROOT" | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}'; }

# Prints text wrapped to 80 columns with an indent
wrap() {
    fold -s -w $(( 80 - ${#2} )) <<< "$1" | sed "s/^/$2/"
}

# Prints a command (quoting only arguments that need it) and runs it, unless DRY_RUN=1.
# Use "x env -C DIR command" to run in another directory.
x() {
    local arg shown=()
    for arg in "$@"; do
        if [[ "$arg" =~ ^[[:alnum:]_./:=,@%+-]+$ ]]; then
            shown+=("$arg")
        else
            shown+=("${arg@Q}")
        fi
    done
    echo "+ ${shown[*]}"
    if [[ "$DRY_RUN" != 1 ]]; then
        "$@"
    fi
}

need_cmd() {
    local missing=() cmd
    for cmd in "$@"; do
        if ! command -v "$cmd" > /dev/null; then
            missing+=("$cmd")
        fi
    done
    if (( ${#missing[@]} )); then
        echo "Missing commands: ${missing[*]}" >&2
        return 1
    fi
}

# Paths and checks from evaluation_utils/reproduce_checks.py (read-only)
checks() {
    "${PX[@]}" python -m evaluation_utils.reproduce_checks "$@"
}

# Executes a notebook with its own folder as working directory; the executed
# copy goes to the run folder, so the tracked notebook stays unchanged
run_nb() {
    local out
    out="${RUN_DIR}/notebooks/$(dirname "$1")"
    mkdir -p "$out"
    x "${PX[@]}" jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 --output-dir "$out" "$1"
}

stop_controller() {
    if [[ "${FC_CONTROLLER_USED:-n}" == y ]]; then
        FC_CONTROLLER_USED=n
        x "${PX[@]}" featurecloud controller stop || echo "WARNING: could not stop the FeatureCloud controller"
    fi
}

# --- Output checks --------------------------------------------------------
require_exist() {
    local missing=() path
    for path in "$@"; do
        if [[ ! -e "$path" ]]; then
            missing+=("$path")
        fi
    done
    if (( ${#missing[@]} )); then
        echo "Required files are missing:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        return 1
    fi
}

# Fails unless each file was written during the current step, or each folder
# received at least one file during it: a script that exits successfully
# without writing its output fails the step
require_fresh() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "(dry run) would check that the step wrote: $*"
        return 0
    fi
    require_exist "$@"
    local stale=() path
    for path in "$@"; do
        if [[ -d "$path" ]]; then
            if [[ -z "$(find "$path" -type f -newer "$STEP_MARKER" -print -quit)" ]]; then
                stale+=("$path/")
            fi
        elif [[ ! "$path" -nt "$STEP_MARKER" ]]; then
            stale+=("$path")
        fi
    done
    if (( ${#stale[@]} )); then
        echo "Not written by this step:" >&2
        printf '  %s\n' "${stale[@]}" >&2
        return 1
    fi
    echo "Checked ${#} output(s) written by this step"
}

# require_listed exist|fresh <reproduce_checks.py arguments>
require_listed() {
    local check="$1" listing paths
    shift
    listing="$(checks "$@")"
    mapfile -t paths <<< "$listing"
    "require_${check}" "${paths[@]}"
}

# --- Progress -------------------------------------------------------------
# reproduce_runs/progress.tsv has one line per event; the last line of a step
# wins: step, status (pending, done), time, commit, answers hash, run folder
declare -A P_STATUS=() P_COMMIT=() P_HASH=()

load_progress() {
    P_STATUS=()
    P_COMMIT=()
    P_HASH=()
    if [[ -f "$PROGRESS" ]]; then
        local id status _time commit hash _dir
        while IFS=$'\t' read -r id status _time commit hash _dir; do
            P_STATUS[$id]="$status"
            P_COMMIT[$id]="$commit"
            P_HASH[$id]="$hash"
        done < "$PROGRESS"
    fi
}

record() {
    local id="$1" status="$2" commit hash
    commit="$(head_commit)"
    hash="$(step_hash "$id")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$status" "$(now)" "$commit" "$hash" "${RUN_DIR:-}" \
        >> "$PROGRESS"
    P_STATUS[$id]="$status"
    P_COMMIT[$id]="$commit"
    P_HASH[$id]="$hash"
}

# Short hash of the answers that change a step's result
step_hash() {
    local var text=""
    for var in ${STEP_RERUN_WHEN[$1]}; do
        text+="${var}=${!var};"
    done
    if [[ -z "$text" ]]; then
        echo "-"
    else
        printf '%s' "$text" | sha256sum | cut -c1-12
    fi
}

step_done() {
    [[ "${P_STATUS[$1]:-}" == "done" && "${P_HASH[$1]:-}" == "$(step_hash "$1")" ]]
}

status_of() {
    if step_done "$1"; then
        echo "done"
    elif [[ "${P_STATUS[$1]:-}" == "done" ]]; then
        echo "done (other answers)"
    else
        echo "${P_STATUS[$1]:--}"
    fi
}

list_steps() {
    local id
    printf '%-4s %-20s %-8s %s\n' "Step" "Recorded" "Commit" "Title"
    for id in "${STEP_IDS[@]}"; do
        printf '%-4s %-20s %-8s %s\n' "$id" "$(status_of "$id")" "${P_COMMIT[$id]:--}" "${STEP_TITLE[$id]}"
    done
}

explain_all() {
    local var id
    echo "QUESTIONS"
    for var in "${ANSWER_VARS[@]}"; do
        echo ""
        echo "${var}: ${Q_ASK[$var]}"
        if [[ -n "${Q_IF[$var]}" ]]; then
            echo "  (asked only if ${Q_IF[$var]// / and } = y)"
        fi
        wrap "${Q_EXPLAIN[$var]}" "  "
    done
    echo ""
    echo "STEPS"
    for id in "${STEP_IDS[@]}"; do
        echo ""
        echo "${id}: ${STEP_TITLE[$id]}"
        if [[ -n "${STEP_IF[$id]}" ]]; then
            echo "  (runs only if ${STEP_IF[$id]// / and } = y)"
        fi
        wrap "${STEP_EXPLAIN[$id]}" "  "
    done
}

valid_step() { [[ -n "${STEP_TITLE[$1]+set}" ]]; }

# Prints the unknown ids of a step list ("3 7a" or "3,7a"), nothing when all are valid
unknown_steps() {
    local id
    for id in ${1//,/ }; do
        if ! valid_step "$id"; then
            printf '%s ' "$id"
        fi
    done
}

in_plan() {
    local id
    for id in "${RUN_PLAN[@]}"; do
        if [[ "$id" == "$1" ]]; then
            return 0
        fi
    done
    return 1
}

# Every step and question needs its fields, and every step its function
check_definitions() {
    local id var
    for id in "${STEP_IDS[@]}"; do
        [[ -n "${STEP_TITLE[$id]:-}" && -n "${STEP_EXPLAIN[$id]:-}" ]] \
            || die "Step ${id} needs --title and --explain"
        declare -F "step_${id}" > /dev/null || die "Step ${id} has no function step_${id}"
    done
    for var in "${ANSWER_VARS[@]}"; do
        [[ -n "${Q_ASK[$var]:-}" && -n "${Q_EXPLAIN[$var]:-}" ]] \
            || die "Question ${var} needs --ask and --explain"
    done
}

# --- Answers --------------------------------------------------------------
# all_yes "VAR1 VAR2": true when every listed variable is y (also for an empty list)
all_yes() {
    local var
    for var in $1; do
        if [[ "${!var:-n}" != y ]]; then
            return 1
        fi
    done
}

step_enabled() { all_yes "${STEP_IF[$1]}"; }

# Built-in defaults; a later answers file overrides them
init_answers() {
    local var by_ram
    for var in "${ANSWER_VARS[@]}"; do
        printf -v "$var" '%s' "${Q_DEFAULT[$var]}"
    done
    by_ram=$(( $(ram_gb) / 4 ))
    N_JOBS=$(( by_ram < $(nproc) ? by_ram : $(nproc) ))
    N_JOBS=$(( N_JOBS < 1 ? 1 : N_JOBS ))
    set_derived
}

load_answers() {
    # shellcheck disable=SC1090
    source "$1"
    set_derived
}

write_answers() {
    local var
    for var in "${ANSWER_VARS[@]}"; do
        printf '%s=%q\n' "$var" "${!var}"
    done > "$1"
}

# Variables used in step and question conditions (read indirectly)
# shellcheck disable=SC2034
set_derived() {
    HAVE_DCOMBAT=n
    if [[ -n "$DCOMBAT_DIR" ]]; then
        HAVE_DCOMBAT=y
    fi
    LOW_RAM=n
    if (( $(ram_gb) < MIN_RAM_GB )); then
        LOW_RAM=y
    fi
}

# Maps the accepted spellings to the stored value
normalize_answer() {
    case "${Q_KIND[$1]}:${2,,}" in
        yn:yes) echo y ;;
        yn:no) echo n ;;
        yn:*) echo "${2,,}" ;;
        dir:none | dir:-) echo "" ;;
        *) echo "$2" ;;
    esac
}

# Prints what is wrong with an answer, nothing when it is valid
answer_problem() {
    local var="$1" value="$2"
    case "${Q_KIND[$var]}" in
        yn) [[ "$value" == [yn] ]] || echo "${var} must be y or n, not '${value}'" ;;
        int) [[ "$value" =~ ^[1-9][0-9]*$ ]] || echo "${var} must be a positive integer, not '${value}'" ;;
        dir) [[ -z "$value" || -f "${value}/distributedCombat.R" ]] \
                 || echo "${var}: ${value}/distributedCombat.R not found" ;;
    esac
}

validate_answers() {
    local var problem
    for var in "${ANSWER_VARS[@]}"; do
        problem="$(answer_problem "$var" "${!var}")"
        if [[ -n "$problem" ]]; then
            die "$problem"
        fi
    done
    set_derived
}

# Asks one question with its explanation; the current value is the default
ask() {
    local var="$1" current="${!1}" hint reply problem
    case "${Q_KIND[$var]}" in
        yn) if [[ "$current" == y ]]; then hint="Y/n"; else hint="y/N"; fi ;;
        *) hint="${current:-none}" ;;
    esac
    echo ""
    wrap "${Q_EXPLAIN[$var]}" "  "
    while true; do
        read -r -p "${Q_ASK[$var]} [${hint}] " reply || die "No answer (end of input)"
        reply="$(normalize_answer "$var" "${reply:-$current}")"
        problem="$(answer_problem "$var" "$reply")"
        if [[ -z "$problem" ]]; then
            printf -v "$var" '%s' "$reply"
            set_derived
            return 0
        fi
        echo "  ${problem}"
    done
}

questionnaire() {
    local var
    echo ""
    echo "Which expensive stages should run? Enter keeps the answer in brackets."
    for var in "${ANSWER_VARS[@]}"; do
        if all_yes "${Q_IF[$var]}"; then
            ask "$var"
        fi
    done
}

confirm() {
    local reply
    read -r -p "$1 [y/N] " reply || die "No answer (end of input)"
    [[ "${reply,,}" == y || "${reply,,}" == yes ]]
}

# ask_steps <prompt> one|many: prints the entered step ids
ask_steps() {
    local reply unknown
    while true; do
        read -r -p "$1 " reply || die "No answer (end of input)"
        reply="${reply//,/ }"
        unknown="$(unknown_steps "$reply")"
        if [[ -z "${reply// /}" ]]; then
            echo "  Enter a step." >&2
        elif [[ -n "$unknown" ]]; then
            echo "  Unknown: ${unknown}(steps: ${STEP_IDS[*]})" >&2
        elif [[ "$2" == one && "$(wc -w <<< "$reply")" -ne 1 ]]; then
            echo "  Enter one step." >&2
        else
            echo "$reply"
            return 0
        fi
    done
}

# --- Planning (foreground) ------------------------------------------------
# MODE: resume, from, selected or fresh; MODE_STEPS: the step of "from", the
# steps of "selected"
choose_mode() {
    if [[ -n "$MODE" || ${#P_STATUS[@]} -eq 0 ]]; then
        MODE="${MODE:-resume}"
        return 0
    fi
    echo "Recorded progress (${PROGRESS}):"
    list_steps | sed 's/^/  /'
    local last_status
    last_status="$(find "$STATE_DIR" -mindepth 2 -maxdepth 2 -name STATUS -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-)"
    if [[ -n "$last_status" ]]; then
        echo "Last run: $(dirname "$last_status") -> $(< "$last_status")"
    fi
    if [[ "$INTERACTIVE" != y ]]; then
        MODE=resume
        return 0
    fi
    echo ""
    echo "[c] continue: start at the first step that is not done; later steps run again"
    echo "[r] rerun from a chosen step: that step and every later step"
    echo "[s] rerun selected steps only"
    echo "[f] fresh start: set the recorded progress aside and run everything"
    local reply
    while true; do
        read -r -p "Choice [c] " reply || die "No answer (end of input)"
        reply="${reply,,}"
        case "${reply:-c}" in
            c) MODE=resume; return 0 ;;
            r) MODE=from; MODE_STEPS="$(ask_steps "Rerun from step:" one)"; return 0 ;;
            s) MODE=selected; MODE_STEPS="$(ask_steps "Steps to rerun (e.g. 3 7a 7b):" many)"; return 0 ;;
            f) MODE=fresh; return 0 ;;
        esac
        echo "  Please answer c, r, s or f."
    done
}

# Index in STEP_IDS of the first step that "resume" runs; empty when all are done
resume_start() {
    local i j id head stale=() first_stale=""
    for (( i = 1; i < ${#STEP_IDS[@]}; i++ )); do
        if step_enabled "${STEP_IDS[$i]}" && ! step_done "${STEP_IDS[$i]}"; then
            break
        fi
    done
    # Done steps before that point that ran at another commit
    head="$(head_commit)"
    for (( j = 1; j < i; j++ )); do
        id="${STEP_IDS[$j]}"
        if step_enabled "$id" && [[ "${P_COMMIT[$id]:-}" != "$head" ]]; then
            stale+=("${id} (done at ${P_COMMIT[$id]:-?})")
            first_stale="${first_stale:-$j}"
        fi
    done
    if (( ${#stale[@]} )); then
        {
            echo ""
            echo "WARNING: these done steps ran at another commit than HEAD ${head}:"
            printf '  %s\n' "${stale[@]}"
        } >&2
        if [[ "$INTERACTIVE" == y ]] && confirm "Rerun from step ${STEP_IDS[$first_stale]}?" >&2; then
            i="$first_stale"
        fi
    fi
    if (( i < ${#STEP_IDS[@]} )); then
        echo "$i"
    fi
}

# Fills RUN_PLAN with step 0 and the enabled steps to run
build_plan() {
    local start="" i id
    case "$MODE" in
        fresh) start=1 ;;
        resume) start="$(resume_start)" ;;
        from)
            for i in "${!STEP_IDS[@]}"; do
                if [[ "${STEP_IDS[$i]}" == "$MODE_STEPS" ]]; then
                    start=$(( i > 0 ? i : 1 ))
                fi
            done
            ;;
        selected)
            for id in $MODE_STEPS; do
                if ! step_enabled "$id"; then
                    die "Step ${id} is switched off by the answers (${STEP_IF[$id]// /, } = n)"
                fi
            done
            ;;
    esac
    RUN_PLAN=(0)
    for (( i = 1; i < ${#STEP_IDS[@]}; i++ )); do
        id="${STEP_IDS[$i]}"
        if ! step_enabled "$id"; then
            continue
        fi
        if [[ "$MODE" == selected ]]; then
            if [[ " ${MODE_STEPS} " != *" ${id} "* ]]; then
                continue
            fi
        elif [[ -z "$start" ]] || (( i < start )); then
            continue
        fi
        RUN_PLAN+=("$id")
    done
}

show_plan() {
    local id
    echo ""
    echo "Plan:"
    for id in "${STEP_IDS[@]}"; do
        if in_plan "$id"; then
            printf '  %-4s RUN   %s\n' "$id" "${STEP_TITLE[$id]}"
        elif ! step_enabled "$id"; then
            printf '  %-4s skip  %s\n' "$id" "${STEP_TITLE[$id]}"
        else
            printf '  %-4s keep  %s (recorded: %s)\n' "$id" "${STEP_TITLE[$id]}" "$(status_of "$id")"
        fi
    done
    echo ""
}

# --- Running (inside screen, or in this terminal with --no-screen) ---------
SUMMARY=()
CURRENT_STEP="setup"

print_summary() {
    if (( ${#SUMMARY[@]} )); then
        echo ""
        echo "Steps of this run:"
        printf '  %s\n' "${SUMMARY[@]}"
    fi
}

on_err() {
    local code="$1" line="$2" command="$3"
    if (( BASH_SUBSHELL > 0 )); then
        exit "$code"
    fi
    trap - ERR
    set +e
    echo "FAILED ${CURRENT_STEP}" > "${RUN_DIR}/STATUS"
    stop_controller > /dev/null 2>&1
    SUMMARY+=("${CURRENT_STEP} FAILED")
    {
        echo ""
        echo "############################################################"
        echo "FAILED at step ${CURRENT_STEP}: ${STEP_TITLE[$CURRENT_STEP]:-}"
        echo "  command:   ${command} (in step_${CURRENT_STEP})"
        echo "  exit code: ${code} (reproduce.sh line ${line})"
        echo "  log:       ${LOG}"
        echo "  elapsed:   $(( SECONDS / 60 )) min"
        echo "Continue with: bash reproduce.sh --resume"
        echo "############################################################"
        print_summary
    } >&2
    exit "$code"
}

on_signal() {
    trap - ERR INT TERM HUP
    echo "INTERRUPTED ${CURRENT_STEP}" > "${RUN_DIR}/STATUS"
    stop_controller > /dev/null 2>&1 || true
    echo "INTERRUPTED at step ${CURRENT_STEP}" >&2
    exit 130
}

# run_plan <run folder>: runs the steps written to the folder by launch
run_plan() {
    RUN_DIR="$1"
    RUN_ID="$(basename "$RUN_DIR")"
    LOG="${RUN_DIR}/reproduce.log"
    exec > >(tee -a "$LOG") 2>&1
    trap 'on_err $? $LINENO "$BASH_COMMAND"' ERR
    trap on_signal INT TERM HUP

    exec 9> "$LOCK"
    flock -n 9 || die "Another reproduction holds ${LOCK}"

    init_answers
    load_answers "${RUN_DIR}/answers.env"
    validate_answers
    mapfile -t RUN_PLAN < "${RUN_DIR}/plan"
    load_progress
    export PYTHONUNBUFFERED=1

    local dry=""
    if [[ "$DRY_RUN" == 1 ]]; then
        dry=", DRY RUN"
    fi
    echo "[$(now)] fedRBE reproduction ${RUN_ID}, commit $(head_commit)${dry}"
    echo "Answers:"
    sed 's/^/  /' "${RUN_DIR}/answers.env"
    echo "Steps: ${RUN_PLAN[*]}"

    # Planned steps lose their earlier "done" until they succeed again
    local id started
    for id in "${RUN_PLAN[@]}"; do
        record "$id" pending
    done

    for id in "${RUN_PLAN[@]}"; do
        CURRENT_STEP="$id"
        echo ""
        echo "============================================================"
        echo "[$(now)] Step ${id}: ${STEP_TITLE[$id]}"
        wrap "${STEP_EXPLAIN[$id]}" "  "
        echo "============================================================"
        echo "RUNNING ${id}" > "${RUN_DIR}/STATUS"
        STEP_MARKER="${RUN_DIR}/.step_${id}_start"
        touch "$STEP_MARKER"
        started=$SECONDS
        if [[ "${REPRO_FAIL_AT:-}" == "$id" ]]; then
            echo "REPRO_FAIL_AT=${id}: failing on purpose"
            false
        fi
        "step_${id}"
        record "$id" "done"
        SUMMARY+=("${id} done in $(( (SECONDS - started) / 60 )) min")
        echo "[$(now)] Step ${id} done"
    done

    CURRENT_STEP="end"
    echo "SUCCEEDED" > "${RUN_DIR}/STATUS"
    print_summary
    echo ""
    echo "[$(now)] All steps completed in $(( SECONDS / 60 )) min."
    echo "Figures: figures/main/, figures/suppl/, figures/candidate/"
}

# Writes the run folder and starts run_plan in screen or in this terminal
launch() {
    local run_id run_dir session
    run_id="$(date +%Y%m%d_%H%M%S)_$$"
    run_dir="${STATE_DIR}/${run_id}"
    mkdir -p "$run_dir"
    write_answers "${run_dir}/answers.env"
    cp "${run_dir}/answers.env" "$LAST_ANSWERS"
    printf '%s\n' "${RUN_PLAN[@]}" > "${run_dir}/plan"
    echo "STARTING" > "${run_dir}/STATUS"

    if [[ "$NO_SCREEN" == y ]]; then
        exec bash "$SCRIPT_PATH" --run "$run_dir"
    fi
    session="fedrbe_reproduce_${run_id}"
    screen -dmS "$session" bash "$SCRIPT_PATH" --run "$run_dir"
    echo "Started screen session ${session}"
    echo "Log:    ${run_dir}/reproduce.log"
    echo "Status: cat ${run_dir}/STATUS"
    echo "Follow: tail -f ${run_dir}/reproduce.log"
    echo "Attach: screen -r ${session} (detach with Ctrl-A, D)"
}

usage() {
    cat << EOF
Usage: bash reproduce.sh [options]

Runs the reproduction of docs/reproduce.md and stops at the first failing step.

Where to start (asked when progress exists; default: --resume):
  --resume               start at the first enabled step that is not done
  --rerun-from STEP      run STEP and every later step again
  --rerun STEP[,STEP..]  run only these steps again (plus preflight)
  --fresh                set the recorded progress aside and run everything

Answers (default: the questionnaire, with the previous answers as defaults):
  --answers FILE         take the answers from FILE, e.g. an earlier run's answers.env
  --yes-defaults         take the previous or default answers without asking
  --yes                  skip the final confirmation

Other:
  --no-screen            run in this terminal instead of a detached screen session
  --list-steps           list the steps with their recorded status
  --explain              explain every question and step
  -h, --help             this help

Environment: DRY_RUN=1 prints the commands instead of running them (progress
then goes to reproduce_runs/dry_run/), REPRO_FAIL_AT=STEP makes that step fail,
REPRO_STATE_DIR sets the progress folder (now ${STATE_DIR}).
EOF
}

main() {
    check_definitions
    local answers_file="" use_defaults=n assume_yes=n
    MODE=""
    MODE_STEPS=""
    NO_SCREEN=n
    while (( $# )); do
        case "$1" in
            --run) run_plan "${2:?--run needs a run folder}"; return 0 ;;
            --resume) MODE=resume ;;
            --rerun-from) MODE=from; MODE_STEPS="${2:?--rerun-from needs a step}"; shift ;;
            --rerun) MODE=selected; MODE_STEPS="${2:?--rerun needs steps}"; shift ;;
            --fresh) MODE=fresh ;;
            --answers) answers_file="${2:?--answers needs a file}"; shift ;;
            --yes-defaults) use_defaults=y ;;
            --yes) assume_yes=y ;;
            --no-screen) NO_SCREEN=y ;;
            --list-steps)
                init_answers
                if [[ -f "$LAST_ANSWERS" ]]; then
                    load_answers "$LAST_ANSWERS"
                fi
                load_progress
                list_steps
                return 0
                ;;
            --explain) explain_all; return 0 ;;
            -h | --help) usage; return 0 ;;
            *) usage >&2; die "Unknown option: $1" ;;
        esac
        shift
    done

    MODE_STEPS="${MODE_STEPS//,/ }"
    local unknown
    unknown="$(unknown_steps "$MODE_STEPS")"
    if [[ -n "$unknown" ]]; then
        die "Unknown step(s): ${unknown}(steps: ${STEP_IDS[*]})"
    fi
    INTERACTIVE=n
    if [[ -t 0 && "$use_defaults" != y ]]; then
        INTERACTIVE=y
    fi
    if [[ "$INTERACTIVE" != y && "$use_defaults" != y && -z "$answers_file" ]]; then
        die "No terminal for the questionnaire: use --answers FILE or --yes-defaults"
    fi
    if [[ "$NO_SCREEN" != y ]]; then
        need_cmd screen || die "Install screen or use --no-screen"
    fi
    need_cmd flock git || die "flock and git are required"
    mkdir -p "$STATE_DIR"
    flock -n "$LOCK" true || die "Another reproduction is running (lock ${LOCK})"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY RUN: commands are printed, not run; progress goes to ${PROGRESS}"
    fi

    # Answers: defaults, then the previous (or given) answers
    init_answers
    if [[ -n "$answers_file" ]]; then
        [[ -f "$answers_file" ]] || die "Answers file not found: ${answers_file}"
        load_answers "$answers_file"
    elif [[ -f "$LAST_ANSWERS" ]]; then
        load_answers "$LAST_ANSWERS"
    fi
    load_progress

    choose_mode
    if [[ "$INTERACTIVE" == y && -z "$answers_file" ]]; then
        questionnaire
    fi
    validate_answers
    build_plan
    if (( ${#RUN_PLAN[@]} == 1 )) && [[ "$MODE" == resume ]]; then
        echo "All enabled steps are done. Use --rerun-from STEP, --rerun STEP or --fresh to run steps again."
        return 0
    fi

    show_plan
    if [[ "$INTERACTIVE" == y && "$assume_yes" != y ]] && ! confirm "Start?"; then
        echo "Not started."
        return 0
    fi
    if [[ "$MODE" == fresh && -f "$PROGRESS" ]]; then
        local backup
        backup="${PROGRESS}.$(date +%Y%m%d_%H%M%S).bak"
        mv "$PROGRESS" "$backup"
        echo "Earlier progress moved to ${backup}"
    fi
    launch
}

main "$@"
