#!/bin/bash

# Script to run all classification analysis experiments sequentially

set -e  # Exit on any error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Starting classification analysis workflow..."
echo "=============================================="

echo ""
echo "Step 1: Running classification with train-test split..."
python3 "$SCRIPT_DIR/run_classification_train_test_split.py"
echo "✓ Train-test split classification completed"

echo ""
echo "Step 2: Running classification with leave-one-cohort-out..."
python3 "$SCRIPT_DIR/run_classification_leave_one_cohort_out.py"
echo "✓ Leave-one-cohort-out classification completed"

echo ""
echo "Step 3: Analyzing classification metrics..."
python3 "$SCRIPT_DIR/analyse_classification_metric_report.py"
echo "✓ Classification analysis completed"

echo ""
echo "=============================================="
echo "All classification analysis steps completed successfully!"
