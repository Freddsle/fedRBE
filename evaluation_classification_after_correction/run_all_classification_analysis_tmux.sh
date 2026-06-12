#!/bin/bash
# Runs the helper script run_all_classification_analysis.sh in a tmux session,
# if tmux is available. Otherwise, throws an error
set -e  # Exit on any error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Generate a dynamic datetime stamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TMUX_SESSION_NAME="classification_analysis_${TIMESTAMP}"

if command -v tmux &> /dev/null
then
    echo "tmux is available. Running the classification analysis in a tmux session..."
    tmux new-session -d -s "$TMUX_SESSION_NAME" "bash $SCRIPT_DIR/run_all_classification_analysis.sh; read -n 1 -s -r -p 'Press any key to exit...'"
    echo "tmux session '$TMUX_SESSION_NAME' started. You can attach to it using 'tmux attach-session -t $TMUX_SESSION_NAME'"
else
    echo "Error: tmux is not installed. Please install tmux to run this script."
    exit 1
fi