#!/bin/bash

# Directory where generated JSON files are stored
JSON_DIR="json/hltSeedTracks_"

# Your Python plotting script
PLOT_SCRIPT="plotter_withratio.py"

# Loop over all JSON files in the directory
for json_file in "$JSON_DIR"*.json; do
    echo "[INFO] Running plotter for $json_file ..."
    pyroot "$PLOT_SCRIPT" "$json_file"
done
