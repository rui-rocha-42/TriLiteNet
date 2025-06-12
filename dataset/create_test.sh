#!/bin/bash

# Check if the required arguments are provided
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <original_dataset_dir> <sample_dataset_dir> <test_dataset_dir>"
    exit 1
fi

ORIGINAL_DIR="$1"
SAMPLE_DIR="$2"
TEST_DIR="$3"

# Define subdirectories
SUBDIRS=("images" "da_seg_annotations" "det_annotations" "ll_seg_annotations")

# Create test directories
for SUBDIR in "${SUBDIRS[@]}"; do
    mkdir -p "$TEST_DIR/$SUBDIR/train"
    mkdir -p "$TEST_DIR/$SUBDIR/val"
done

# Function to gather remaining files
gather_remaining_files() {
    local ORIGINAL_SUBDIR="$1"
    local SAMPLE_SUBDIR="$2"
    local TEST_SUBDIR="$3"
    local EXTENSION="$4"

    for SPLIT in "train" "val"; do
        ORIGINAL_SPLIT_DIR="$ORIGINAL_SUBDIR/$SPLIT"
        SAMPLE_SPLIT_DIR="$SAMPLE_SUBDIR/$SPLIT"
        TEST_SPLIT_DIR="$TEST_SUBDIR/$SPLIT"

        # Create a list of basenames in the original directory
        ORIGINAL_BASENAMES=$(mktemp)
        find "$ORIGINAL_SPLIT_DIR" -type f -name "*.$EXTENSION" -printf "%f\n" | sed 's/\.[^.]*$//' > "$ORIGINAL_BASENAMES"

        # Create a list of basenames in the sample directory
        SAMPLE_BASENAMES=$(mktemp)
        find "$SAMPLE_SPLIT_DIR" -type f -name "*.$EXTENSION" -printf "%f\n" | sed 's/\.[^.]*$//' > "$SAMPLE_BASENAMES"

        # Find basenames that are in the original directory but not in the sample directory
        REMAINING_BASENAMES=$(mktemp)
        grep -Fxv -f "$SAMPLE_BASENAMES" "$ORIGINAL_BASENAMES" > "$REMAINING_BASENAMES"

        # Copy remaining files to the test directory
        while read -r BASENAME; do
            if [[ -f "$ORIGINAL_SPLIT_DIR/$BASENAME.$EXTENSION" ]]; then
                cp "$ORIGINAL_SPLIT_DIR/$BASENAME.$EXTENSION" "$TEST_SPLIT_DIR/"
            fi
        done < "$REMAINING_BASENAMES"

        # Clean up temporary files
        rm "$ORIGINAL_BASENAMES" "$SAMPLE_BASENAMES" "$REMAINING_BASENAMES"
    done
}

# Process each subdirectory
gather_remaining_files "$ORIGINAL_DIR/images" "$SAMPLE_DIR/images" "$TEST_DIR/images" "jpg"
gather_remaining_files "$ORIGINAL_DIR/da_seg_annotations" "$SAMPLE_DIR/da_seg_annotations" "$TEST_DIR/da_seg_annotations" "png"
gather_remaining_files "$ORIGINAL_DIR/det_annotations" "$SAMPLE_DIR/det_annotations" "$TEST_DIR/det_annotations" "json"
gather_remaining_files "$ORIGINAL_DIR/ll_seg_annotations" "$SAMPLE_DIR/ll_seg_annotations" "$TEST_DIR/ll_seg_annotations" "png"

echo "Test dataset created in $TEST_DIR"