#!/bin/bash

# Default values
MERGED_DATASET="./merged_dataset"
DATASETS=()

# Function to display usage
usage() {
    echo "Usage: $0 [-d DATASET]... [-m MERGED_DATASET]"
    echo "  -d DATASET        Path to a dataset directory (can be specified multiple times)"
    echo "  -m MERGED_DATASET Path to the merged dataset directory (default: ./merged_dataset)"
    exit 1
}

# Parse arguments
while getopts "d:m:h" opt; do
    case $opt in
        d) DATASETS+=("$OPTARG") ;;  # Add dataset to the list
        m) MERGED_DATASET="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Ensure at least one dataset is provided
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: At least one dataset must be specified with -d."
    usage
fi

# Create merged dataset directories
mkdir -p "$MERGED_DATASET/images/train" "$MERGED_DATASET/images/val"
mkdir -p "$MERGED_DATASET/ll_seg_annotations/train" "$MERGED_DATASET/ll_seg_annotations/val"
mkdir -p "$MERGED_DATASET/da_seg_annotations/train" "$MERGED_DATASET/da_seg_annotations/val"
mkdir -p "$MERGED_DATASET/det_annotations/train" "$MERGED_DATASET/det_annotations/val"

# Function to merge directories
merge_dir() {
    local DEST="$1"
    shift
    local SOURCES=("$@")

    for SRC in "${SOURCES[@]}"; do
        if [ -d "$SRC" ]; then
            cp -r "$SRC/"* "$DEST/"
        fi
    done
}

# Merge train and val subdirectories for each folder
TRAIN_SOURCES=()
VAL_SOURCES=()

for DATASET in "${DATASETS[@]}"; do
    TRAIN_SOURCES+=("$DATASET/images/train")
    VAL_SOURCES+=("$DATASET/images/val")
done
merge_dir "$MERGED_DATASET/images/train" "${TRAIN_SOURCES[@]}"
merge_dir "$MERGED_DATASET/images/val" "${VAL_SOURCES[@]}"

TRAIN_SOURCES=()
VAL_SOURCES=()

for DATASET in "${DATASETS[@]}"; do
    TRAIN_SOURCES+=("$DATASET/ll_seg_annotations/train")
    VAL_SOURCES+=("$DATASET/ll_seg_annotations/val")
done
merge_dir "$MERGED_DATASET/ll_seg_annotations/train" "${TRAIN_SOURCES[@]}"
merge_dir "$MERGED_DATASET/ll_seg_annotations/val" "${VAL_SOURCES[@]}"

TRAIN_SOURCES=()
VAL_SOURCES=()

for DATASET in "${DATASETS[@]}"; do
    TRAIN_SOURCES+=("$DATASET/da_seg_annotations/train")
    VAL_SOURCES+=("$DATASET/da_seg_annotations/val")
done
merge_dir "$MERGED_DATASET/da_seg_annotations/train" "${TRAIN_SOURCES[@]}"
merge_dir "$MERGED_DATASET/da_seg_annotations/val" "${VAL_SOURCES[@]}"

TRAIN_SOURCES=()
VAL_SOURCES=()

for DATASET in "${DATASETS[@]}"; do
    TRAIN_SOURCES+=("$DATASET/det_annotations/train")
    VAL_SOURCES+=("$DATASET/det_annotations/val")
done
merge_dir "$MERGED_DATASET/det_annotations/train" "${TRAIN_SOURCES[@]}"
merge_dir "$MERGED_DATASET/det_annotations/val" "${VAL_SOURCES[@]}"

echo "Datasets merged into $MERGED_DATASET"