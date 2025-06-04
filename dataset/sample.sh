#!/bin/bash

# Default values
DATASET_DIR="/dataset/bdd100k"
SAMPLE_DIR="/dataset/bdd100k_sample"
TRAIN_SAMPLES=1000
VAL_SAMPLES=250

# Function to display usage
usage() {
    echo "Usage: $0 [-d DATASET_DIR] [-s SAMPLE_DIR] [-t TRAIN_SAMPLES] [-v VAL_SAMPLES]"
    echo "  -d DATASET_DIR   Path to the dataset directory (default: /dataset/bdd100k)"
    echo "  -s SAMPLE_DIR    Path to the sample directory (default: /dataset/bdd100k_sample)"
    echo "  -t TRAIN_SAMPLES Number of train samples to create (default: 1000)"
    echo "  -v VAL_SAMPLES   Number of validation samples to create (default: 250)"
    exit 1
}

# Parse arguments
while getopts "d:s:t:v:h" opt; do
    case $opt in
        d) DATASET_DIR="$OPTARG" ;;
        s) SAMPLE_DIR="$OPTARG" ;;
        t) TRAIN_SAMPLES="$OPTARG" ;;
        v) VAL_SAMPLES="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Paths to the original dataset folders
DA_SEG_DIR="$DATASET_DIR/da_seg_annotations"
DET_DIR="$DATASET_DIR/det_annotations"
IMAGES_DIR="$DATASET_DIR/images"
LL_SEG_DIR="$DATASET_DIR/ll_seg_annotations"

# Paths to the new dataset folders
mkdir -p "$SAMPLE_DIR/da_seg_annotations/train" "$SAMPLE_DIR/da_seg_annotations/val"
mkdir -p "$SAMPLE_DIR/det_annotations/train" "$SAMPLE_DIR/det_annotations/val"
mkdir -p "$SAMPLE_DIR/images/train" "$SAMPLE_DIR/images/val"
mkdir -p "$SAMPLE_DIR/ll_seg_annotations/train" "$SAMPLE_DIR/ll_seg_annotations/val"

# Create a list of valid basenames (files that exist in all four folders)
VALID_BASENAMES=$(mktemp)
find "$IMAGES_DIR/train" -type f -printf "%f\n" | sed 's/\.[^.]*$//' > "$VALID_BASENAMES"

# Filter basenames to ensure they exist in all folders
SAMPLE_BASENAMES=$(mktemp)
while read -r BASENAME; do
    if [[ -f "$DA_SEG_DIR/train/$BASENAME.png" && -f "$DET_DIR/train/$BASENAME.json" && -f "$LL_SEG_DIR/train/$BASENAME.png" ]]; then
        echo "$BASENAME" >> "$SAMPLE_BASENAMES"
    fi
done < "$VALID_BASENAMES"

# Shuffle and split basenames into train and val samples
shuf "$SAMPLE_BASENAMES" > "$SAMPLE_BASENAMES.shuffled"
head -n "$TRAIN_SAMPLES" "$SAMPLE_BASENAMES.shuffled" > "$SAMPLE_BASENAMES.train"
head -n "$((TRAIN_SAMPLES + VAL_SAMPLES))" "$SAMPLE_BASENAMES.shuffled" | tail -n "$VAL_SAMPLES" > "$SAMPLE_BASENAMES.val"

# Copy train files to the new dataset folder
while read -r BASENAME; do
    cp "$IMAGES_DIR/train/$BASENAME.jpg" "$SAMPLE_DIR/images/train/"
    cp "$DA_SEG_DIR/train/$BASENAME.png" "$SAMPLE_DIR/da_seg_annotations/train/"
    cp "$DET_DIR/train/$BASENAME.json" "$SAMPLE_DIR/det_annotations/train/"
    cp "$LL_SEG_DIR/train/$BASENAME.png" "$SAMPLE_DIR/ll_seg_annotations/train/"
done < "$SAMPLE_BASENAMES.train"

# Copy val files to the new dataset folder
while read -r BASENAME; do
    cp "$IMAGES_DIR/train/$BASENAME.jpg" "$SAMPLE_DIR/images/val/"
    cp "$DA_SEG_DIR/train/$BASENAME.png" "$SAMPLE_DIR/da_seg_annotations/val/"
    cp "$DET_DIR/train/$BASENAME.json" "$SAMPLE_DIR/det_annotations/val/"
    cp "$LL_SEG_DIR/train/$BASENAME.png" "$SAMPLE_DIR/ll_seg_annotations/val/"
done < "$SAMPLE_BASENAMES.val"

# Clean up temporary files
rm "$VALID_BASENAMES" "$SAMPLE_BASENAMES" "$SAMPLE_BASENAMES.shuffled" "$SAMPLE_BASENAMES.train" "$SAMPLE_BASENAMES.val"

echo "Train sample of $TRAIN_SAMPLES images and val sample of $VAL_SAMPLES images created in $SAMPLE_DIR"