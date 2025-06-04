#!/bin/bash

# Default values
SOURCE_DIR="./images"
SAMPLE_DIR="./sample_dataset"
TRAIN_SAMPLES=1000
VAL_SAMPLES=250

# Function to display usage
usage() {
    echo "Usage: $0 [-s SOURCE_DIR] [-d SAMPLE_DIR] [-t TRAIN_SAMPLES] [-v VAL_SAMPLES]"
    echo "  -s SOURCE_DIR    Path to the source directory containing images (default: ./images)"
    echo "  -d SAMPLE_DIR    Path to the sample dataset directory (default: ./sample_dataset)"
    echo "  -t TRAIN_SAMPLES Number of train samples to create (default: 1000)"
    echo "  -v VAL_SAMPLES   Number of validation samples to create (default: 250)"
    exit 1
}

# Parse arguments
while getopts "s:d:t:v:h" opt; do
    case $opt in
        s) SOURCE_DIR="$OPTARG" ;;
        d) SAMPLE_DIR="$OPTARG" ;;
        t) TRAIN_SAMPLES="$OPTARG" ;;
        v) VAL_SAMPLES="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Create train and val directories
mkdir -p "$SAMPLE_DIR/train" "$SAMPLE_DIR/val"

# Create a list of image filenames
IMAGE_LIST=$(mktemp)
find "$SOURCE_DIR" -type f -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" > "$IMAGE_LIST"

# Shuffle and split the image list into train and val samples
shuf "$IMAGE_LIST" > "$IMAGE_LIST.shuffled"
head -n "$TRAIN_SAMPLES" "$IMAGE_LIST.shuffled" > "$IMAGE_LIST.train"
head -n "$((TRAIN_SAMPLES + VAL_SAMPLES))" "$IMAGE_LIST.shuffled" | tail -n "$VAL_SAMPLES" > "$IMAGE_LIST.val"

# Copy train files to the train directory
while read -r IMAGE_PATH; do
    cp "$IMAGE_PATH" "$SAMPLE_DIR/train/"
done < "$IMAGE_LIST.train"

# Copy val files to the val directory
while read -r IMAGE_PATH; do
    cp "$IMAGE_PATH" "$SAMPLE_DIR/val/"
done < "$IMAGE_LIST.val"

# Clean up temporary files
rm "$IMAGE_LIST" "$IMAGE_LIST.shuffled" "$IMAGE_LIST.train" "$IMAGE_LIST.val"

echo "Train sample of $TRAIN_SAMPLES images and val sample of $VAL_SAMPLES images created in $SAMPLE_DIR"