DATASET_DIR="dataset"
TEST_DIR="dataset_test"
mkdir -p "$TEST_DIR"

for CLASS_DIR in "$DATASET_DIR"/*; do

  [ -d "$CLASS_DIR" ] || continue

  CLASS_NAME=$(basename "$CLASS_DIR")
  mkdir -p "$TEST_DIR/$CLASS_NAME"

  # Calculate 10% of the images
  TOTAL_IMAGES=$(find "$CLASS_DIR" -type f | wc -l)
  IMAGES_TO_MOVE=$((TOTAL_IMAGES / 10))

  # Move the images
  find "$CLASS_DIR" -type f | shuf -n "$IMAGES_TO_MOVE" | xargs -I {} mv {} "$TEST_DIR/$CLASS_NAME"
done