#!/bin/bash
source_dir="dataset"
dest_dir="small_dataset"

for folder in "$source_dir"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        
        mkdir -p "$dest_dir/$folder_name"
        find "$folder" -maxdepth 1 -type f -name "*.jpg" | head -n 10 | xargs -I {} cp {} "$dest_dir/$folder_name"
    fi
done