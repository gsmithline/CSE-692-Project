#!/usr/bin/env bash
#
# undo_organize_csvs.sh
#
# This script attempts to "undo" the folder structure created by organize_csvs.sh
# by moving all .csv files in subfolders back into the current directory.
# Then it removes any empty directories if they remain.

# 1) Move all CSV files from any subdirectory up to the current directory.
#    We'll use `find` commands to locate them rather than globstar.

while IFS= read -r -d '' file; do
  echo "Moving '$file' => '$PWD/'"
  mv "$file" "$PWD/"
done < <(find . -type f -name '*.csv' -not -path './*.csv' -print0)

# 2) Remove any now-empty subdirectories
find . -type d -empty -delete

echo "Done undoing the folder structure. All CSVs have been moved back to this directory."