#!/usr/bin/env bash
#
# organize_csvs.sh
# 
# A simple script to organize experiment CSV files by:
#  1) Date (MM_DD_YYYY)
#  2) Model/LLM type (anthropic, openai, llama, gemini, etc.)
#  3) Prompt style or game configuration (basic, maximize_value, outside_offer, cot, etc.)
#
# For each CSV, it will attempt to parse these fields from the filename and then
# create a directory structure like:
#   ./YYYY_MM_DD/LLM_TYPE/PROMPT_STYLE/
# and move the file there.
#
# Usage:
#   1. Make this script executable:
#      chmod +x organize_csvs.sh
#   2. Run it inside the folder with your CSV files:
#      ./organize_csvs.sh
#
# Notes:
#   - This script does best-effort pattern matching and may need to be adjusted
#     if your naming conventions differ.
#   - Always review the resulting directories to ensure correctness.

shopt -s nocaseglob  # for case-insensitive glob (in case you have CSV vs csv)
shopt -s extglob     # for extended pattern matching

for file in *.csv; do
  # If no matches (e.g., no CSV files), skip
  [ -e "$file" ] || continue

  # Capture the date component in the filename, e.g. "12_10_2024"
  # The pattern looks for one or two digits for month, an underscore,
  # two digits for day, another underscore, and four digits for year.
  datePart="$(echo "$file" | grep -oE '[0-1]?[0-9]_[0-3][0-9]_[0-9]{4}')"
  
  # If we didn't find a date, just default to "unknown_date"
  if [ -z "$datePart" ]; then
    datePart="unknown_date"
  fi

  # LLM/Model type. We’ll try to detect common patterns:
  #   - anthropic
  #   - openai
  #   - llama
  #   - gemini
  #   - bargain
  #   - etc.
  # If not matched, default to "misc_model".
  model="misc_model"
  if   [[ "$file" =~ anthropic ]]; then
    model="anthropic"
  elif [[ "$file" =~ openai ]]; then
    model="openai"
  elif [[ "$file" =~ llama ]]; then
    model="llama"
  elif [[ "$file" =~ gemini ]]; then
    model="gemini"
  elif [[ "$file" =~ bargain ]]; then
    model="bargain"
  fi

  # Prompt style or game config. We'll look for patterns:
  #   - maximize_value_outside_offer
  #   - maximize_value
  #   - outside_offer
  #   - basic
  #   - no_cot or cot
  #   - etc.
  # We'll guess based on the filename. If multiple keywords are present,
  # we'll concatenate them in an underscore.
  # Adjust as desired if you have more/different naming conventions.
  style="misc_style"
  # Build an array of possible style tags we might see:
  declare -a style_tags=("maximize_value_outside_offer" "maximize_value" "outside_offer" "basic" "bargain" "example" "no_cot" "cot")
  # We'll loop through these in a meaningful order so, for example,
  # "maximize_value_outside_offer" (the longest pattern) is tested first.
  # That way we don’t immediately stop at "maximize_value".
  
  matched_styles=""
  for s in "${style_tags[@]}"; do
    if [[ "$file" =~ $s ]]; then
      if [ -z "$matched_styles" ]; then
        matched_styles="$s"
      else
        # If multiple are matched, just underscore them together
        matched_styles="${matched_styles}_$s"
      fi
    fi
  done
  if [ -n "$matched_styles" ]; then
    style="$matched_styles"
  fi

  # We also want year_month_day for the top-level directory for readability:
  # transform e.g. "12_10_2024" -> "2024_12_10"
  # This depends on the exact naming structure, 
  # so be sure your files are always MM_DD_YYYY in that order.
  # We'll parse them out:
  month="$(echo "$datePart" | cut -d'_' -f1)"
  day="$(echo "$datePart" | cut -d'_' -f2)"
  year="$(echo "$datePart" | cut -d'_' -f3)"
  
  # Reformat:
  dateReformat="${year}_${month}_${day}"

  # Create the directory structure
  # e.g. ./2024_12_10/anthropic/maximize_value_outside_offer/
  outdir="./${dateReformat}/${model}/${style}"
  mkdir -p "$outdir"

  # Move the file
  mv "$file" "$outdir/"
  echo "Moved '$file' -> '$outdir/'"
done

echo "Done organizing!"