#!/bin/zsh

for g in baseline hpa qmix; do
  for ext in csv json; do

    out="combined_${g}.${ext}"

    if [[ -f "$out" ]]; then
      echo "⚠️  $out already exists — skipping"
      continue
    fi

    echo "Creating $out"

    first_csv=1

    find . -type f -iname "*${g}*.${ext}" | while read f; do

      [[ "$f" == *combined_* ]] && continue

      echo "Appending $f"

      if [[ "$ext" == "csv" ]]; then
        if [[ $first_csv -eq 1 ]]; then
          cat "$f" >> "$out"
          first_csv=0
        else
          tail -n +2 "$f" >> "$out"
        fi
      else
        cat "$f" >> "$out"
        echo >> "$out"
      fi

    done

    echo "Done -> $out"
    echo

  done
done
