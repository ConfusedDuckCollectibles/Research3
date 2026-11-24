"""Posthoc formatting diagnostic; never replaces the prespecified exact match.

This diagnostic was chosen after inspecting the first corrected pilot. It
extracts a color only at the start of the generated answer, consistently for
every condition, to expose text outputs such as ' red. The light'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def analyze(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    colors = json.loads((directory / "dataset_manifest.json").read_text())["colors"]
    pattern = re.compile(r"^\s*(" + "|".join(re.escape(color) for color in colors) + r")\b", re.IGNORECASE)
    result = {
        "status": "posthoc_descriptive_formatting_diagnostic_not_primary_outcome",
        "definition": "Extract an allowed color only at the start of generated text, ignoring leading whitespace and case; require a word boundary. Additional words and punctuation do not invalidate this diagnostic.",
        "reason": "The frozen text baseline often generated the right initial color followed by a period and more words, failing prespecified full-answer exact match.",
        "warning": "Chosen after inspecting predictions; do not use as preregistered success evidence or replace the primary results.",
        "splits": {},
    }
    for split in sorted({row["split"] for row in rows}):
        result["splits"][split] = {}
        for condition in sorted({row["condition"] for row in rows}):
            selected = [row for row in rows if row["split"] == split and row["condition"] == condition]
            correct, covered = 0, 0
            for row in selected:
                match = pattern.match(row["generated"])
                covered += int(match is not None)
                correct += bool(match and match.group(1).casefold() == row["expected"].strip().casefold())
            result["splits"][split][condition] = {
                "first_color_correct": correct, "examples": len(selected),
                "first_color_accuracy": correct / len(selected),
                "leading_color_present": covered, "extraction_coverage": covered / len(selected),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/pretrained_pilot"))
    args = parser.parse_args()
    report = analyze(args.directory)
    (args.directory / "formatting_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
