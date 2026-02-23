#!/usr/bin/env python3
"""
Generate responses using the full MH-SIGNALS pipeline.

Single post:
  python scripts/generate.py --config configs/pipeline.yaml \
    --post "I can't focus before my exams and I'm panicking."

Batch (JSONL):
  python scripts/generate.py --config configs/pipeline.yaml \
    --input data/splits/test_gold.jsonl \
    --output data/splits/test_pred.jsonl

The pipeline automatically:
  1. Classifies intent + concern from the post text
  2. Retrieves KB snippets ranked by those classifications
  3. Generates a grounded, empathetic response
"""

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from mhsignals.pipeline import MHSignalsPipeline  # noqa: E402


def run_single(pipeline, post):
    """Process a single post and print the JSON result."""
    response = pipeline(post)
    print(response.to_json(indent=2))


def run_batch(pipeline, input_path, output_path, max_rows=0, start_row=1):
    """Process a JSONL file of posts through the pipeline."""
    n_ok = n_fail = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_no, line in enumerate(fin, 1):
            if line_no < start_row:
                continue
            line = line.strip()
            if not line:
                continue
            if max_rows and (n_ok + n_fail) >= max_rows:
                break

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON at line {line_no}: {e}", file=sys.stderr)
                n_fail += 1
                continue

            post = row.get("post", "")
            if not post:
                print(f"[WARN] No 'post' at line {line_no}", file=sys.stderr)
                n_fail += 1
                continue

            try:
                response = pipeline(post)
                pred = {
                    "post": post,
                    "predicted_intents": response.intents,
                    "predicted_concern": response.concern,
                    "crisis_level": response.crisis_level,
                    "reply": response.reply,
                    "citations": response.snippets,
                }
                n_ok += 1
            except Exception as e:
                pred = {
                    "post": post,
                    "predicted_intents": None,
                    "predicted_concern": None,
                    "reply": "",
                    "citations": [],
                    "error": f"{type(e).__name__}: {e}",
                }
                n_fail += 1
                print(f"[ERROR] Line {line_no}: {e}", file=sys.stderr)

            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            fout.flush()

            if (n_ok + n_fail) % 25 == 0:
                print(
                    f"[PROGRESS] {n_ok + n_fail} rows (ok={n_ok}, fail={n_fail})",
                    file=sys.stderr,
                )

    print(f"\nWrote {n_ok + n_fail} predictions to {output_path} (ok={n_ok}, fail={n_fail})")


def main():
    ap = argparse.ArgumentParser(
        description="Generate responses using the MH-SIGNALS pipeline."
    )
    ap.add_argument(
        "-c", "--config", required=True,
        help="Path to pipeline.yaml config.",
    )

    # Single post mode
    ap.add_argument("--post", default=None, help="Single post text to process.")

    # Batch mode
    ap.add_argument("--input", default=None, help="Input JSONL file for batch mode.")
    ap.add_argument("--output", default=None, help="Output JSONL file for batch mode.")
    ap.add_argument("--max_rows", type=int, default=0, help="Max rows to process (0=all).")
    ap.add_argument("--start_row", type=int, default=1, help="1-based start row for resume.")

    args = ap.parse_args()

    if not args.post and not args.input:
        ap.error("Provide either --post (single) or --input (batch).")

    print("Loading pipeline...", file=sys.stderr)
    pipeline = MHSignalsPipeline.from_config(args.config)
    print("Pipeline ready.", file=sys.stderr)

    if args.post:
        run_single(pipeline, args.post)
    else:
        if not args.output:
            ap.error("--output required for batch mode.")
        run_batch(pipeline, args.input, args.output, args.max_rows, args.start_row)

    pipeline.cleanup()


if __name__ == "__main__":
    main()
