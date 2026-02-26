"""
Score Human Labels
==================
Scores each image-caption pair in human_labels.json using ViT-H-14-quickgelu.
Adds a 'clip_score' field to each entry. Rejected entries get clip_score = -1.

Usage:
  python score_human_labels.py \
    --input /projectnb/ivc-ml/maxwh/code/labeling_effort/filter/human_labels.json \
    --output /projectnb/ivc-ml/oktayozel/babyvlm_data_filtering/scored_human_labels.json \
    --cache_dir /projectnb/ivc-ml/oktayozel/model_cache
"""

import json
import argparse
from tqdm import tqdm
from open_clip_filter import load_model, compute_similarity, DEVICE

MODEL_NAME = "ViT-H-14-quickgelu"
PRETRAINED = "dfn5b"


def main():
    parser = argparse.ArgumentParser(description="Score human labels with CLIP")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to human_labels.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save scored JSON (will NOT overwrite input)")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    # Safety check: don't overwrite the original
    if args.input == args.output:
        print("ERROR: --output must be different from --input!")
        return

    # Load human labels
    with open(args.input) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {args.input}")

    # Load model
    model, preprocess, tokenizer = load_model(MODEL_NAME, PRETRAINED, cache_dir=args.cache_dir)

    # Score each entry
    scored = 0
    rejected = 0
    errors = 0

    for filename, entry in tqdm(data.items(), desc="Scoring"):
        # Auto-detect caption key
        caption = entry.get("human_caption", entry.get("gemini_caption", "")).strip()

        # Skip rejected entries
        if caption.lower() == "reject":
            entry["clip_score"] = -1
            rejected += 1
            continue

        # Compute similarity
        try:
            scores = compute_similarity(
                model, preprocess, tokenizer,
                entry["frame_path"], caption
            )
            entry["clip_score"] = round(scores[0], 4)
            scored += 1
        except Exception as e:
            entry["clip_score"] = -2  # error marker
            errors += 1
            if errors <= 3:
                print(f"  Error on {filename}: {e}")

    # Save
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone!")
    print(f"  Scored: {scored}")
    print(f"  Rejected (set to -1): {rejected}")
    print(f"  Errors (set to -2): {errors}")
    print(f"  Saved to: {args.output}")

    # Quick stats on scores
    valid_scores = [e["clip_score"] for e in data.values() if e["clip_score"] > 0]
    if valid_scores:
        valid_scores.sort()
        print(f"\n  Score stats (non-rejected):")
        print(f"    Count: {len(valid_scores)}")
        print(f"    Min:   {min(valid_scores):.4f}")
        print(f"    Max:   {max(valid_scores):.4f}")
        print(f"    Mean:  {sum(valid_scores)/len(valid_scores):.4f}")
        print(f"    Median:{valid_scores[len(valid_scores)//2]:.4f}")


if __name__ == "__main__":
    main()
