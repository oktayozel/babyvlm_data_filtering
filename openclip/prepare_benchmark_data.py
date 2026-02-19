"""
Prepare Benchmark Data
======================
Converts human_labels.json into the benchmark format expected by benchmark.py.

For each labeled image (excluding "reject" entries):
  - Uses the human caption as the true caption
  - Randomly samples 2 other captions as distractors
  - Randomizes the position of the true caption

Usage:
  python prepare_benchmark_data.py --input /path/to/human_labels.json --output benchmark_data.json
"""

import json
import argparse
import random


def prepare_benchmark_data(input_path, output_path, num_distractors=2, seed=42):
    random.seed(seed)
    
    # Load human labels
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} total entries from {input_path}")
    
    # Filter out "reject" entries — keep only items with real captions
    valid_entries = []
    rejected = 0
    for filename, info in data.items():
        caption = info["human_caption"].strip()
        if caption.lower() == "reject":
            rejected += 1
            continue
        valid_entries.append({
            "filename": filename,
            "caption": caption,
            "frame_path": info["frame_path"],
        })
    
    print(f"Valid entries (with captions): {len(valid_entries)}")
    print(f"Rejected entries: {rejected}")
    
    if len(valid_entries) < num_distractors + 1:
        print(f"Error: Need at least {num_distractors + 1} valid entries, got {len(valid_entries)}")
        return
    
    # Collect all unique captions for distractor sampling
    all_captions = list(set(entry["caption"] for entry in valid_entries))
    print(f"Unique captions: {len(all_captions)}")
    
    # Build benchmark data
    benchmark_data = []
    for entry in valid_entries:
        true_caption = entry["caption"]
        
        # Sample distractors (different from the true caption)
        distractor_pool = [c for c in all_captions if c != true_caption]
        if len(distractor_pool) < num_distractors:
            # Fallback: allow duplicate captions if pool is too small
            distractors = random.choices(distractor_pool, k=num_distractors)
        else:
            distractors = random.sample(distractor_pool, num_distractors)
        
        # Combine and shuffle — track where the true caption ends up
        captions = [true_caption] + distractors
        
        # Shuffle and track true index
        indices = list(range(len(captions)))
        random.shuffle(indices)
        shuffled_captions = [captions[i] for i in indices]
        true_caption_index = indices.index(0)  # 0 was the original position of true caption
        
        benchmark_data.append({
            "image_path": entry["frame_path"],
            "captions": shuffled_captions,
            "true_caption_index": true_caption_index,
        })
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\nSaved {len(benchmark_data)} benchmark items to {output_path}")
    
    # Print a few samples
    print("\n--- Sample entries ---")
    for item in benchmark_data[:3]:
        print(f"\nImage: {item['image_path']}")
        for i, cap in enumerate(item['captions']):
            marker = " ✓" if i == item['true_caption_index'] else ""
            print(f"  [{i}] {cap}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark data from human labels")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to human_labels.json")
    parser.add_argument("--output", type=str, default="benchmark_data.json",
                        help="Output path for benchmark JSON (default: benchmark_data.json)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    
    prepare_benchmark_data(args.input, args.output, seed=args.seed)
