# OpenCLIP Data Filtering & Benchmarking

Filter image-caption pairs by cosine similarity and benchmark top OpenCLIP models on caption-matching accuracy.

## Scripts

- **`open_clip_filter.py`** — Computes cosine similarity between images and captions. Supports test mode (single image) and batch mode (full dataset → JSONL output).
- **`benchmark.py`** — Evaluates top 5 OpenCLIP models on a caption-matching task: given an image and 3 captions, does the model rank the true caption highest?

## Usage

```bash
# Test mode (sanity check)
python open_clip_filter.py --mode test

# Use a different model
python open_clip_filter.py --mode test --model ViT-L-14 --pretrained laion2b_s32b_b82k

# Batch filter a dataset
python open_clip_filter.py --mode batch --metadata data.json --image_dir /path/to/images

# Benchmark top 5 models
python benchmark.py --data benchmark_data.json

# Benchmark only top 3, save detailed results
python benchmark.py --data benchmark_data.json --models 3 --output results.json
```

**Benchmark input format:**
```json
[
  {
    "image_path": "/path/to/image.jpg",
    "captions": ["true caption", "distractor 1", "distractor 2"],
    "true_caption_index": 0
  }
]
```

## Requirements

`torch`, `open_clip_torch`, `Pillow`, `tqdm`, `numpy`




# Generate benchmark data
python prepare_benchmark_data.py \
  --input /projectnb/ivc-ml/maxwh/code/labeling_effort/filter/human_labels.json \
  --output benchmark_data.json
# Then run the benchmark
python benchmark.py --data benchmark_data.json