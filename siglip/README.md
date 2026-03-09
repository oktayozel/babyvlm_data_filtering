# SigLIP 2 Image-Caption Scoring

Score image-caption pairs using SigLIP 2 and write per-record sigmoid match scores into a copy of the input JSON.

## Scripts

- **`compute_siglip_score.py`** — Loads a human_labels-style JSON, scores each image-caption pair with SigLIP 2, and writes siglip_score into a duplicate of the input JSON.
- **`data_loader.py`** — Shared data-loading module. 

## Usage

```bash
# Basic run (uses defaults)
python compute_siglip_score.py --input_json /path/to/human_labels.json

# Specify output directory and caption field
python compute_siglip_score.py \
  --input_json /path/to/human_labels.json \
  --output_dir ./siglip_scores \
  --caption_field human_caption

# Resume an interrupted run (automatic — just re-run the same command)
python compute_siglip_score.py --input_json /path/to/human_labels.json
```


