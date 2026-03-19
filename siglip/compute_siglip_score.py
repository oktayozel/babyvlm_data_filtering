import os
import json
import shutil
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from load_json_data import load_records


DEFAULT_INPUT_JSON = "/projectnb/ivc-ml/maxwh/code/labeling_effort/filter/human_labels.json"
DEFAULT_MODEL_ID = "google/siglip2-so400m-patch14-384"
DEFAULT_OUTPUT_DIR = "./siglip_scores"
DEFAULT_OUTPUT_JSON = "data_with_siglip_scores.json"
DEFAULT_EVENTS_JSONL = "siglip_events.jsonl"
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_siglip2(model_id: str) -> Tuple[Any, Any, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor, device


def _load_image(image_path: str) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (384, 384), (0, 0, 0))


def compute_siglip2_scores_batch(
    model,
    processor,
    device: str,
    image_paths: List[str],
    captions: List[str],
) -> List[float]:
    """
    Compute SigLIP 2 sigmoid match scores for a batch of (image, caption) pairs.

    Each pair (image_paths[i], captions[i]) is scored independently using the
    diagonal of the N×N image-text similarity matrix, which is valid because
    SigLIP uses a per-pair sigmoid loss (not softmax).

    Returns a list of float scores in [0, 1], one per pair.
    """
    images = [_load_image(p) for p in image_paths]

    inputs = processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    ).to(device)

    with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
        outputs = model(**inputs)
        # logits_per_image: [N, N]; diagonal gives paired (img_i, text_i) logits
        scores = torch.sigmoid(outputs.logits_per_image.diagonal()).cpu().tolist()

    return scores


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def duplicate_json(src: str, dst: str):
    """Copy the input JSON to the output path (only if dst doesn't already exist)."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"Duplicated input JSON to: {dst}")
    else:
        print(f"Output JSON already exists, skipping copy: {dst}")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]):
    """Atomic write via temp file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_done_keys(events_jsonl: str) -> set:
    """Read already-processed keys from the incremental JSONL log for resume support."""
    done = set()
    if not os.path.exists(events_jsonl):
        return done
    with open(events_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("key")
                if k:
                    done.add(k)
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score image-caption pairs with SigLIP 2 in batches and write scores to a copy of the input JSON."
    )
    parser.add_argument("--input_json",     type=str, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--model_id",       type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir",     type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_json",    type=str, default=DEFAULT_OUTPUT_JSON,
                        help="Filename (not full path) for the scored JSON copy.")
    parser.add_argument("--events_jsonl",   type=str, default=DEFAULT_EVENTS_JSONL,
                        help="Filename (not full path) for the incremental JSONL event log.")
    parser.add_argument("--caption_field",  type=str, default=None,
                        help="Which caption field to use: 'human_caption' or 'gemini_caption'. "
                             "If not set, tries both in order.")
    parser.add_argument("--batch_size",     type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of (image, caption) pairs to score per forward pass.")
    parser.add_argument("--snapshot_every", type=int, default=100,
                        help="Write a full JSON snapshot every N batches (0 = disable).")
    parser.add_argument("--skip_token",     type=str, default="reject",
                        help="Caption value treated as rejected. Set to '' to disable.")
    args = parser.parse_args()

    output_json_path  = os.path.join(args.output_dir, args.output_json)
    events_jsonl_path = os.path.join(args.output_dir, args.events_jsonl)
    skip_token        = args.skip_token or None

    # 1. Duplicate input JSON → output path (skip if already exists)
    duplicate_json(args.input_json, output_json_path)

    # 2. Load the live output JSON (may already have partial scores from a previous run)
    output_data = load_json(output_json_path)

    # 3. Load records via data loader
    records = load_records(
        args.input_json,
        caption_field=args.caption_field,
        skip_token=skip_token,
        require_image_exists=False,
    )
    print(f"Loaded {len(records)} records from: {args.input_json}")

    # 4. Resume: skip keys already logged
    done_keys = load_done_keys(events_jsonl_path)
    if done_keys:
        print(f"Resuming — {len(done_keys)} keys already processed.")

    # 5. Load model
    print(f"Loading SigLIP 2 model: {args.model_id}")
    model, processor, device = load_siglip2(args.model_id)

    # 6. Partition pending records into null (no score) vs scoreable
    null_records: List[Tuple[Any, str]] = []   # (record, reason)
    score_records: List[Any] = []

    for record in records:
        if record.key in done_keys:
            continue
        if record.caption is None:
            null_records.append((record, "no_valid_caption"))
        elif not record.image_path or not os.path.exists(record.image_path):
            null_records.append((record, "missing_image"))
        else:
            score_records.append(record)

    print(f"  {len(null_records)} records skipped (no caption / missing image)")
    print(f"  {len(score_records)} records to score in batches of {args.batch_size}")

    # 7. Score
    os.makedirs(args.output_dir, exist_ok=True)
    processed = 0
    batches_done = 0

    with open(events_jsonl_path, "a") as ef:

        # Write null records first
        for record, reason in null_records:
            output_data[record.key]["siglip_score"] = None
            ef.write(json.dumps({"key": record.key, "siglip_score": None, "reason": reason}) + "\n")
            processed += 1
        ef.flush()

        # Process scoreable records in batches
        n_batches = (len(score_records) + args.batch_size - 1) // args.batch_size
        for batch_idx in tqdm(range(n_batches), desc="Scoring SigLIP 2 (batches)"):
            start = batch_idx * args.batch_size
            batch = score_records[start: start + args.batch_size]

            image_paths = [r.image_path for r in batch]
            captions    = [r.caption    for r in batch]

            try:
                scores = compute_siglip2_scores_batch(model, processor, device, image_paths, captions)
                for record, score in zip(batch, scores):
                    output_data[record.key]["siglip_score"] = score
                    ef.write(json.dumps({"key": record.key, "siglip_score": score}) + "\n")
            except Exception as e:
                # Fall back to null for the whole batch; log the error
                err_str = str(e)
                for record in batch:
                    output_data[record.key]["siglip_score"] = None
                    ef.write(json.dumps({"key": record.key, "siglip_score": None, "error": err_str}) + "\n")

            ef.flush()
            processed += len(batch)
            batches_done += 1

            if args.snapshot_every > 0 and batches_done % args.snapshot_every == 0:
                write_json(output_json_path, output_data)

    # 8. Final write
    write_json(output_json_path, output_data)

    print("\nDone.")
    print(f"Scored JSON written to : {os.path.abspath(output_json_path)}")
    print(f"Event log written to   : {os.path.abspath(events_jsonl_path)}")
    print(f"Records processed      : {processed}")


if __name__ == "__main__":
    main()
