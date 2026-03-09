import os
import json
import shutil
import argparse
from typing import Any, Dict, Optional

from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from data_loader import load_records


DEFAULT_INPUT_JSON = "/projectnb/ivc-ml/maxwh/code/labeling_effort/filter/human_labels.json"
DEFAULT_MODEL_ID = "google/siglip2-so400m-patch14-384"
DEFAULT_OUTPUT_DIR = "./siglip_scores"
DEFAULT_OUTPUT_JSON = "data_with_siglip_scores.json"
DEFAULT_EVENTS_JSONL = "siglip_events.jsonl"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_siglip2(model_id: str):
    return pipeline(model=model_id, task="zero-shot-image-classification")


def compute_siglip2_score(pipe, image_path: str, caption: str) -> Optional[float]:
    """
    Returns SigLIP 2 sigmoid match score for a single (image, caption) pair.
    Score near 1.0 = strong match, near 0.0 = no match.
    Single-caption scoring is valid due to SigLIP's sigmoid (not softmax) loss.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (384, 384), (0, 0, 0))

    outputs = pipe(image, candidate_labels=[caption])
    return outputs[0]["score"]


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
        description="Score image-caption pairs with SigLIP 2 and write scores to a copy of the input JSON."
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
    parser.add_argument("--snapshot_every", type=int, default=25,
                        help="Write a full JSON snapshot every N processed items.")
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

    # 3. Load records via data_loader
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
    pipe = load_siglip2(args.model_id)

    # 6. Score loop
    processed = 0
    os.makedirs(args.output_dir, exist_ok=True)

    with open(events_jsonl_path, "a") as ef:
        for record in tqdm(records, desc="Scoring SigLIP 2"):
            if record.key in done_keys:
                continue

            event: Dict[str, Any] = {"key": record.key}

            # Rejected / empty caption → siglip_score: null
            if record.caption is None:
                output_data[record.key]["siglip_score"] = None
                event.update({"siglip_score": None, "reason": "no_valid_caption"})

            # Missing image → siglip_score: null
            elif not record.image_path or not os.path.exists(record.image_path):
                output_data[record.key]["siglip_score"] = None
                event.update({"siglip_score": None, "reason": "missing_image"})

            # Score
            else:
                try:
                    score = compute_siglip2_score(pipe, record.image_path, record.caption)
                    output_data[record.key]["siglip_score"] = score
                    event["siglip_score"] = score
                except Exception as e:
                    output_data[record.key]["siglip_score"] = None
                    event.update({"siglip_score": None, "error": str(e)})

            ef.write(json.dumps(event) + "\n")
            ef.flush()
            processed += 1

            if args.snapshot_every > 0 and processed % args.snapshot_every == 0:
                write_json(output_json_path, output_data)

    # 7. Final write
    write_json(output_json_path, output_data)

    print("\nDone.")
    print(f"Scored JSON written to : {os.path.abspath(output_json_path)}")
    print(f"Event log written to   : {os.path.abspath(events_jsonl_path)}")
    print(f"Records processed      : {processed}")


if __name__ == "__main__":
    main()