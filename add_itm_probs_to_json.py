import os
import json
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForImageTextRetrieval


DEFAULT_INPUT_JSON = "/projectnb/ivc-ml/maxwh/code/labeling_effort/filter/human_labels.json"
DEFAULT_MODEL_ID = "Salesforce/blip-itm-base-coco"


def load_blip_itm(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = BlipForImageTextRetrieval.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_done_keys(events_jsonl_path: str) -> set:
    """
    Read previously processed keys from the incremental jsonl log, so we can resume.
    Each line is a json object like {"key": "...", "itm_prob": ..., ...}
    """
    done = set()
    if not os.path.exists(events_jsonl_path):
        return done
    with open(events_jsonl_path, "r") as f:
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
                # ignore malformed lines
                pass
    return done


@torch.no_grad()
def compute_itm_prob(
    model,
    processor,
    device: str,
    image_path: str,
    caption: str,
) -> Optional[float]:
    """
    Returns ITM probability for (image, caption).
    If image cannot be loaded, uses a black image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224), (0, 0, 0))

    inputs = processor(
        images=img,
        text=caption,
        return_tensors="pt",
        padding=True,
    ).to(device)

    out = model(**inputs)
    logits = out.itm_score  # HF BLIP-ITM output field

    # Handle either [B,2] or [B] / [B,1]
    if logits.ndim == 2 and logits.size(-1) == 2:
        prob = F.softmax(logits, dim=-1)[:, 1]
    else:
        prob = torch.sigmoid(logits.squeeze(-1))

    return float(prob.item())


def write_snapshot_json(output_json_path: str, base_data: Dict[str, Any], updates: Dict[str, Any]):
    """
    Writes a full JSON copy (snapshot) that merges the original base_data with updates.
    """
    merged = dict(base_data)
    for k, upd in updates.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(upd, dict):
            merged[k] = {**merged[k], **upd}
        else:
            merged[k] = upd

    tmp_path = output_json_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp_path, output_json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLIP-ITM probabilities for human_labels.json and write to a COPY incrementally."
    )
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)

    # outputs in your code directory (current working directory by default)
    parser.add_argument("--output_json", type=str, default="human_labels_with_itm_probs.json")
    parser.add_argument("--events_jsonl", type=str, default="human_labels_itm_events.jsonl")

    # behavior
    parser.add_argument("--snapshot_every", type=int, default=25,
                        help="Write full JSON snapshot every N processed items.")
    parser.add_argument("--skip_reject", action="store_true", default=True,
                        help="Skip items where human_caption == 'reject' (default True).")
    parser.add_argument("--reject_token", type=str, default="reject")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_data = safe_load_json(args.input_json)
    print(f"Loaded {len(base_data)} entries from: {args.input_json}")

    # Resume support: don't recompute keys already logged
    done_keys = load_done_keys(args.events_jsonl)
    if done_keys:
        print(f"Found {len(done_keys)} already-processed keys in: {args.events_jsonl} (resume mode)")

    model, processor = load_blip_itm(args.model_id, device)

    # Keep updates in memory for periodic snapshots
    updates: Dict[str, Dict[str, Any]] = {}

    processed_count = 0
    wrote_count = 0

    # Open in append mode to write one-by-one
    os.makedirs(os.path.dirname(os.path.abspath(args.events_jsonl)), exist_ok=True)
    with open(args.events_jsonl, "a") as ef:
        for key, entry in tqdm(base_data.items(), desc="Scoring BLIP-ITM"):
            if key in done_keys:
                continue

            if not isinstance(entry, dict):
                # log as error and continue
                ef.write(json.dumps({"key": key, "error": "entry_not_dict"}) + "\n")
                ef.flush()
                continue

            caption = (entry.get("human_caption") or "").strip()
            frame_path = entry.get("frame_path")

            if args.skip_reject and caption.lower() == args.reject_token.lower():
                # mark explicitly skipped in events
                ef.write(json.dumps({
                    "key": key,
                    "skipped": True,
                    "reason": "reject_caption",
                    "human_caption": caption
                }) + "\n")
                ef.flush()

                updates[key] = {"itm_prob": None, "keep": False, "skipped": True}
                processed_count += 1
                wrote_count += 1
                continue

            if not caption:
                ef.write(json.dumps({
                    "key": key,
                    "skipped": True,
                    "reason": "empty_caption"
                }) + "\n")
                ef.flush()

                updates[key] = {"itm_prob": None, "keep": False, "skipped": True}
                processed_count += 1
                wrote_count += 1
                continue

            if not frame_path or not os.path.exists(frame_path):
                ef.write(json.dumps({
                    "key": key,
                    "error": "missing_frame_path",
                    "frame_path": frame_path
                }) + "\n")
                ef.flush()

                updates[key] = {"itm_prob": None, "keep": False, "error": "missing_frame_path"}
                processed_count += 1
                wrote_count += 1
                continue

            try:
                itm_prob = compute_itm_prob(model, processor, device, frame_path, caption)
                # for now: keep=True if prob >= 0.7 (same default you used)
                keep = itm_prob >= 0.7

                # Write ONE line immediately (incremental)
                ef.write(json.dumps({
                    "key": key,
                    "itm_prob": itm_prob,
                    "keep": keep
                }) + "\n")
                ef.flush()

                updates[key] = {"itm_prob": itm_prob, "keep": keep}
                wrote_count += 1

            except Exception as e:
                ef.write(json.dumps({
                    "key": key,
                    "error": "scoring_failed",
                    "exception": str(e)
                }) + "\n")
                ef.flush()

                updates[key] = {"itm_prob": None, "keep": False, "error": "scoring_failed"}

            processed_count += 1

            # Periodically write a full JSON snapshot COPY (safe overwrite)
            if args.snapshot_every > 0 and (processed_count % args.snapshot_every == 0):
                write_snapshot_json(args.output_json, base_data, updates)

    # Final snapshot
    write_snapshot_json(args.output_json, base_data, updates)

    print("\nDone.")
    print(f"Incremental events written to: {os.path.abspath(args.events_jsonl)}")
    print(f"Final JSON copy written to:     {os.path.abspath(args.output_json)}")
    print(f"New/processed this run: {processed_count}")
    print(f"Lines appended this run: {wrote_count}")


if __name__ == "__main__":
    main()
