#Adding json probabilities for gemini labels 

import os
os.environ["HF_HOME"] = "/projectnb/ivc-ml/visista/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/projectnb/ivc-ml/visista/.cache/huggingface"


import json
import argparse
from typing import Dict, Any, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

DEFAULT_INPUT_JSON = "/projectnb/ivc-ml/maxwh/code/labeling_effort/filter/gemini_labels.json"
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"



def load_qwen_vl(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return model, processor



def safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_done_keys(events_jsonl_path: str) -> set:
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
                pass
    return done



@torch.no_grad()
def compute_qwen_score(
    model,
    processor,
    device: str,
    image_path: str,
    caption: str,
) -> Optional[float]:
    """
    Returns pseudo-probability based on Qwen YES/NO judgment.
    """

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (224, 224), (0, 0, 0))

  
    prompt = (
        "Does this caption correctly describe the image? "
        "Answer ONLY with YES or NO.\n"
        f"Caption: {caption}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )

    output_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip().lower()

   
    if "yes" in output_text:
        return 1.0
    elif "no" in output_text:
        return 0.0
    else:
        # uncertain
        return 0.5



def write_snapshot_json(output_json_path: str, base_data: Dict[str, Any], updates: Dict[str, Any]):
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
        description="Compute Qwen-VL caption filtering scores."
    )
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)

    parser.add_argument("--output_json", type=str, default="gemini_labels_with_qwen_7B.json")
    parser.add_argument("--events_jsonl", type=str, default="gemini_labels_qwen_events_7B.jsonl")

    parser.add_argument("--snapshot_every", type=int, default=25)
    parser.add_argument("--skip_reject", action="store_true", default=True)
    parser.add_argument("--reject_token", type=str, default="reject")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_data = safe_load_json(args.input_json)
    print(f"Loaded {len(base_data)} entries")

    done_keys = load_done_keys(args.events_jsonl)

 
    model, processor = load_qwen_vl(args.model_id, device)

    updates: Dict[str, Dict[str, Any]] = {}
    processed_count = 0
    wrote_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.events_jsonl)), exist_ok=True)

    with open(args.events_jsonl, "a") as ef:
        for key, entry in tqdm(base_data.items(), desc="Scoring Qwen-VL"):
            if key in done_keys:
                continue

            caption = (entry.get("gemini_caption") or "").strip()
            frame_path = entry.get("frame_path")

            if not caption or not frame_path or not os.path.exists(frame_path):
                updates[key] = {"qwen_score": None, "keep": False}
                continue

            try:
                score = compute_qwen_score(
                    model, processor, device, frame_path, caption
                )

                keep = score >= 0.5

                ef.write(json.dumps({
                    "key": key,
                    "qwen_score": score,
                    "keep": keep
                }) + "\n")
                ef.flush()

                updates[key] = {"qwen_score": score, "keep": keep}
                wrote_count += 1

            except Exception as e:
                ef.write(json.dumps({
                    "key": key,
                    "error": "scoring_failed",
                    "exception": str(e)
                }) + "\n")
                ef.flush()

                updates[key] = {"qwen_score": None, "keep": False}

            processed_count += 1

            if args.snapshot_every > 0 and (processed_count % args.snapshot_every == 0):
                write_snapshot_json(args.output_json, base_data, updates)

    write_snapshot_json(args.output_json, base_data, updates)

    print("\nDone.")
    print(f"Processed: {processed_count}")
    print(f"Wrote: {wrote_count}")


if __name__ == "__main__":
    main()