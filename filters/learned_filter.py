import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch.nn.functional as F


def load_blip_itm(model_id: str, device: str):
    """
    Loads BLIP ITM model + processor using the exact HF style from the model card.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = BlipForImageTextRetrieval.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def _load_metadata(metadata_path: str):
    if metadata_path.endswith(".jsonl"):
        with open(metadata_path, "r") as f:
            return [json.loads(line) for line in f]
    else:
        with open(metadata_path, "r") as f:
            return json.load(f)


def _write_metadata(metadata, out_path: str):
    if out_path.endswith(".jsonl"):
        with open(out_path, "w") as f:
            for row in metadata:
                f.write(json.dumps(row) + "\n")
    else:
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)


def run_test_mode_learned_filter(
    device: str,
    model_id: str = "Salesforce/blip-itm-base-coco",
    test_img_path: str = "/projectnb/ivc-ml/ac25/BabyFM/Dataset/pretraining_dataset_raw/image/S_20140112_1426_04_180680_185280_frame_2.jpg",
    threshold: float | None = 0.7,
):
    """
    Test mode:
      - runs ITM on one image with multiple candidate captions
      - prints itm_prob and optional keep/drop using threshold
    """
    print("\n--- RUNNING BLIP-ITM TEST MODE ---")

    captions = [
        "In a Book Reading Setting. Turning a page a Book with your hand.",
        "Being touched on an unknown object on your hand.",
        "Interacting with your own body",
        "Balloon floating in the sky",
        "Cars crashing on the highway"
    ]

    model, processor = load_blip_itm(model_id, device)

    image = Image.open(test_img_path).convert("RGB")
    inputs = processor(
        images=[image] * len(captions),
        text=captions,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():

        out = model(**inputs)
        logits = out.itm_score  # <-- key change

        if logits.ndim == 2 and logits.size(-1) == 2:
            probs = F.softmax(logits, dim=-1)[:, 1]
        else:
            probs = torch.sigmoid(logits.squeeze(-1))

    print(f"\nImage: {test_img_path}")
    if threshold is not None:
        print(f"Threshold: {threshold}  (keep if itm_prob >= threshold)\n")
    else:
        print("Threshold: None (no keep/drop decision)\n")

    for p, c in zip(probs, captions):
        if threshold is None:
            print(f"ITM Prob: {p:.4f} -> {c}")
        else:
            decision = "KEEP" if p >= threshold else "DROP"
            print(f"ITM Prob: {p:.4f} [{decision}] -> {c}")


def run_batch_mode_learned_filter(
    dataloader,
    metadata_path: str,
    device: str,
    scores_output_path: str,
    filtered_metadata_output_path: str,
    model_id: str = "Salesforce/blip-itm-base-coco",
    threshold: float = 0.7,
):
    """
    Batch mode:
      - loads original metadata (does NOT modify it on disk)
      - computes itm_prob for each pair (in dataloader order)
      - writes:
          (1) scores_output_path: jsonl with {image_id, itm_prob, keep}
          (2) filtered_metadata_output_path: copy of original metadata with keep + itm_prob added

    IMPORTANT:
      This assumes dataloader iterates dataset in the SAME order as the loaded metadata list.
      (shuffle=False in DataLoader)
    """
    print("\n--- RUNNING BLIP-ITM BATCH MODE ---")
    print(f"Checkpoint: {model_id}")
    print(f"Threshold:  {threshold} (keep if itm_prob >= threshold)")

    model, processor = load_blip_itm(model_id, device)

    # Load metadata into memory so we can write a modified copy
    original_data = _load_metadata(metadata_path)
    print(f"Loaded metadata entries: {len(original_data)}")

    # We will update a COPY in memory
    updated_data = list(original_data)

    kept = 0
    total = 0
    meta_idx = 0

    with open(scores_output_path, "w") as sf:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Filtering(BLIP-ITM)"):
                image_paths = list(batch["image_path"])
                captions = list(batch["raw_text"])
                ids = list(batch["id"])

                # Load PIL images (BLIP expects PIL RGB)
                images_pil = []
                for p in image_paths:
                    try:
                        images_pil.append(Image.open(p).convert("RGB"))
                    except Exception:
                        images_pil.append(Image.new("RGB", (224, 224), (0, 0, 0)))

                inputs = processor(
                    images=images_pil,
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                out = model(**inputs)
                logits = out.itm_score 

                if logits.ndim == 2 and logits.size(-1) == 2:
                    probs = F.softmax(logits, dim=-1)[:, 1]
                else:
                    probs = torch.sigmoid(logits.squeeze(-1))



                bsz = len(ids)
                for i in range(bsz):
                    prob_i = float(probs[i].item())
                    keep_i = prob_i >= threshold

                    # (1) write scores JSONL
                    sf.write(json.dumps({
                        "image_id": ids[i],
                        "itm_prob": prob_i,
                        "keep": keep_i
                    }) + "\n")

                    # (2) update metadata copy (by order)
                    if meta_idx < len(updated_data):
                        updated_data[meta_idx]["itm_prob"] = prob_i
                        updated_data[meta_idx]["keep"] = keep_i
                    meta_idx += 1

                    total += 1
                    kept += int(keep_i)

    # Safety check
    if total != len(updated_data):
        print(f"WARNING: processed {total} items but metadata has {len(updated_data)} entries.")
        print("This usually means dataset order != metadata order, or some items were skipped.")

    # Write updated metadata copy
    _write_metadata(updated_data, filtered_metadata_output_path)

    print("\n--- SUMMARY ---")
    print(f"Total processed: {total}")
    print(f"Kept: {kept} ({(kept/total*100 if total>0 else 0):.2f}%)")
    print(f"Dropped: {total-kept} ({((total-kept)/total*100 if total>0 else 0):.2f}%)")
    print(f"Scores saved to: {scores_output_path}")
    print(f"Filtered metadata copy saved to: {filtered_metadata_output_path}")
    print("Done!")
