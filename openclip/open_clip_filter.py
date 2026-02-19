"""
OpenCLIP Data Filtering
=======================
Computes cosine similarity between images and captions using OpenCLIP models.
Can be used as a standalone CLI or imported by other scripts (e.g., benchmark.py).
"""

import torch
import open_clip
from PIL import Image
import json
import os
import argparse
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Defaults (used when running as CLI)
DEFAULT_MODEL = 'ViT-H-14'
DEFAULT_PRETRAINED = 'laion2b_s32b_b79k'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Shared Utilities ---

def load_model(model_name=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, device=None):
    """Load an OpenCLIP model, returns (model, preprocess, tokenizer)."""
    if device is None:
        device = DEVICE
    print(f"Loading {model_name} ({pretrained}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def compute_similarity(model, preprocess, tokenizer, image_path, captions, device=None):
    """
    Compute cosine similarity between a single image and one or more captions.
    
    Args:
        model: OpenCLIP model
        preprocess: image preprocessing transform
        tokenizer: text tokenizer
        image_path: path to image file
        captions: string or list of strings
        device: torch device
    
    Returns:
        list of float similarity scores (one per caption)
    """
    if device is None:
        device = DEVICE
    if isinstance(captions, str):
        captions = [captions]
    
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(captions).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
    
    return similarities.tolist() if len(captions) > 1 else [similarities.item()]


# --- Dataset for Batch Processing ---

class SAYCamDataset(Dataset):
    def __init__(self, metadata_path, image_dir, preprocess):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.data = []
        self.tokenizer = None  # Set externally after creation
        
        print(f"Loading metadata from {metadata_path}...")
        try:
            if metadata_path.endswith('.jsonl'):
                with open(metadata_path, 'r') as f:
                    self.data = [json.loads(line) for line in f]
            else:
                with open(metadata_path, 'r') as f:
                    self.data = json.load(f)
            
            if len(self.data) > 0:
                print("Sample entry:", self.data[0])
                if 'caption' not in self.data[0] and 'text' not in self.data[0]:
                    print("WARNING: Neither 'caption' nor 'text' keys found in metadata!")
        except Exception as e:
            print(f"Error loading metadata: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_id = item.get('image_id', str(idx))
        img_filename = item.get('filename', item.get('image_path', f"{img_id}.jpg"))
        text = item.get('caption', item.get('text', item.get('utterance', '')))
        
        image_path = os.path.join(self.image_dir, img_filename)

        try:
            image = self.preprocess(Image.open(image_path))
        except Exception as e:
            image = torch.zeros((3, 224, 224))

        text_tokens = self.tokenizer([text])[0]

        return {
            "id": str(img_id),
            "image": image,
            "text_tokens": text_tokens,
            "raw_text": text
        }


# --- MODE 1: TEST (Sanity Check) ---

def run_test_mode(model, preprocess, tokenizer):
    print("\n--- RUNNING TEST MODE ---")
    
    test_img_path = "/projectnb/ivc-ml/ac25/BabyFM/Dataset/pretraining_dataset_raw/image/S_20140112_1426_04_180680_185280_frame_2.jpg"
    captions = [
        "In a Book Reading Setting. Turning a page a Book with your hand.",
        "Being touched on an unknown object on your hand.",
        "Interacting with your own body "
    ]
    
    try:
        scores = compute_similarity(model, preprocess, tokenizer, test_img_path, captions)
        
        print(f"\nImage: {test_img_path}")
        for i, caption in enumerate(captions):
            print(f"Score: {scores[i]:.4f} -> {caption}")

    except Exception as e:
        print(f"Test failed: {e}")


# --- MODE 2: BATCH PROCESSING (Production) ---

def run_batch_mode(model, preprocess, tokenizer, args):
    print("\n--- RUNNING BATCH PROCESSING MODE ---")
    
    dataset = SAYCamDataset(args.metadata, args.image_dir, preprocess)
    dataset.tokenizer = tokenizer
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    output_filename = "saycam_openclip_scores.jsonl"
    print(f"Processing {len(dataset)} items...")
    print(f"Streaming results to {output_filename} (safe against crashes)...")
    
    with open(output_filename, 'w') as f:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Filtering"):
                images = batch["image"].to(DEVICE)
                text_tokens = batch["text_tokens"].to(DEVICE)
                
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity_scores = (image_features * text_features).sum(dim=1)

                for i, score in enumerate(similarity_scores):
                    record = {
                        "image_id": batch["id"][i],
                        "score": score.item(),
                        "caption": batch["raw_text"][i]
                    }
                    f.write(json.dumps(record) + "\n")
    
    print("Done!")


# --- CLI Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCLIP Data Filtering")
    parser.add_argument('--mode', type=str, choices=['test', 'batch'], required=True)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f"OpenCLIP model name (default: {DEFAULT_MODEL})")
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED,
                        help=f"Pretrained weights (default: {DEFAULT_PRETRAINED})")
    parser.add_argument('--metadata', type=str, help="Path to metadata JSON")
    parser.add_argument('--image_dir', type=str, help="Directory containing images")
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args.model, args.pretrained)

    if args.mode == 'test':
        run_test_mode(model, preprocess, tokenizer)
    elif args.mode == 'batch':
        if not args.metadata or not args.image_dir:
            print("Error: --metadata and --image_dir required for batch mode.")
            sys.exit(1)
        run_batch_mode(model, preprocess, tokenizer, args)
