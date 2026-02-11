import torch
import open_clip
from PIL import Image
import json
import os
import argparse
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

MODEL_NAME = 'ViT-H-14' 
PRETRAINED_SOURCE = 'laion2b_s32b_b79k'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- UPDATE 1: Renamed Class to avoid conflict ---
class SAYCamDataset(Dataset):
    def __init__(self, metadata_path, image_dir, preprocess):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.data = []
        
        # Load the metadata
        print(f"Loading metadata from {metadata_path}...")
        try:
            # Detect if it's a JSON list or JSON Lines file
            if metadata_path.endswith('.jsonl'):
                with open(metadata_path, 'r') as f:
                    self.data = [json.loads(line) for line in f]
            else:
                with open(metadata_path, 'r') as f:
                    self.data = json.load(f)
            
            # --- CRITICAL: CHECK KEYS ---
            if len(self.data) > 0:
                print("Sample entry:", self.data[0])
                # Check if 'caption' or 'text' exists
                if 'caption' not in self.data[0] and 'text' not in self.data[0]:
                    print("WARNING: Neither 'caption' nor 'text' keys found in metadata!")
        
        except Exception as e:
            print(f"Error loading metadata: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- UPDATE 2: Robust Key Handling ---
        # Adjust these if your JSON keys are different (e.g., 'image_path', 'utterance')
        img_id = item.get('image_id', str(idx))
        img_filename = item.get('filename', item.get('image_path', f"{img_id}.jpg"))
        
        # Try finding the text in common keys
        text = item.get('caption', item.get('text', item.get('utterance', '')))
        
        image_path = os.path.join(self.image_dir, img_filename)

        # Preprocess image
        try:
            image = self.preprocess(Image.open(image_path))
        except Exception as e:
            # print(f"Warning: Could not load {image_path}") # Optional: reduce spam
            image = torch.zeros((3, 224, 224)) 

        # Tokenize text
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        text_tokens = tokenizer([text])[0]

        return {
            "id": str(img_id),
            "image": image,
            "text_tokens": text_tokens,
            "raw_text": text
        }

# --- MODE 1: TEST (Sanity Check) ---
def run_test_mode(model, preprocess):
    print("\n--- RUNNING TEST MODE ---")
    
    test_img_path = "/projectnb/ivc-ml/ac25/BabyFM/Dataset/pretraining_dataset_raw/image/S_20140112_1426_04_180680_185280_frame_2.jpg"
    captions = [
        "In a Book Reading Setting. Turning a page a Book with your hand.", 
        "Being touched on an unknown object on your hand.", 
        "Interacting with your own body "
    ]
    
    try:
        image = preprocess(Image.open(test_img_path)).unsqueeze(0).to(DEVICE)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        text = tokenizer(captions).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        print(f"\nImage: {test_img_path}")
        for i, caption in enumerate(captions):
            print(f"Prob: {probs[0][i]:.4f} -> {caption}")

    except Exception as e:
        print(f"Test failed: {e}")

# --- MODE 2: BATCH PROCESSING (Production) ---
def run_batch_mode(model, preprocess, args):
    print("\n--- RUNNING BATCH PROCESSING MODE ---")
    
    # Initialize Dataset
    dataset = SAYCamDataset(args.metadata, args.image_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- UPDATE 3: Streaming Save (JSONL) ---
    output_filename = "saycam_openclip_scores.jsonl"
    print(f"Processing {len(dataset)} items...")
    print(f"Streaming results to {output_filename} (safe against crashes)...")
    
    # Open file in append mode ('a') or write mode ('w')
    with open(output_filename, 'w') as f:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Filtering"):
                images = batch["image"].to(DEVICE)
                text_tokens = batch["text_tokens"].to(DEVICE)
                
                # Encode & Normalize
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Dot Product for Similarity
                similarity_scores = (image_features * text_features).sum(dim=1)

                # Write batch results immediately
                for i, score in enumerate(similarity_scores):
                    record = {
                        "image_id": batch["id"][i],
                        "score": score.item(),
                        "caption": batch["raw_text"][i]
                    }
                    # Write one line per item
                    f.write(json.dumps(record) + "\n")
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCLIP Data Filtering")
    parser.add_argument('--mode', type=str, choices=['test', 'batch'], required=True)
    parser.add_argument('--metadata', type=str, help="Path to metadata JSON")
    parser.add_argument('--image_dir', type=str, help="Directory containing images")
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_SOURCE, device=DEVICE)
    model.eval()

    if args.mode == 'test':
        run_test_mode(model, preprocess)
    elif args.mode == 'batch':
        if not args.metadata or not args.image_dir:
            print("Error: --metadata and --image_dir required for batch mode.")
            sys.exit(1)
        run_batch_mode(model, preprocess, args)