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

# --- DATASET CLASS ---
class Dataset(Dataset):
    def __init__(self, metadata_path, image_dir, preprocess):
        self.image_dir = image_dir
        self.preprocess = preprocess
        
        # Load the actual metadata file
        # Assumes a JSON list of dicts: [{"image_id": "...", "caption": "..."}]
        try:
            with open(metadata_path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Adjust these keys based on your actual JSON structure!
        img_filename = item.get('filename', f"{item.get('image_id')}.jpg")
        text = item.get('caption', '')
        image_path = os.path.join(self.image_dir, img_filename)

        # Preprocess image
        try:
            image = self.preprocess(Image.open(image_path))
        except Exception as e:
            # Return a blank tensor if image fails (handle gracefully)
            image = torch.zeros((3, 224, 224)) 

        # Tokenize text
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        text_tokens = tokenizer([text])[0]

        return {
            "id": item.get('image_id', idx),
            "image": image,
            "text_tokens": text_tokens,
            "raw_text": text
        }

# --- MODE 1: THE TEST (Replicating your colleague's example) ---
def run_test_mode(model, preprocess):
    print("\n--- RUNNING TEST MODE ---")
    
    # Use the specific example path from your colleague
    test_img_path = "/projectnb/ivc-ml/ac25/BabyFM/Dataset/pretraining_dataset_raw/image/S_20140112_1426_04_180680_185280_frame_2.jpg"
    
    # The 3 candidate captions
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
            # Calculate features
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            # Normalize (Important for OpenCLIP!)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate probabilities (Softmax)
            # Logit scale is learned by the model, essentially a "temperature"
            logit_scale = model.logit_scale.exp()
            probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        print(f"\nImage: {test_img_path}")
        print("Probabilities:")
        for i, caption in enumerate(captions):
            print(f"{probs[0][i]:.4f} -> {caption}")
            
        print("\nSuccess! The highest probability should correspond to the most accurate caption.")

    except FileNotFoundError:
        print(f"Test image not found at {test_img_path}. Please check the path.")

# --- MODE 2: BATCH PROCESSING ---
def run_batch_mode(model, preprocess, args):
    print("\n--- RUNNING BATCH PROCESSING MODE ---")
    
    dataset = Dataset(args.metadata, args.image_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Filtering Data"):
            images = batch["image"].to(DEVICE)
            text_tokens = batch["text_tokens"].to(DEVICE)
            
            # Encode
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate Cosine Similarity (Dot Product)
            # We want the similarity between image[i] and text[i], so we just multiply element-wise and sum
            similarity_scores = (image_features * text_features).sum(dim=1)

            # Store results
            for i, score in enumerate(similarity_scores):
                results.append({
                    "image_id": str(batch["id"][i].item()) if isinstance(batch["id"][i], torch.Tensor) else batch["id"][i],
                    "score": score.item(),
                    "caption": batch["raw_text"][i]
                })
    
    # Save to JSON
    output_filename = "saycam_openclip_scores.json"
    print(f"Saving {len(results)} scores to {output_filename}...")
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCLIP Data Filtering for BabyVLM")
    parser.add_argument('--mode', type=str, choices=['test', 'batch'], required=True, help="Run a 'test' on one image or 'batch' process the dataset")
    parser.add_argument('--metadata', type=str, help="Path to the metadata JSON file (required for batch mode)")
    parser.add_argument('--image_dir', type=str, help="Directory containing images (required for batch mode)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for processing")

    args = parser.parse_args()

    # Load Model (done once for both modes)
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_SOURCE, device=DEVICE)
    model.eval()

    if args.mode == 'test':
        run_test_mode(model, preprocess)
    elif args.mode == 'batch':
        if not args.metadata or not args.image_dir:
            print("Error: --metadata and --image_dir are required for batch mode.")
            sys.exit(1)
        run_batch_mode(model, preprocess, args)