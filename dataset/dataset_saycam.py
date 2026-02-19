import os
import json
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset

MODEL_NAME = 'ViT-H-14'  

class SAYCamDataset(Dataset):
    def __init__(self, metadata_path, image_dir, preprocess):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.data = []

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

        # Preprocess image
        try:
            image = self.preprocess(Image.open(image_path))
        except Exception:
            image = torch.zeros((3, 224, 224))

        # Tokenize text (OpenCLIP tokenizer)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        text_tokens = tokenizer([text])[0]

        return {
            "id": str(img_id),
            "image": image,
            "text_tokens": text_tokens,
            "raw_text": text,
            "image_path": image_path,     # helpful for BLIP stage
            "filename": img_filename      # helpful for output/debug
        }
