import json
import torch
from tqdm import tqdm

def run_openclip_test(model, preprocess, device, model_name, test_img_path, captions, open_clip):
    print("\n--- RUNNING OPENCLIP TEST MODE ---")

    from PIL import Image
    tokenizer = open_clip.get_tokenizer(model_name)

    image = preprocess(Image.open(test_img_path)).unsqueeze(0).to(device)
    text = tokenizer(captions).to(device)

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


def run_openclip_batch(model, dataloader, device, output_filename):
    print("\n--- RUNNING OPENCLIP BATCH MODE ---")
    print(f"Streaming results to {output_filename}...")

    with open(output_filename, 'w') as f:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Filtering(OpenCLIP)"):
                images = batch["image"].to(device)
                text_tokens = batch["text_tokens"].to(device)

                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity_scores = (image_features * text_features).sum(dim=1)

                for i, score in enumerate(similarity_scores):
                    record = {
                        "image_id": batch["id"][i],
                        "score": float(score.item()),
                        "caption": batch["raw_text"][i],
                        "filename": batch.get("filename", [""] * len(similarity_scores))[i],
                        "image_path": batch.get("image_path", [""] * len(similarity_scores))[i],
                    }
                    f.write(json.dumps(record) + "\n")

    print("Done!")
