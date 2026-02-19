import argparse
import sys
import torch
import open_clip
from torch.utils.data import DataLoader

from dataset.dataset_saycam import SAYCamDataset
from filters.learned_filter import run_test_mode_learned_filter, run_batch_mode_learned_filter

MODEL_NAME = "ViT-H-14"
PRETRAINED_SOURCE = "laion2b_s32b_b79k"


def main():
    parser = argparse.ArgumentParser(description="SAYCam BLIP-ITM Data Filtering")
    parser.add_argument("--mode", type=str, choices=["test", "batch"], required=True)

    # batch inputs
    parser.add_argument("--metadata", type=str, help="Path to metadata JSON/JSONL")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # outputs (batch)
    parser.add_argument("--scores_output", type=str, default="saycam_blip_itm_scores.jsonl",
                        help="Where to write per-sample scores (jsonl).")
    parser.add_argument("--filtered_metadata_output", type=str, default="saycam_metadata_with_keep.jsonl",
                        help="Copy of original metadata with keep + itm_prob added (jsonl or json).")

    # BLIP ITM options
    parser.add_argument("--blip_model_id", type=str, default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--itm_threshold", type=float, default=0.7)

    # test options
    parser.add_argument("--test_img_path", type=str, default="/projectnb/ivc-ml/ac25/BabyFM/Dataset/pretraining_dataset_raw/image/S_20140112_1426_04_180680_185280_frame_2.jpg")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.mode == "test":
        run_test_mode_learned_filter(
            device=device,
            model_id=args.blip_model_id,
            test_img_path=args.test_img_path,
            threshold=args.itm_threshold,
        )
        return

    # batch mode checks
    if not args.metadata or not args.image_dir:
        print("Error: --metadata and --image_dir are required for batch mode.")
        sys.exit(1)

    # Dataset uses OpenCLIP preprocess (even though we don't use CLIP for scoring)
    print(f"Loading OpenCLIP preprocess for dataset: {MODEL_NAME} ({PRETRAINED_SOURCE})")
    _, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_SOURCE, device=device
    )

    dataset = SAYCamDataset(args.metadata, args.image_dir, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,              # IMPORTANT: must match metadata order
        num_workers=args.num_workers,
    )

    run_batch_mode_learned_filter(
        dataloader=dataloader,
        metadata_path=args.metadata,
        device=device,
        scores_output_path=args.scores_output,
        filtered_metadata_output_path=args.filtered_metadata_output,
        model_id=args.blip_model_id,
        threshold=args.itm_threshold,
    )


if __name__ == "__main__":
    main()
