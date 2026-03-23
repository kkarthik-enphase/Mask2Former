"""
Mask2Former Fine-tuning for Roof Facet Instance Segmentation
=============================================================
Dataset format:
    img_folder/   - 1024x1024 RGB images (.jpg)
    gt_folder/    - COCO RLE JSON files (one per image, same stem name)

RLE JSON format:
    {
        "image": {"file_name": "xxx.jpg", "height": 1024, "width": 1024, ...},
        "annotations": [
            {"id": 1, "bbox": [...], "area": ...,
             "segmentation": {"size": [H, W], "counts": "<RLE string>"}},
            ...
        ]
    }

Usage:
    Train:
        python train_mask2former.py --mode train \
            --img_folder /mnt/harddrive/data/dataset/sam2_format/rgb_resized \
            --gt_folder  /mnt/harddrive/data/dataset/sam2_format/gt_facetmasks \
            --output_dir ./checkpoints \
            --epochs 50 --batch_size 2

    Inference:
        python train_mask2former.py --mode infer \
            --checkpoint ./checkpoints/epoch_10 \
            --input /path/to/image.jpg \
            --output result.png
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# RLE utilities (pycocotools-compatible, no pycocotools dependency)
# ─────────────────────────────────────────────────────────────────────────────

def rle_decode(rle: dict) -> np.ndarray:
    """Decode COCO RLE to binary mask (H x W)."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        # compressed RLE string — use pycocotools if available, else decode manually
        try:
            from pycocotools import mask as coco_mask
            mask = coco_mask.decode(rle).astype(bool)
            return mask
        except ImportError:
            pass
        # manual decode of compressed RLE
        import re
        counts = _decompress_rle(counts, h * w)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, cnt in enumerate(counts):
        if i % 2 == 1:
            flat[pos:pos + cnt] = 1
        pos += cnt
    return flat.reshape((h, w), order="F").astype(bool)


def _decompress_rle(s: str, n: int):
    """Decompress COCO compressed RLE string to list of counts."""
    counts = []
    m = 0
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            p += 1
            more = c > 31
            x |= (c & 0x1f) << (5 * k)
            k += 1
        if x & 1:
            x = ~x
        x >>= 1
        if len(counts) > 0:
            x += counts[-1] + (1 if len(counts) % 2 == 0 else 0)
            # actually just accumulate
        counts.append(x)
    # rebuild absolute counts
    return counts


def load_annotations(json_path: str):
    """Load COCO RLE JSON and return list of binary masks."""
    with open(json_path) as f:
        data = json.load(f)
    masks = []
    for ann in data.get("annotations", []):
        seg = ann["segmentation"]
        mask = rle_decode(seg)
        masks.append(mask)
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RoofFacetDataset(torch.utils.data.Dataset):
    """
    Loads RGB images + COCO RLE annotations.
    Returns encoding compatible with Mask2FormerForUniversalSegmentation.
    """

    def __init__(self, img_folder: str, gt_folder: str, processor):
        self.img_folder = Path(img_folder)
        self.gt_folder = Path(gt_folder)
        self.processor = processor

        # Match image files to annotation JSON files
        self.samples = []
        for img_path in sorted(self.img_folder.glob("*.jpg")):
            json_path = self.gt_folder / (img_path.stem + ".json")
            if json_path.exists():
                self.samples.append((img_path, json_path))
            else:
                # try same name with .json
                json_path2 = self.gt_folder / (img_path.name.replace(".jpg", ".json")
                                                              .replace(".jpeg", ".json")
                                                              .replace(".png", ".json"))
                if json_path2.exists():
                    self.samples.append((img_path, json_path2))

        print(f"Found {len(self.samples)} image-annotation pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        masks = load_annotations(str(json_path))

        if len(masks) == 0:
            # No annotations — skip with empty instance map
            instance_map = np.zeros((image.height, image.width), dtype=np.int32)
            instance_id_to_semantic_id = {}
        else:
            # Build instance segmentation map: each pixel = instance ID (0 = background)
            instance_map = np.zeros((image.height, image.width), dtype=np.int32)
            instance_id_to_semantic_id = {}
            for i, mask in enumerate(masks):
                instance_id = i + 1  # 1-indexed
                instance_map[mask] = instance_id
                instance_id_to_semantic_id[instance_id] = 0  # class 0 = "roof_facet"

        instance_seg = Image.fromarray(instance_map.astype(np.int32))

        encoding = self.processor(
            images=image,
            segmentation_maps=instance_seg,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(img_folder, gt_folder, output_dir, epochs=50, batch_size=2,
                 lr=5e-5, base_model="facebook/mask2former-swin-small-coco-instance"):

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Single class: roof_facet
    id2label = {0: "roof_facet"}
    label2id = {"roof_facet": 0}

    processor = AutoImageProcessor.from_pretrained(base_model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        base_model,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    dataset = RoofFacetDataset(img_folder, gt_folder, processor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Training for {epochs} epochs, {len(dataset)} samples, batch_size={batch_size}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"  Epoch {epoch+1} step {step}/{len(dataloader)} loss={loss.item():.4f}")

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")

        ckpt = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        print(f"  Saved: {ckpt}")

    print("Training complete.")


def collate_fn(batch):
    """Custom collate to handle variable-length mask tensors."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    # mask_labels and class_labels are lists of tensors (variable length per image)
    mask_labels = [b["mask_labels"] for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(checkpoint, input_path, output_path):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint).to(device).eval()

    image = Image.open(input_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    overlay = np.array(image).copy()
    patches = []
    seg_map = result["segmentation"].cpu().numpy()

    colors = [
        [255, 80, 80], [80, 255, 80], [80, 80, 255],
        [255, 200, 0], [0, 200, 255], [200, 0, 255],
        [255, 120, 0], [0, 255, 120], [120, 0, 255],
    ]

    for i, seg in enumerate(result["segments_info"]):
        mask = (seg_map == seg["id"])
        color = colors[i % len(colors)]
        overlay[mask] = (overlay[mask] * 0.35 + np.array(color) * 0.65).astype(np.uint8)
        label = model.config.id2label.get(seg["label_id"], str(seg["label_id"]))
        patches.append(mpatches.Patch(color=np.array(color)/255,
                                      label=f"facet {i+1}: {label} ({seg['score']:.2f})"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image); ax1.set_title("Input Image"); ax1.axis("off")
    ax2.imshow(overlay); ax2.set_title(f"Roof Facets ({len(result['segments_info'])} detected)")
    ax2.axis("off")
    if patches:
        ax2.legend(handles=patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Detected {len(result['segments_info'])} roof facets")
    for i, seg in enumerate(result["segments_info"]):
        print(f"  facet {i+1}: score={seg['score']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)

    # Train args
    parser.add_argument("--img_folder")
    parser.add_argument("--gt_folder")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--base_model", default="facebook/mask2former-swin-small-coco-instance")

    # Infer args
    parser.add_argument("--checkpoint", help="Path to saved checkpoint dir")
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--output", default="result_facets.png")

    args = parser.parse_args()

    if args.mode == "train":
        assert args.img_folder and args.gt_folder, "--img_folder and --gt_folder required"
        run_training(args.img_folder, args.gt_folder, args.output_dir,
                     args.epochs, args.batch_size, args.lr, args.base_model)
    else:
        assert args.checkpoint and args.input, "--checkpoint and --input required"
        run_inference(args.checkpoint, args.input, args.output)


if __name__ == "__main__":
    main()
