"""
Simple Mask2Former inference that saves mask visualizations with original filenames.
No polygon conversion - just displays the instance masks.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import argparse
from pathlib import Path


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path).to(device)
    model.eval()
    return processor, model, device


def run_inference(image_path, checkpoint_path, output_dir):
    processor, model, device = load_model(checkpoint_path)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_name = Path(image_path).stem
    
    # Process
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process to get instance segmentation map
    target_size = (image.height, image.width)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[target_size], threshold=0.5)[0]
    
    # segmentation is a label map [H, W] where each pixel = instance_id
    seg_map = results["segmentation"].cpu().numpy()
    segments_info = results["segments_info"]
    
    print(f"  {image_name}: found {len(segments_info)} instances")
    
    # Create RGBA overlay
    img_rgba = image.convert("RGBA")
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    
    # Generate distinct colors per instance
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(max(len(segments_info), 1), 3))
    
    for idx, seg_info in enumerate(segments_info):
        inst_id = seg_info["id"]
        mask = (seg_map == inst_id)
        if mask.sum() == 0:
            continue
        r, g, b = colors[idx % len(colors)]
        overlay[mask] = [r, g, b, 140]  # Semi-transparent
    
    # Composite overlay onto original image
    overlay_img = Image.fromarray(overlay, "RGBA")
    vis_image = Image.alpha_composite(img_rgba, overlay_img).convert("RGB")
    
    # Save with original filename
    output_path = Path(output_dir) / f"{image_name}_masks.jpg"
    vis_image.save(output_path)
    print(f"  Saved: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    input_path = Path(args.input)
    if input_path.is_file():
        run_inference(str(input_path), args.checkpoint, args.output)
    elif input_path.is_dir():
        for img_path in sorted(input_path.glob('*.jpg')):
            run_inference(str(img_path), args.checkpoint, args.output)


if __name__ == '__main__':
    main()
