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
    
    # Post-process to get instance masks
    target_size = (image.height, image.width)
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[target_size])[0]
    
    # Get masks and scores
    masks = results["segmentation"].cpu().numpy()  # [N, H, W]
    # Some models don't return scores, use all masks
    if "scores" in results:
        scores = results["scores"].cpu().numpy()  # [N]
    else:
        scores = np.ones(len(masks))  # Assume all masks are valid
    
    # Create visualization
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image, "RGBA")
    
    # Generate colors for each mask
    colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128), 
              (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128)]
    
    for i in range(len(masks)):
        mask = masks[i]  # Binary mask [H, W]
        score = scores[i]
        if score < 0.5:  # Skip low-confidence predictions
            continue
        color = colors[i % len(colors)]
        
        # Create overlay for this mask
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Fill mask regions
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        overlay_draw.bitmap((0, 0), mask_img, fill=color)
        
        # Composite onto original image
        vis_image = Image.alpha_composite(vis_image.convert('RGBA'), overlay).convert('RGB')
    
    # Save with original filename
    output_path = Path(output_dir) / f"{image_name}_masks.jpg"
    vis_image.save(output_path)
    print(f"Saved: {output_path}")
    
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
