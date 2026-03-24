"""
postprocess_masks.py

Takes a fine-tuned Mask2Former checkpoint, runs inference on an image,
and converts instance masks → clean polygons using Douglas-Peucker approximation.

Outputs:
  - Visualization PNG with colored polygon overlays + edges
  - JSON file with polygon coordinates per instance

Usage:
    python notebooks/postprocess_masks.py \
        --checkpoint ./checkpoints/epoch_50 \
        --input /path/to/image.jpg \
        --output /path/to/output_dir \
        --epsilon 2.0 \
        --min_area 500 \
        --score_threshold 0.5
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch


def load_model(checkpoint):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, processor, device


def run_inference(model, processor, device, image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.0,
        target_sizes=[image.size[::-1]],
    )[0]
    return results


def mask_to_polygon(binary_mask, epsilon=2.0, min_area=500):
    """
    Convert a binary mask to a simplified polygon using Douglas-Peucker.
    Returns list of polygons (each polygon is a numpy array of shape [N, 2]).
    """
    mask_u8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        # Douglas-Peucker simplification
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri / 1000.0, True)
        if len(approx) < 3:
            continue
        poly = approx.reshape(-1, 2)  # [N, 2] (x, y)
        polygons.append(poly)

    return polygons


def visualize_polygons(image, instance_polygons, scores):
    """
    Draw filled semi-transparent polygons + crisp edges on the image.
    Returns annotated PIL image.
    """
    img_np = np.array(image).copy()
    overlay = img_np.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0),
    ]

    for idx, (polygons, score) in enumerate(zip(instance_polygons, scores)):
        color = colors[idx % len(colors)]
        for poly in polygons:
            pts = poly.reshape(-1, 1, 2)
            # filled
            cv2.fillPoly(overlay, [pts], color)
            # crisp edges
            cv2.polylines(img_np, [pts], isClosed=True, color=color, thickness=2)
            # vertex dots
            for pt in poly:
                cv2.circle(img_np, tuple(pt), 3, (255, 255, 255), -1)

        # label score near centroid of first polygon
        if polygons:
            cx = int(polygons[0][:, 0].mean())
            cy = int(polygons[0][:, 1].mean())
            cv2.putText(img_np, f"{score:.2f}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # blend filled overlay
    blended = cv2.addWeighted(overlay, 0.35, img_np, 0.65, 0)
    # re-draw crisp edges on top of blend
    for idx, polygons in enumerate(instance_polygons):
        color = colors[idx % len(colors)]
        for poly in polygons:
            pts = poly.reshape(-1, 1, 2)
            cv2.polylines(blended, [pts], isClosed=True, color=color, thickness=2)
            for pt in poly:
                cv2.circle(blended, tuple(pt), 3, (255, 255, 255), -1)

    return Image.fromarray(blended)


def polygons_to_json(instance_polygons, scores, image_path):
    results = {
        "image": os.path.basename(image_path),
        "instances": []
    }
    for idx, (polygons, score) in enumerate(zip(instance_polygons, scores)):
        inst = {
            "instance_id": idx + 1,
            "score": round(float(score), 4),
            "polygons": [poly.tolist() for poly in polygons]
        }
        results["instances"].append(inst)
    return results


def postprocess(checkpoint, input_path, output_dir, epsilon=2.0,
                min_area=500, score_threshold=0.5):

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(input_path).convert("RGB")
    print(f"Image size: {image.size}")

    print("Loading model...")
    model, processor, device = load_model(checkpoint)

    print("Running inference...")
    results = run_inference(model, processor, device, image)

    segments = results["segments_info"]
    masks_tensor = results["segmentation"]  # H x W, each pixel = segment id or 0

    print(f"Found {len(segments)} instances (before score filter)")

    instance_polygons = []
    instance_scores = []

    for seg in segments:
        score = seg.get("score", 1.0)
        if score < score_threshold:
            continue
        seg_id = seg["id"]
        binary_mask = (masks_tensor.cpu().numpy() == seg_id)
        polygons = mask_to_polygon(binary_mask, epsilon=epsilon, min_area=min_area)
        if not polygons:
            continue
        instance_polygons.append(polygons)
        instance_scores.append(score)

    print(f"After filtering: {len(instance_polygons)} instances")

    # Visualization
    vis = visualize_polygons(image, instance_polygons, instance_scores)
    vis_path = os.path.join(output_dir, "polygons_viz.png")
    vis.save(vis_path)
    print(f"Saved visualization: {vis_path}")

    # JSON
    data = polygons_to_json(instance_polygons, instance_scores, input_path)
    json_path = os.path.join(output_dir, "polygons.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved polygons JSON: {json_path}")
    print(f"  {len(data['instances'])} roof facet polygons saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint dir")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epsilon", type=float, default=2.0,
                        help="Douglas-Peucker epsilon (higher = more simplified, default=2.0)")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Minimum mask area in pixels to keep (default=500)")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Minimum instance score to keep (default=0.5)")
    args = parser.parse_args()

    postprocess(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.output,
        epsilon=args.epsilon,
        min_area=args.min_area,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
