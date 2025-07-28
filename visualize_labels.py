import json
import os
from PIL import Image, ImageDraw, ImageFont
import tqdm

def visualize_coco_split(annotation_file, image_dir, output_dir):
    """
    Visualizes COCO annotations for a specific split (train/val/test)
    by drawing green bounding boxes and labels on images.
    """
    print(f"\nProcessing split: {os.path.basename(image_dir)}")
    
    # Create output directory if it doesn't exist
    print(f"Ensuring output directory exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    print(f"Loading annotations from {annotation_file}...")
    try:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        print("Annotations loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_file}. Skipping this split.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {annotation_file}. Skipping this split.")
        return

    # Create mappings for easy access
    images = {img['id']: img for img in coco_data.get('images', [])}
    annotations = {}
    # The test split may not have annotations
    if 'annotations' in coco_data:
        for ann in coco_data.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id:
                if img_id not in annotations:
                    annotations[img_id] = []
                annotations[img_id].append(ann)
    
    categories = {}
    if 'categories' in coco_data:
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    # Process each image
    print(f"Processing {len(images)} images...")
    for img_id, img_info in tqdm.tqdm(images.items(), desc=f"Visualizing {os.path.basename(image_dir)}"):
        img_filename = img_info.get('file_name')
        if not img_filename:
            continue
            
        img_path = os.path.join(image_dir, img_filename)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        if img_id in annotations:
            for ann in annotations[img_id]:
                bbox = ann.get('bbox')
                category_id = ann.get('category_id')
                
                if not bbox or category_id is None:
                    continue

                category_name = categories.get(category_id, 'Unknown')
                x, y, w, h = bbox
                
                # Define the box coordinates and draw it in green
                box_coords = [x, y, x + w, y + h]
                draw.rectangle(box_coords, outline="lime", width=3)

                # Prepare and draw the label text
                text = f"{category_name}"
                text_position = (x, y - 12 if y - 12 > 0 else y)
                
                # Add a background to the text for better visibility
                try:
                    text_bbox = draw.textbbox(text_position, text, font=font)
                    draw.rectangle(text_bbox, fill="lime")
                    draw.text(text_position, text, fill='black', font=font)
                except AttributeError: # Fallback for older Pillow versions
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle([text_position[0], text_position[1], text_position[0] + text_size[0], text_position[1] + text_size[1]], fill="lime")
                    draw.text(text_position, text, fill='black', font=font)


        # Save the visualized image
        output_path = os.path.join(output_dir, img_filename)
        try:
            image.save(output_path)
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")

    print(f"\nVisualization for split '{os.path.basename(image_dir)}' complete.")
    print(f"Annotated images have been saved to: {output_dir}")

def main():
    """
    Main function to run visualization for all dataset splits.
    """
    configs = [
        {
            "annotation_file": "dataset/annotations/instances_train2017.json",
            "image_dir": "dataset/train2017",
            "output_dir": "dataset/visual/train"
        },
        {
            "annotation_file": "dataset/annotations/instances_val2017.json",
            "image_dir": "dataset/val2017",
            "output_dir": "dataset/visual/val"
        },
        {
            "annotation_file": "dataset/annotations/image_info_test-dev2017.json",
            "image_dir": "dataset/test2017",
            "output_dir": "dataset/visual/test"
        }
    ]

    for config in configs:
        visualize_coco_split(
            annotation_file=config["annotation_file"],
            image_dir=config["image_dir"],
            output_dir=config["output_dir"]
        )

if __name__ == '__main__':
    main()