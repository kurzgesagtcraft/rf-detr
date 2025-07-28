import os
import json
import glob
from PIL import Image
from tqdm import tqdm

# 类别映射 (根据用户反馈更新)
# 格式为 "类别名称": id
CATEGORY_MAP = {
    "person": 1,
    "rider": 2,
    "car": 3,
    "bus": 4,
    "truck": 5,
    "bike": 6,
    "bicycle": 6, # "bike" 和 "bicycle" 都映射到 6
    "motor": 7,
    "traffic light": 8,
    "traffic sign": 9,
    "train": 10,
}

# 忽略的类别
IGNORED_CATEGORIES = {
    "area/drivable",
    "lane/road curb",
}

def get_image_size(image_path):
    """使用Pillow获取图片尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Using default size (1280, 720).")
        return 1280, 720
    except Exception as e:
        print(f"Error reading image {image_path}: {e}. Using default size (1280, 720).")
        return 1280, 720

def convert_to_coco(label_dir, image_dir, output_json_path, is_test_set=False):
    """
    将BDD100k格式的标签转换为COCO格式。

    :param label_dir: 输入标签文件夹路径 (e.g., 'dataset/annotations/train_labels')
    :param image_dir: 图像文件夹路径 (e.g., 'dataset/train2017')
    :param output_json_path: 输出COCO格式JSON文件的路径
    :param is_test_set: 如果是测试集，则为True（不包含标注信息）
    """
    coco_output = {
        "info": {
            "description": "BDD100k to COCO format",
            "version": "1.0",
            "year": 2025,
            "date_created": "2025-07-15"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充类别信息
    # 创建一个不包含重复项的类别列表
    processed_categories = {}
    for name, cat_id in CATEGORY_MAP.items():
        if cat_id not in processed_categories:
            processed_categories[cat_id] = name
            coco_output["categories"].append({
                "id": cat_id,
                "name": name,
                "supercategory": name
            })


    image_id_counter = 0
    annotation_id_counter = 0
    
    label_files = glob.glob(os.path.join(label_dir, '*.json'))
    print(f"Found {len(label_files)} label files in {label_dir}")

    for label_file in tqdm(label_files, desc=f"Processing {os.path.basename(label_dir)}"):
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        image_id_counter += 1
        image_name = label_data.get("name")
        if not image_name:
            print(f"Warning: 'name' not found in {label_file}. Skipping.")
            continue
            
        image_filename = f"{image_name}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        
        width, height = get_image_size(image_path)

        image_info = {
            "id": image_id_counter,
            "width": width,
            "height": height,
            "file_name": image_filename,
            "license": 0,
            "date_captured": ""
        }
        coco_output["images"].append(image_info)

        if is_test_set:
            continue

        frames = label_data.get("frames", [])
        if not frames:
            continue
            
        objects = frames[0].get("objects", [])
        for obj in objects:
            category_name = obj.get("category")
            if not category_name or category_name in IGNORED_CATEGORIES:
                continue

            if category_name not in CATEGORY_MAP:
                print(f"Warning: Category '{category_name}' in {label_file} not in CATEGORY_MAP. Skipping object.")
                continue

            category_id = CATEGORY_MAP[category_name]
            
            box2d = obj.get("box2d")
            if not box2d:
                continue

            x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            annotation_id_counter += 1
            annotation_info = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": category_id,
                "bbox": [x1, y1, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "segmentation": [],
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation_info)

    # 对于测试集，annotations字段应该为空，并且categories也应该为空
    if is_test_set:
        coco_output.pop("annotations", None)
        coco_output.pop("categories", None)


    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"Successfully created {output_json_path}")
    print(f"Total images: {len(coco_output.get('images', []))}")
    if not is_test_set:
        print(f"Total annotations: {len(coco_output.get('annotations', []))}")
    print("-" * 30)


if __name__ == '__main__':
    base_dir = 'dataset'
    annotations_dir = os.path.join(base_dir, 'annotations')
    
    # 确保输出目录存在
    os.makedirs(annotations_dir, exist_ok=True)

    # --- 处理训练集 ---
    train_label_dir = os.path.join(annotations_dir, 'train_labels')
    train_image_dir = os.path.join(base_dir, 'train2017')
    train_output_json = os.path.join(annotations_dir, 'instances_train2017.json')
    convert_to_coco(train_label_dir, train_image_dir, train_output_json)

    # --- 处理验证集 ---
    val_label_dir = os.path.join(annotations_dir, 'val_labels')
    val_image_dir = os.path.join(base_dir, 'val2017')
    val_output_json = os.path.join(annotations_dir, 'instances_val2017.json')
    convert_to_coco(val_label_dir, val_image_dir, val_output_json)

    # --- 处理测试集 ---
    test_label_dir = os.path.join(annotations_dir, 'test_labels')
    test_image_dir = os.path.join(base_dir, 'test2017')
    test_output_json = os.path.join(annotations_dir, 'image_info_test-dev2017.json')
    convert_to_coco(test_label_dir, test_image_dir, test_output_json, is_test_set=True)

    print("All conversions finished!")