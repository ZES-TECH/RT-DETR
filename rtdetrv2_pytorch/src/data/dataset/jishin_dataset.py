import torch
import os
import yaml
import glob
from PIL import Image
from typing import Optional, Callable

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

@register()
class JishinDetection(DetDataset):
    __inject__ = ['transforms']

    def __init__(self, yaml_file: str, dataset_type: str = "train", transforms: Optional[Callable] = None):
        """
        Args:
            yaml_file (str): Path to Jishin dataset YAML file (Ultralytics YOLO format).
            dataset_type (str): Type of dataset ('train', 'val', or 'test').
            transforms (Optional[Callable]): Transformations to apply to the images.
        """
        assert dataset_type in ["train", "val", "test"], "dataset_type must be 'train', 'val', or 'test'"

        with open(yaml_file, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.image_dirs = data_config[dataset_type]
        self.labels_map = {int(k): v for k, v in data_config['names'].items()}
        self.transforms = transforms

        # Collect all image and label file paths
        self.images = []
        self.targets = []
        for img_dir in self.image_dirs:
            image_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
            for img_path in image_files:
                label_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.targets.append(label_path)
        
        assert len(self.images) == len(self.targets), "Mismatch between images and labels. Check dataset structure."

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)
        return image, target

    def load_item(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        label_path = self.targets[index]

        output = {
            "image_id": torch.tensor([index]),
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": []
        }

        w, h = image.size
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                
                # Convert YOLO (x_center, y_center, width, height) to Pascal VOC (x_min, y_min, x_max, y_max)
                x_min = (xc - bw / 2) * w
                y_min = (yc - bh / 2) * h
                x_max = (xc + bw / 2) * w
                y_max = (yc + bh / 2) * h

                output["boxes"].append([x_min, y_min, x_max, y_max])
                output["labels"].append(class_id)
                output["area"].append((x_max - x_min) * (y_max - y_min))
                output["iscrowd"].append(0)

        boxes = torch.tensor(output["boxes"]) if len(output["boxes"]) > 0 else torch.zeros((0, 4))
        output['boxes'] = convert_to_tv_tensor(boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])
        output['labels'] = torch.tensor(output['labels'], dtype=torch.long)
        output['area'] = torch.tensor(output['area'], dtype=torch.float32)
        output['iscrowd'] = torch.tensor(output['iscrowd'], dtype=torch.uint8)
        output['orig_size'] = torch.tensor([w, h], dtype=torch.int32)
        
        return image, output

    def __len__(self):
        return len(self.images)