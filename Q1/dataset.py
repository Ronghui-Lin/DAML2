import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from torchvision import transforms

class AircraftDataset(Dataset):
    def __init__(self, csv_file, root_dir, split='train', transform=None):
        self.annotations_df = pd.read_csv(csv_file)
        self.annotations_df = self.annotations_df[self.annotations_df['split'] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # unique filenames for this split to determine dataset size
        self.image_filenames = self.annotations_df['filename'].unique().tolist()

        # Map all specific aircraft classes to a single "aircraft" class index (0)
        self.class_to_idx = {"aircraft": 0}
        print(f"AircraftDataset: Loaded {len(self.image_filenames)} images for split '{split}'.")
        print(f"Mapping all classes to index: {self.class_to_idx}")


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and path
        img_name = self.image_filenames[idx]
        # CSV includes split folder (e.g., "train/image1.jpg")
        img_path = os.path.join(self.root_dir, img_name)

        try:
            # Load image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"Could not read image: {img_path}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            print(f"Skipping image {img_path} due to loading error.")
            return None

        # Get annotations for this image
        img_annotations = self.annotations_df[self.annotations_df['filename'] == img_name]

        boxes = []
        labels = []
        for _, row in img_annotations.iterrows():
            # Extract box coordinates
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            boxes.append([xmin, ymin, xmax, ymax])

            # Assign the single class label ("aircraft" -> 0)
            labels.append(self.class_to_idx["aircraft"])

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # Apply transformations
        if self.transform:
             image = self.transform(image)

        return image, target

# Collate Function
def collate_fn(batch):
    # Filter out images that failed to load
    batch = [item for item in batch if item is not None]
    if not batch: # If all items in batch failed
        return None, None

    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images along the batch dimension
    images = torch.stack(images, dim=0)

    # Targets are list of dictionaries
    return images, targets

# Transforms
def get_transform(train):
    # Use ImageNet mean and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_list = []
    transform_list.append(transforms.ToTensor()) # Converts image to tensor and scales pixels to [0, 1]
    transform_list.append(normalize)             # Normalizes image

    return transforms.Compose(transform_list)