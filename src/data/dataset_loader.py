import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


class TuSimpleBezierDataset(Dataset):
    def __init__(self, data_root="tusimple/TUSimple/train_set", split="train",
                 img_size=(720, 1280), transform=None):
        self.data_root = data_root
        self.img_size = img_size
        self.transform = transform
        self.samples = torch.load(os.path.join(
            data_root, "bezier_gt", f"{split}_bezier.pt"
        ))

        # Default image transform (MiT normalization)
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_root, sample["image_path"])
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)
        bezier_ctrl = sample["bezier_ctrl"]  # [num_lanes, num_ctrl, 2]
        
        # Get actual number of control points from data
        num_ctrl_pts = bezier_ctrl.shape[1]  # Should be 6
        
        # Pad to max lanes for batching
        max_lanes = 6
        padded_ctrl = torch.zeros((max_lanes, num_ctrl_pts, 2))
        num_lanes = min(bezier_ctrl.shape[0], max_lanes)
        padded_ctrl[:num_lanes] = bezier_ctrl[:max_lanes]
        
        # Create lane existence labels (1 = lane exists, 0 = no lane)
        lane_exist = torch.zeros(max_lanes)
        lane_exist[:num_lanes] = 1.0

        target = {
            "bezier_ctrl": padded_ctrl,  # [max_lanes, num_ctrl_pts, 2]
            "lane_exist": lane_exist,     # [max_lanes]
            "num_lanes": num_lanes
        }

        return image, target


class TuSimpleBezierTestSet(Dataset):
    def __init__(self, data_root="tusimple/TUSimple/test_set",
                 json_file="test_tasks_0627.json",
                 img_size=(720, 1280), transform=None):
        self.data_root = data_root
        self.img_size = img_size
        self.json_path = os.path.join(data_root, json_file)

        with open(self.json_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]

        # Same normalization as training
        self.transform = transform or T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_root, sample["raw_file"])
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)
        meta = {
            "image_path": sample["raw_file"],
            "h_samples": sample["h_samples"]
        }

        return image, meta


def visualize_sample(dataset, idx=0, image_width=1280, image_height=720):
    img, target = dataset[idx]
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]), 0, 1)

    plt.imshow(img_np)
    for i, ctrl in enumerate(target["bezier_ctrl"]):
        if target["lane_exist"][i] == 0:  # Skip non-existent lanes
            continue
        pts = ctrl.clone()
        pts[:, 0] *= image_width
        pts[:, 1] *= image_height
        plt.plot(pts[:, 0], pts[:, 1], 'ro-', label=f'Lane {i+1}')
    plt.legend()
    plt.title(f"Sample {idx} - {target['num_lanes']} lanes")
    plt.show()

def dataloaders():
    # Full dataset

    full_dataset = TuSimpleBezierDataset(split="train")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return train_loader, val_loader

if __name__ == '__main__':
    # Test training dataset
    full_dataset = TuSimpleBezierDataset(split="train")
    train_loader, val_loader = dataloaders()
    
    # Check shapes
    for img, target in train_loader:
        print(f"Image shape: {img.shape}")           # [B, 3, 720, 1280]
        print(f"Bezier ctrl shape: {target['bezier_ctrl'].shape}")  # [B, 6, num_ctrl, 2]
        print(f"Lane exist shape: {target['lane_exist'].shape}")    # [B, 6]
        print(f"Num control points: {target['bezier_ctrl'].shape[2]}")
        print(f"First sample num_lanes: {target['num_lanes'][0]}")
        break

    # Test test dataset
    test_dataset = TuSimpleBezierTestSet(
        data_root="tusimple/TUSimple/test_set",
        json_file="test_tasks_0627.json",
        img_size=(720, 1280)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for img, meta in test_loader:
        print(f"\nTest image shape: {img.shape}")
        print(f"Meta: {meta}")
        break
    
    # Test visualization
    dataset = TuSimpleBezierDataset(split="train")
    print(f"\nTotal samples: {len(dataset)}")
    visualize_sample(dataset, idx=10)