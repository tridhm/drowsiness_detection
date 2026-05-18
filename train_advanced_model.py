"""
Advanced Drowsiness Detection Model Training
- Combines multiple datasets: CEW, dataset_eyes&yawn, mrleyedataset, dataset_nthuddd2
- Uses segmentation-based eye detection
- Multi-task learning: eye state + yawn detection
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cli_json_config import parse_args_with_json_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train advanced multitask drowsiness model (eye state + yawn).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to shared JSON config file (section: train_advanced_model).",
    )
    parser.add_argument(
        "--cew-dir",
        default="CEW",
        help="Root folder for CEW dataset (expects closed/open subfolders).",
    )
    parser.add_argument(
        "--mrl-dir",
        default="mrleyedataset",
        help="Root folder for MRL dataset (expects Close-Eyes/Open-Eyes subfolders).",
    )
    parser.add_argument(
        "--eyes-yawn-dir",
        default="dataset_eyes&yawn/train",
        help="Root folder for eyes&yawn train split (Closed/Open/yawn/no_yawn).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (0-1).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument(
        "--output-model",
        default="advanced_drowsiness_model.pth",
        help="Output path for best model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    return parser


# ====================== DATASET CLASS ======================
class MultiDataset(Dataset):
    """Unified dataset loader for all eye/yawn datasets"""
    def __init__(self, data_list, transform=None):
        """
        data_list: list of (image_path, label) tuples
        label format: {'eye': 0/1, 'yawn': 0/1} where 0=closed/yawn, 1=open/no_yawn
        """
        self.data = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Return image and labels
            return image, label['eye'], label['yawn']
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image
            dummy = torch.zeros(3, 64, 64)
            return dummy, 1, 1  # default: open, no_yawn


# ====================== DATA LOADER ======================
def _collect_images(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return [str(path) for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def load_all_datasets(cew_dir: str, mrl_dir: str, eyes_yawn_dir: str):
    """Load and combine all available datasets"""
    data_list = []
    
    print("📂 Loading datasets...")
    
    # 1. CEW Dataset (Closed Eyes in the Wild)
    print("  - CEW dataset...")
    cew_root = Path(cew_dir)
    cew_closed = _collect_images(cew_root / "closed")
    cew_open = _collect_images(cew_root / "open")
    
    for img in cew_closed:
        data_list.append((img, {'eye': 0, 'yawn': 1}))  # closed, no_yawn
    for img in cew_open:
        data_list.append((img, {'eye': 1, 'yawn': 1}))  # open, no_yawn
    print(f"    CEW: {len(cew_closed)} closed, {len(cew_open)} open")
    
    # 2. MRL Eye Dataset
    print("  - MRL Eye dataset...")
    mrl_root = Path(mrl_dir)
    mrl_closed = _collect_images(mrl_root / "Close-Eyes")
    mrl_open = _collect_images(mrl_root / "Open-Eyes")
    
    for img in mrl_closed:
        data_list.append((img, {'eye': 0, 'yawn': 1}))
    for img in mrl_open:
        data_list.append((img, {'eye': 1, 'yawn': 1}))
    print(f"    MRL: {len(mrl_closed)} closed, {len(mrl_open)} open")
    
    # 3. Dataset Eyes & Yawn (Training set)
    print("  - Eyes & Yawn dataset...")
    ey_root = Path(eyes_yawn_dir)
    ey_train_closed = _collect_images(ey_root / "Closed")
    ey_train_open = _collect_images(ey_root / "Open")
    ey_train_yawn = _collect_images(ey_root / "yawn")
    ey_train_no_yawn = _collect_images(ey_root / "no_yawn")
    
    for img in ey_train_closed:
        data_list.append((img, {'eye': 0, 'yawn': 1}))
    for img in ey_train_open:
        data_list.append((img, {'eye': 1, 'yawn': 1}))
    for img in ey_train_yawn:
        data_list.append((img, {'eye': 1, 'yawn': 0}))  # assume open when yawning
    for img in ey_train_no_yawn:
        data_list.append((img, {'eye': 1, 'yawn': 1}))
    
    print(f"    Eyes&Yawn: {len(ey_train_closed)} closed, {len(ey_train_open)} open")
    print(f"               {len(ey_train_yawn)} yawn, {len(ey_train_no_yawn)} no_yawn")
    
    print(f"\n✅ Total samples: {len(data_list)}")
    return data_list

# ====================== ADVANCED MODEL ======================
class AdvancedDrowsinessModel(nn.Module):
    """
    Multi-task model with:
    - Segmentation-inspired feature extraction
    - Eye state classification
    - Yawn detection
    """
    def __init__(self):
        super(AdvancedDrowsinessModel, self).__init__()
        
        # Encoder (Feature Extraction with Segmentation-like structure)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Classification heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Eye state classifier
        self.eye_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Closed/Open
        )
        
        # Yawn detector
        self.yawn_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Yawn/No_Yawn
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        x = self.pool3(x3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Global pooling
        features = self.global_pool(x)
        
        # Classification
        eye_out = self.eye_classifier(features)
        yawn_out = self.yawn_classifier(features)
        
        return eye_out, yawn_out

# ====================== TRAINING FUNCTION ======================
def train_model(args: argparse.Namespace):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    torch.manual_seed(args.seed)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load data
    data_list = load_all_datasets(
        cew_dir=args.cew_dir,
        mrl_dir=args.mrl_dir,
        eyes_yawn_dir=args.eyes_yawn_dir,
    )
    if not data_list:
        raise RuntimeError("No images found. Check dataset paths and folder structure.")
    
    # Split data
    train_data, val_data = train_test_split(
        data_list,
        test_size=args.val_split,
        random_state=args.seed,
    )
    
    train_dataset = MultiDataset(train_data, transform=transform)
    val_dataset = MultiDataset(val_data, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Model
    model = AdvancedDrowsinessModel().to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = args.epochs
    best_val_loss = float('inf')
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_eye_correct = 0
        train_yawn_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, eye_labels, yawn_labels in pbar:
            images = images.to(device)
            eye_labels = eye_labels.to(device)
            yawn_labels = yawn_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            eye_out, yawn_out = model(images)
            
            # Loss
            loss_eye = criterion(eye_out, eye_labels)
            loss_yawn = criterion(yawn_out, yawn_labels)
            loss = loss_eye + loss_yawn
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            _, eye_pred = torch.max(eye_out, 1)
            _, yawn_pred = torch.max(yawn_out, 1)
            train_eye_correct += (eye_pred == eye_labels).sum().item()
            train_yawn_correct += (yawn_pred == yawn_labels).sum().item()
            train_total += images.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_eye_acc = 100 * train_eye_correct / train_total
        train_yawn_acc = 100 * train_yawn_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_eye_correct = 0
        val_yawn_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, eye_labels, yawn_labels in val_loader:
                images = images.to(device)
                eye_labels = eye_labels.to(device)
                yawn_labels = yawn_labels.to(device)
                
                eye_out, yawn_out = model(images)
                
                loss_eye = criterion(eye_out, eye_labels)
                loss_yawn = criterion(yawn_out, yawn_labels)
                loss = loss_eye + loss_yawn
                
                val_loss += loss.item()
                _, eye_pred = torch.max(eye_out, 1)
                _, yawn_pred = torch.max(yawn_out, 1)
                val_eye_correct += (eye_pred == eye_labels).sum().item()
                val_yawn_correct += (yawn_pred == yawn_labels).sum().item()
                val_total += images.size(0)
        
        val_loss /= len(val_loader)
        val_eye_acc = 100 * val_eye_correct / val_total
        val_yawn_acc = 100 * val_yawn_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Eye Acc: {train_eye_acc:.2f}% | Yawn Acc: {train_yawn_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Eye Acc: {val_eye_acc:.2f}% | Yawn Acc: {val_yawn_acc:.2f}%")
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(output_model))
            print(f"  ✅ Best model saved!")
        
        print()
    
    print("🎉 Training completed!")
    print(f"📦 Model saved as: {output_model}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_json_config(parser, argv, section="train_advanced_model")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1 (exclusive).")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0.")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")
    train_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
